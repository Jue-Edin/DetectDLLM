import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_utils import load_human_corpus, save_dataset_split, split_records, summarize_lengths
from src.duo_adapter import DuoAdapter
from src.utils import configure_logging, ensure_dir, load_json, stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local human-vs-DUO dataset.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--human-path", default=None)
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--allow-online-tokenizer", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-tag", default="default")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generation-steps", type=int, default=None)
    parser.add_argument("--generation-strategy", default=None, choices=["top_p", "greedy"])
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--prompt-fraction", type=float, default=None)
    parser.add_argument("--unconditional-only", action="store_true")
    return parser.parse_args()


def _resolve_setting(args: argparse.Namespace, config: dict[str, Any], key: str, default: Any = None) -> Any:
    value = getattr(args, key.replace("-", "_"), None)
    if value is not None:
        return value
    return config.get(key, default)


def run_build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    configure_logging()
    checkpoint_dir = _resolve_setting(args, config, "checkpoint_dir", "models/duo-distilled")
    tokenizer_dir = _resolve_setting(args, config, "tokenizer_dir", "assets/tokenizer/gpt2")
    allow_online_tokenizer = bool(_resolve_setting(args, config, "allow_online_tokenizer", False))
    if args.allow_online_tokenizer:
        allow_online_tokenizer = True
    human_path = _resolve_setting(args, config, "human_path", "data/human")
    if args.human_path is not None:
        human_path = args.human_path
    output_root = Path(_resolve_setting(args, config, "output_root", "outputs"))
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "datasets" / args.dataset_tag
    resolved_generation_steps = _resolve_setting(args, config, "generation_steps", 8)
    if isinstance(resolved_generation_steps, list):
        resolved_generation_steps = resolved_generation_steps[0]
    generation_steps = int(resolved_generation_steps)
    generation_strategy = _resolve_setting(args, config, "generation_strategy", "top_p")
    generation_batch_size = int(_resolve_setting(args, config, "generation_batch_size", 1))
    temperature = float(_resolve_setting(args, config, "generation_temperature", 1.0))
    top_p = float(_resolve_setting(args, config, "generation_top_p", 0.9))
    prompt_fraction = float(_resolve_setting(args, config, "prompt_fraction", 0.25))
    seed = int(_resolve_setting(args, config, "seed", 7))

    adapter = DuoAdapter(
        checkpoint_dir=checkpoint_dir,
        tokenizer_dir=tokenizer_dir,
        allow_online_tokenizer=allow_online_tokenizer,
    ).load(require_tokenizer=True)
    tokenizer = adapter.tokenizer
    assert tokenizer is not None
    records = load_human_corpus(human_path, text_field=args.text_field)
    split_map = split_records(records, seed=seed)
    ensure_dir(output_dir)
    manifest = {
        "dataset_tag": args.dataset_tag,
        "human_path": str(human_path),
        "checkpoint_dir": str(checkpoint_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "tokenizer_load_report": adapter.tokenizer_load_report,
        "tokenizer_vocab_audit": adapter.vocab_audit,
        "generation_steps": generation_steps,
        "generation_strategy": generation_strategy,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_fraction": prompt_fraction,
        "approximation_mode": "iterative_masked_denoising_generation",
        "splits": {},
    }

    for split_name, split_records_for_name in split_map.items():
        enriched_records = []
        lengths = []
        prompt_texts = []
        for record in split_records_for_name:
            token_ids = tokenizer.encode(
                record["text"],
                add_special_tokens=False,
                truncation=True,
                max_length=adapter.max_length,
            )
            if not token_ids:
                continue
            token_length = len(token_ids)
            if args.unconditional_only or token_length <= 1:
                prompt_len = 0
                prompt_text = ""
            else:
                prompt_len = max(1, min(token_length - 1, int(round(token_length * prompt_fraction))))
                prompt_text = tokenizer.decode(token_ids[:prompt_len], skip_special_tokens=True)
            example_id = stable_hash({"source_id": record["source_id"], "split": split_name, "label": "human"})
            machine_id = stable_hash({"source_id": record["source_id"], "split": split_name, "label": "machine"})
            enriched_records.append(
                {
                    "source_id": record["source_id"],
                    "text": record["text"],
                    "token_ids": token_ids,
                    "token_length": token_length,
                    "prompt_length": prompt_len,
                    "prompt_text": prompt_text,
                    "human_example_id": example_id,
                    "machine_example_id": machine_id,
                }
            )
            lengths.append(token_length)
            prompt_texts.append(prompt_text)

        if enriched_records:
            generated_texts: list[str] = []
            generated_token_ids: list[list[int]] = []
            for start_idx in range(0, len(enriched_records), generation_batch_size):
                batch_records = enriched_records[start_idx : start_idx + generation_batch_size]
                batch_prompts = None if args.unconditional_only else prompt_texts[start_idx : start_idx + generation_batch_size]
                batch_generated = adapter.generate_texts(
                    target_lengths=[item["token_length"] for item in batch_records],
                    prompt_texts=batch_prompts,
                    num_steps=generation_steps,
                    strategy=generation_strategy,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed + start_idx,
                )
                generated_texts.extend(batch_generated["texts"])
                generated_token_ids.extend(batch_generated["token_ids"])
            generated = {"texts": generated_texts, "token_ids": generated_token_ids}
        else:
            generated = {"texts": [], "token_ids": []}

        original_rows = [item["text"] for item in enriched_records]
        sampled_rows = generated["texts"]
        original_records = []
        sampled_records = []
        for idx, item in enumerate(enriched_records):
            original_records.append(
                {
                    "example_id": item["human_example_id"],
                    "source_id": item["source_id"],
                    "label": "human",
                    "split": split_name,
                    "text": item["text"],
                    "token_ids": item["token_ids"],
                    "token_length": item["token_length"],
                    "prompt_length": item["prompt_length"],
                }
            )
            sampled_records.append(
                {
                    "example_id": item["machine_example_id"],
                    "source_id": item["source_id"],
                    "label": "machine",
                    "split": split_name,
                    "text": generated["texts"][idx],
                    "token_ids": generated["token_ids"][idx],
                    "token_length": len(generated["token_ids"][idx]),
                    "prompt_length": item["prompt_length"],
                    "generation_steps": generation_steps,
                    "generation_strategy": generation_strategy,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )

        metadata = {
            "split": split_name,
            "approximation_mode": "iterative_masked_denoising_generation",
            "length_summary": summarize_lengths(lengths),
            "original_records": original_records,
            "sampled_records": sampled_records,
        }
        dataset_path, metadata_path = save_dataset_split(
            split_name=split_name,
            original_rows=original_rows,
            sampled_rows=sampled_rows,
            output_dir=output_dir,
            metadata=metadata,
        )
        manifest["splits"][split_name] = {
            "dataset_path": str(dataset_path),
            "metadata_path": str(metadata_path),
            "count": len(original_rows),
        }
    manifest_path = output_dir / "manifest.json"
    from src.utils import dump_json

    dump_json(manifest, manifest_path)
    return {"output_dir": str(output_dir), "manifest_path": str(manifest_path)}


def main() -> None:
    args = parse_args()
    result = run_build_dataset(args)
    print(f"Dataset build complete: {result['output_dir']}")


if __name__ == "__main__":
    main()
