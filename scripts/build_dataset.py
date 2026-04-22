import argparse
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_utils import save_dataset_split, split_records, summarize_lengths, load_human_corpus
from src.duo_adapter import DuoAdapter
from src.utils import configure_logging, dump_json, ensure_dir, load_json, stable_hash


SPLIT_ORDER = ("train", "val", "test")
SPLIT_TARGET_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local human-vs-DUO dataset.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--human-path", default=None)
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--allow-online-tokenizer", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-tag", default=None)
    parser.add_argument("--data-export-root", default=None)
    parser.add_argument("--no-data-export", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generation-steps", type=int, default=None)
    parser.add_argument("--generation-strategy", default=None, choices=["top_p", "greedy"])
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--prompt-fraction", type=float, default=None)
    parser.add_argument("--prompt-tokens", type=int, default=None)
    parser.add_argument("--target-total-tokens", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--max-windows-per-source", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--unconditional-only", action="store_true")
    return parser.parse_args()


def _resolve_setting(args: argparse.Namespace, config: dict[str, Any], key: str, default: Any = None) -> Any:
    value = getattr(args, key.replace("-", "_"), None)
    if value is not None:
        return value
    return config.get(key, default)


def _compute_prompt_length(
    token_length: int,
    *,
    unconditional_only: bool,
    prompt_tokens: int | None,
    prompt_fraction: float,
) -> int:
    if unconditional_only or token_length <= 1:
        return 0
    if prompt_tokens is not None:
        if prompt_tokens < 0:
            raise ValueError(f"prompt_tokens must be >= 0, got {prompt_tokens}.")
        if prompt_tokens == 0:
            return 0
        return max(1, min(token_length - 1, int(prompt_tokens)))
    return max(1, min(token_length - 1, int(round(token_length * prompt_fraction))))


def _build_windows(
    token_ids: list[int],
    *,
    target_total_tokens: int | None,
    window_stride: int | None,
    max_windows_per_source: int | None,
) -> list[tuple[int, int, list[int]]]:
    if target_total_tokens is None:
        return [(0, len(token_ids), token_ids)]
    if target_total_tokens <= 1:
        raise ValueError(f"target_total_tokens must be >= 2, got {target_total_tokens}.")
    if len(token_ids) < target_total_tokens:
        return []
    stride = target_total_tokens if window_stride is None else window_stride
    if stride <= 0:
        raise ValueError(f"window_stride must be >= 1, got {stride}.")
    max_windows = None if max_windows_per_source is None else int(max_windows_per_source)
    if max_windows is not None and max_windows <= 0:
        raise ValueError(f"max_windows_per_source must be >= 1, got {max_windows}.")
    starts = list(range(0, len(token_ids) - target_total_tokens + 1, stride))
    windows: list[tuple[int, int, list[int]]] = []
    for start in starts:
        end = start + target_total_tokens
        windows.append((start, end, token_ids[start:end]))
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


def _build_candidate_examples(
    *,
    records: list[dict[str, Any]],
    split_name: str,
    tokenizer,
    max_length: int | None,
    prompt_tokens: int | None,
    prompt_fraction: float,
    target_total_tokens: int | None,
    window_stride: int | None,
    max_windows_per_source: int | None,
    unconditional_only: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    stats = {
        "source_records_seen": len(records),
        "source_records_with_tokens": 0,
        "source_records_too_short": 0,
        "candidate_windows": 0,
    }
    for record in records:
        token_ids = tokenizer.encode(
            record["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        if not token_ids:
            continue
        stats["source_records_with_tokens"] += 1
        windows = _build_windows(
            token_ids,
            target_total_tokens=target_total_tokens,
            window_stride=window_stride,
            max_windows_per_source=max_windows_per_source,
        )
        if not windows:
            stats["source_records_too_short"] += 1
            continue
        for segment_index, (segment_start, segment_end, segment_ids) in enumerate(windows):
            token_length = len(segment_ids)
            prompt_len = _compute_prompt_length(
                token_length,
                unconditional_only=unconditional_only,
                prompt_tokens=prompt_tokens,
                prompt_fraction=prompt_fraction,
            )
            prompt_token_ids = segment_ids[:prompt_len]
            continuation_token_ids = segment_ids[prompt_len:]
            prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
            continuation_text = tokenizer.decode(continuation_token_ids, skip_special_tokens=True)
            text = tokenizer.decode(segment_ids, skip_special_tokens=True)
            example_signature = {
                "source_id": record["source_id"],
                "split": split_name,
                "segment_index": segment_index,
                "segment_start": segment_start,
                "segment_end": segment_end,
                "seed": seed,
            }
            candidates.append(
                {
                    "source_id": record["source_id"],
                    "split": split_name,
                    "text": text,
                    "token_ids": segment_ids,
                    "token_length": token_length,
                    "prompt_length": prompt_len,
                    "prompt_text": prompt_text,
                    "prompt_token_ids": prompt_token_ids,
                    "continuation_text": continuation_text,
                    "continuation_token_ids": continuation_token_ids,
                    "segment_index": segment_index,
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "source_token_length": len(token_ids),
                    "selection_rank": stable_hash(example_signature),
                    "human_example_id": stable_hash({**example_signature, "label": "human"}),
                    "machine_example_id": stable_hash({**example_signature, "label": "machine"}),
                }
            )
            stats["candidate_windows"] += 1
    return candidates, stats


def _allocate_counts_by_split(
    candidates_by_split: dict[str, list[dict[str, Any]]],
    max_examples: int,
) -> dict[str, int]:
    if max_examples <= 0:
        raise ValueError(f"max_examples must be >= 1, got {max_examples}.")
    available_total = sum(len(rows) for rows in candidates_by_split.values())
    if available_total <= max_examples:
        return {split_name: len(candidates_by_split[split_name]) for split_name in SPLIT_ORDER}

    desired = {split_name: int(math.floor(max_examples * SPLIT_TARGET_RATIOS[split_name])) for split_name in SPLIT_ORDER}
    remainder = max_examples - sum(desired.values())
    for idx in range(remainder):
        desired[SPLIT_ORDER[idx % len(SPLIT_ORDER)]] += 1

    allocated = {
        split_name: min(len(candidates_by_split[split_name]), desired[split_name])
        for split_name in SPLIT_ORDER
    }
    remaining = max_examples - sum(allocated.values())
    while remaining > 0:
        eligible = [split_name for split_name in SPLIT_ORDER if allocated[split_name] < len(candidates_by_split[split_name])]
        if not eligible:
            break
        eligible.sort(
            key=lambda split_name: (
                len(candidates_by_split[split_name]) - allocated[split_name],
                SPLIT_TARGET_RATIOS[split_name],
                -SPLIT_ORDER.index(split_name),
            ),
            reverse=True,
        )
        chosen = eligible[0]
        allocated[chosen] += 1
        remaining -= 1
    return allocated


def _select_candidates(
    candidates_by_split: dict[str, list[dict[str, Any]]],
    *,
    max_examples: int | None,
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    if max_examples is None:
        for split_name in SPLIT_ORDER:
            selected[split_name] = sorted(candidates_by_split[split_name], key=lambda row: row["selection_rank"])
        return selected

    counts = _allocate_counts_by_split(candidates_by_split, max_examples=max_examples)
    for split_name in SPLIT_ORDER:
        ordered = sorted(candidates_by_split[split_name], key=lambda row: row["selection_rank"])
        selected[split_name] = ordered[: counts[split_name]]
    return selected


def _simplify_human_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": record["example_id"],
        "source_id": record["source_id"],
        "split": record["split"],
        "prompt_length": record["prompt_length"],
        "prompt_text": record["prompt_text"],
        "token_length": record["token_length"],
        "segment_index": record["segment_index"],
        "segment_start": record["segment_start"],
        "segment_end": record["segment_end"],
        "text": record["text"],
    }


def _simplify_machine_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": record["example_id"],
        "source_id": record["source_id"],
        "split": record["split"],
        "prompt_length": record["prompt_length"],
        "prompt_text": record["prompt_text"],
        "token_length": record["token_length"],
        "segment_index": record["segment_index"],
        "segment_start": record["segment_start"],
        "segment_end": record["segment_end"],
        "generation_steps": record.get("generation_steps"),
        "generation_strategy": record.get("generation_strategy"),
        "temperature": record.get("temperature"),
        "top_p": record.get("top_p"),
        "text": record["text"],
    }


def _build_paired_record(human_record: dict[str, Any], machine_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": stable_hash(
            {
                "human_example_id": human_record["example_id"],
                "machine_example_id": machine_record["example_id"],
            }
        ),
        "split": human_record["split"],
        "source_id": human_record["source_id"],
        "prompt_length": human_record["prompt_length"],
        "prompt_text": human_record["prompt_text"],
        "segment_index": human_record["segment_index"],
        "segment_start": human_record["segment_start"],
        "segment_end": human_record["segment_end"],
        "human_example_id": human_record["example_id"],
        "machine_example_id": machine_record["example_id"],
        "human_token_length": human_record["token_length"],
        "machine_token_length": machine_record["token_length"],
        "human_text": human_record["text"],
        "machine_text": machine_record["text"],
    }


def _write_readable_export_readme(
    export_dir: Path,
    *,
    dataset_tag: str,
    source_dataset_dir: Path,
    human_path: str | Path,
    prompt_mode: str,
) -> None:
    content = f"""# Generated data view: {dataset_tag}

This folder is a readable export of the local DetectDLLM dataset.
It is created automatically by `scripts/build_dataset.py`.

- source human corpus: `{human_path}`
- source dataset artifact directory: `{source_dataset_dir}`
- prompt mode: `{prompt_mode}`

Files in this folder:

- `human_train.json`, `human_val.json`, `human_test.json`:
  simplified human records for inspection.
- `machine_train.json`, `machine_val.json`, `machine_test.json`:
  simplified DUO-generated records for inspection.
- `paired_train.json`, `paired_val.json`, `paired_test.json`:
  one human/machine pair per row, with the shared prompt shown explicitly.
- `human_all.json`, `machine_all.json`, `paired_all.json`:
  the same views merged across splits.
- `pair_preview.json`:
  a short preview of prompt/human/machine pairs.
- `manifest.json`:
  paths and counts for this readable export.

This directory is meant for human comparison and GitHub browsing.
The full machine-readable experiment metadata still lives under `outputs/...`.
"""
    export_dir.joinpath("README.md").write_text(content, encoding="utf-8")


def _export_readable_dataset_views(
    *,
    dataset_tag: str,
    export_root: str | Path,
    human_records_by_split: dict[str, list[dict[str, Any]]],
    machine_records_by_split: dict[str, list[dict[str, Any]]],
    preview_rows: list[dict[str, Any]],
    source_dataset_dir: str | Path,
    human_path: str | Path,
    prompt_mode: str,
) -> dict[str, Any]:
    export_dir = ensure_dir(Path(export_root) / dataset_tag)
    manifest: dict[str, Any] = {
        "dataset_tag": dataset_tag,
        "source_dataset_dir": str(source_dataset_dir),
        "human_path": str(human_path),
        "prompt_mode": prompt_mode,
        "splits": {},
    }

    all_human: list[dict[str, Any]] = []
    all_machine: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []

    for split_name in SPLIT_ORDER:
        raw_human_records = human_records_by_split.get(split_name, [])
        raw_machine_records = machine_records_by_split.get(split_name, [])
        human_rows = [_simplify_human_record(record) for record in raw_human_records]
        machine_rows = [_simplify_machine_record(record) for record in raw_machine_records]
        paired_rows = [
            _build_paired_record(human_record, machine_record)
            for human_record, machine_record in zip(raw_human_records, raw_machine_records)
        ]

        human_path_out = export_dir / f"human_{split_name}.json"
        machine_path_out = export_dir / f"machine_{split_name}.json"
        paired_path_out = export_dir / f"paired_{split_name}.json"
        dump_json(human_rows, human_path_out)
        dump_json(machine_rows, machine_path_out)
        dump_json(paired_rows, paired_path_out)

        manifest["splits"][split_name] = {
            "human_path": str(human_path_out),
            "machine_path": str(machine_path_out),
            "paired_path": str(paired_path_out),
            "count": len(paired_rows),
        }
        all_human.extend(human_rows)
        all_machine.extend(machine_rows)
        all_pairs.extend(paired_rows)

    human_all_path = export_dir / "human_all.json"
    machine_all_path = export_dir / "machine_all.json"
    paired_all_path = export_dir / "paired_all.json"
    preview_path = export_dir / "pair_preview.json"
    dump_json(all_human, human_all_path)
    dump_json(all_machine, machine_all_path)
    dump_json(all_pairs, paired_all_path)
    dump_json(preview_rows, preview_path)

    manifest["combined"] = {
        "human_all_path": str(human_all_path),
        "machine_all_path": str(machine_all_path),
        "paired_all_path": str(paired_all_path),
        "pair_preview_path": str(preview_path),
        "count": len(all_pairs),
    }

    manifest_path = export_dir / "manifest.json"
    dump_json(manifest, manifest_path)
    _write_readable_export_readme(
        export_dir,
        dataset_tag=dataset_tag,
        source_dataset_dir=Path(source_dataset_dir),
        human_path=human_path,
        prompt_mode=prompt_mode,
    )
    return {
        "export_dir": str(export_dir),
        "manifest_path": str(manifest_path),
        "pair_preview_path": str(preview_path),
    }


def run_build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    configure_logging()
    dataset_tag = _resolve_setting(args, config, "dataset_tag", "default")
    checkpoint_dir = _resolve_setting(args, config, "checkpoint_dir", "models/duo-distilled")
    tokenizer_dir = _resolve_setting(args, config, "tokenizer_dir", "assets/tokenizer/gpt2")
    allow_online_tokenizer = bool(_resolve_setting(args, config, "allow_online_tokenizer", False))
    if args.allow_online_tokenizer:
        allow_online_tokenizer = True
    human_path = _resolve_setting(args, config, "human_path", "data/human")
    if args.human_path is not None:
        human_path = args.human_path
    output_root = Path(_resolve_setting(args, config, "output_root", "outputs"))
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "datasets" / dataset_tag
    export_data_views = not args.no_data_export and bool(_resolve_setting(args, config, "export_data_views", True))
    data_export_root = _resolve_setting(args, config, "data_export_root", "data/generated")
    resolved_generation_steps = _resolve_setting(args, config, "generation_steps", 8)
    if isinstance(resolved_generation_steps, list):
        resolved_generation_steps = resolved_generation_steps[0]
    generation_steps = int(resolved_generation_steps)
    generation_strategy = _resolve_setting(args, config, "generation_strategy", "top_p")
    generation_batch_size = int(_resolve_setting(args, config, "generation_batch_size", 1))
    temperature = float(_resolve_setting(args, config, "generation_temperature", 1.0))
    top_p = float(_resolve_setting(args, config, "generation_top_p", 0.9))
    prompt_fraction = float(_resolve_setting(args, config, "prompt_fraction", 0.25))
    prompt_tokens = _resolve_setting(args, config, "prompt_tokens", None)
    prompt_tokens = None if prompt_tokens is None else int(prompt_tokens)
    target_total_tokens = _resolve_setting(args, config, "target_total_tokens", None)
    target_total_tokens = None if target_total_tokens is None else int(target_total_tokens)
    window_stride = _resolve_setting(args, config, "window_stride", target_total_tokens)
    window_stride = None if window_stride is None else int(window_stride)
    max_windows_per_source = _resolve_setting(args, config, "max_windows_per_source", 1 if target_total_tokens is not None else None)
    max_windows_per_source = None if max_windows_per_source is None else int(max_windows_per_source)
    max_examples = _resolve_setting(args, config, "max_examples", None)
    max_examples = None if max_examples is None else int(max_examples)
    seed = int(_resolve_setting(args, config, "seed", 7))

    requested_unconditional_only = bool(args.unconditional_only or _resolve_setting(args, config, "dataset_unconditional_only", False))
    if prompt_tokens is not None and prompt_tokens < 0:
        raise ValueError(f"prompt_tokens must be >= 0, got {prompt_tokens}.")
    if target_total_tokens is not None and target_total_tokens <= 1:
        raise ValueError(f"target_total_tokens must be >= 2, got {target_total_tokens}.")
    if (
        target_total_tokens is not None
        and prompt_tokens is not None
        and prompt_tokens >= target_total_tokens
        and not requested_unconditional_only
    ):
        raise ValueError(
            f"prompt_tokens must be < target_total_tokens for conditional generation, got prompt_tokens={prompt_tokens}, "
            f"target_total_tokens={target_total_tokens}."
        )

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

    unconditional_only = requested_unconditional_only
    prompt_mode = "unconditional"
    if not unconditional_only:
        prompt_mode = "fixed_tokens" if prompt_tokens is not None else "fractional"

    manifest = {
        "dataset_tag": dataset_tag,
        "human_path": str(human_path),
        "checkpoint_dir": str(checkpoint_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "tokenizer_load_report": adapter.tokenizer_load_report,
        "tokenizer_vocab_audit": adapter.vocab_audit,
        "generation_steps": generation_steps,
        "generation_strategy": generation_strategy,
        "generation_batch_size": generation_batch_size,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_mode": prompt_mode,
        "prompt_fraction": prompt_fraction,
        "prompt_tokens": prompt_tokens,
        "target_total_tokens": target_total_tokens,
        "window_stride": window_stride,
        "max_windows_per_source": max_windows_per_source,
        "max_examples": max_examples,
        "approximation_mode": "iterative_masked_denoising_generation",
        "splits": {},
        "selection_summary": {},
    }

    candidates_by_split: dict[str, list[dict[str, Any]]] = {}
    candidate_summary: dict[str, Any] = {}
    for split_name, split_records_for_name in split_map.items():
        split_candidates, split_stats = _build_candidate_examples(
            records=split_records_for_name,
            split_name=split_name,
            tokenizer=tokenizer,
            max_length=adapter.max_length,
            prompt_tokens=prompt_tokens,
            prompt_fraction=prompt_fraction,
            target_total_tokens=target_total_tokens,
            window_stride=window_stride,
            max_windows_per_source=max_windows_per_source,
            unconditional_only=unconditional_only,
            seed=seed,
        )
        candidates_by_split[split_name] = split_candidates
        candidate_summary[split_name] = {
            **split_stats,
            "selected_examples": 0,
        }

    selected_by_split = _select_candidates(candidates_by_split, max_examples=max_examples)
    total_selected = sum(len(rows) for rows in selected_by_split.values())
    if total_selected == 0:
        raise ValueError(
            "No usable examples were selected for dataset generation. "
            "If you set target_total_tokens, try reducing it or allowing more windows per source."
        )

    preview_rows: list[dict[str, Any]] = []
    human_records_by_split: dict[str, list[dict[str, Any]]] = {}
    machine_records_by_split: dict[str, list[dict[str, Any]]] = {}

    for split_name in SPLIT_ORDER:
        split_candidates = selected_by_split.get(split_name, [])
        candidate_summary[split_name]["selected_examples"] = len(split_candidates)
        lengths = [item["token_length"] for item in split_candidates]
        original_rows = [item["text"] for item in split_candidates]

        if split_candidates:
            generated_texts: list[str] = []
            generated_token_ids: list[list[int]] = []
            for start_idx in range(0, len(split_candidates), generation_batch_size):
                batch_records = split_candidates[start_idx : start_idx + generation_batch_size]
                batch_prompt_texts = None if unconditional_only else [item["prompt_text"] for item in batch_records]
                batch_prompt_token_ids = None if unconditional_only else [item["prompt_token_ids"] for item in batch_records]
                batch_generated = adapter.generate_texts(
                    target_lengths=[item["token_length"] for item in batch_records],
                    prompt_texts=batch_prompt_texts,
                    prompt_token_id_seqs=batch_prompt_token_ids,
                    num_steps=generation_steps,
                    strategy=generation_strategy,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed + start_idx,
                )
                generated_texts.extend(batch_generated["texts"])
                generated_token_ids.extend(batch_generated["token_ids"])
        else:
            generated_texts = []
            generated_token_ids = []

        sampled_rows = generated_texts
        original_records = []
        sampled_records = []
        for idx, item in enumerate(split_candidates):
            base_metadata = {
                "source_id": item["source_id"],
                "split": split_name,
                "token_length": item["token_length"],
                "token_ids": item["token_ids"],
                "prompt_length": item["prompt_length"],
                "prompt_text": item["prompt_text"],
                "prompt_token_ids": item["prompt_token_ids"],
                "continuation_text": item["continuation_text"],
                "continuation_token_ids": item["continuation_token_ids"],
                "target_total_tokens": item["token_length"],
                "segment_index": item["segment_index"],
                "segment_start": item["segment_start"],
                "segment_end": item["segment_end"],
                "source_token_length": item["source_token_length"],
            }
            original_records.append(
                {
                    "example_id": item["human_example_id"],
                    "label": "human",
                    "text": item["text"],
                    **base_metadata,
                }
            )
            sampled_records.append(
                {
                    "example_id": item["machine_example_id"],
                    "label": "machine",
                    "text": generated_texts[idx],
                    "token_ids": generated_token_ids[idx],
                    "token_length": len(generated_token_ids[idx]),
                    "prompt_length": item["prompt_length"],
                    "prompt_text": item["prompt_text"],
                    "prompt_token_ids": item["prompt_token_ids"],
                    "continuation_text": item["continuation_text"],
                    "continuation_token_ids": item["continuation_token_ids"],
                    "target_total_tokens": item["token_length"],
                    "source_id": item["source_id"],
                    "split": split_name,
                    "segment_index": item["segment_index"],
                    "segment_start": item["segment_start"],
                    "segment_end": item["segment_end"],
                    "source_token_length": item["source_token_length"],
                    "generation_steps": generation_steps,
                    "generation_strategy": generation_strategy,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
            if len(preview_rows) < 12:
                preview_rows.append(
                    {
                        "split": split_name,
                        "source_id": item["source_id"],
                        "segment_index": item["segment_index"],
                        "segment_start": item["segment_start"],
                        "segment_end": item["segment_end"],
                        "prompt_length": item["prompt_length"],
                        "prompt_text": item["prompt_text"],
                        "human_text": item["text"],
                        "machine_text": generated_texts[idx],
                    }
                )

        human_records_by_split[split_name] = original_records
        machine_records_by_split[split_name] = sampled_records

        metadata = {
            "split": split_name,
            "approximation_mode": "iterative_masked_denoising_generation",
            "prompt_mode": prompt_mode,
            "prompt_fraction": prompt_fraction,
            "prompt_tokens": prompt_tokens,
            "target_total_tokens": target_total_tokens,
            "window_stride": window_stride,
            "max_windows_per_source": max_windows_per_source,
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

    manifest["selection_summary"] = {
        "total_source_records": len(records),
        "total_candidate_examples": sum(stats["candidate_windows"] for stats in candidate_summary.values()),
        "total_selected_examples": total_selected,
        "splits": candidate_summary,
    }

    preview_path = output_dir / "prompt_pair_preview.json"
    dump_json(preview_rows, preview_path)
    manifest["prompt_pair_preview_path"] = str(preview_path)

    if export_data_views:
        export_result = _export_readable_dataset_views(
            dataset_tag=dataset_tag,
            export_root=data_export_root,
            human_records_by_split=human_records_by_split,
            machine_records_by_split=machine_records_by_split,
            preview_rows=preview_rows,
            source_dataset_dir=output_dir,
            human_path=human_path,
            prompt_mode=prompt_mode,
        )
        manifest["readable_data_export"] = export_result

    manifest_path = output_dir / "manifest.json"
    dump_json(manifest, manifest_path)
    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "data_export_dir": manifest.get("readable_data_export", {}).get("export_dir"),
        "dataset_tag": str(dataset_tag),
    }


def main() -> None:
    args = parse_args()
    result = run_build_dataset(args)
    print(f"Dataset build complete: {result['output_dir']}")


if __name__ == "__main__":
    main()
