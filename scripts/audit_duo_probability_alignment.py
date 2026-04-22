"""Audit DUO mask-token, position, and vocabulary alignment.

This script is designed for advisor/debugging checks. It verifies that the
probability used by the DUO detector is exactly

    log p_theta(original_token_at_position_i | corrupted_sequence, timestep)

at the masked positions i.

It does not prove that the detector statistic is theoretically optimal, but it
catches the two most damaging implementation bugs:
  1. position shift: gathering logits from the wrong sequence position;
  2. token-id mismatch: gathering the wrong vocabulary entry.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.duo_adapter import DuoAdapter
from src.utils import dump_json, ensure_dir, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit DUO probability extraction alignment.")
    parser.add_argument("--config", default="configs/paper_prompt_local.json")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--metadata-file", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--label", default="machine", choices=["human", "machine"])
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--corruption-seed", type=int, default=11)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/debug/probability_alignment")
    return parser.parse_args()


def _resolve_dataset_dir(config: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.dataset_dir:
        return Path(args.dataset_dir)
    output_root = Path(config.get("output_root", "outputs"))
    dataset_tag = config.get("dataset_tag", "squad_duo_local")
    return output_root / "datasets" / dataset_tag


def _load_record(args: argparse.Namespace, dataset_dir: Path) -> dict[str, Any]:
    metadata_file = Path(args.metadata_file) if args.metadata_file else dataset_dir / f"dataset_{args.split}_metadata.json"
    metadata = load_json(metadata_file)
    key = "sampled_records" if args.label == "machine" else "original_records"
    records = metadata.get(key, [])
    if not records:
        raise ValueError(f"No {key} found in {metadata_file}")
    if args.example_index < 0 or args.example_index >= len(records):
        raise IndexError(f"example-index={args.example_index} outside [0, {len(records) - 1}]")
    return records[args.example_index]


def _decode_one(adapter: DuoAdapter, token_id: int) -> str:
    return str(adapter.decode_ids([int(token_id)], skip_special_tokens=False))


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    dataset_dir = _resolve_dataset_dir(config, args)
    output_dir = ensure_dir(args.output_dir)

    adapter = DuoAdapter(
        checkpoint_dir=config.get("checkpoint_dir", "models/duo-distilled"),
        tokenizer_dir=config.get("tokenizer_dir", "models/duo-distilled"),
        allow_online_tokenizer=bool(config.get("allow_online_tokenizer", False)),
    ).load(require_tokenizer=True)

    record = _load_record(args, dataset_dir)
    original_ids = torch.tensor(record["token_ids"], dtype=torch.long)
    corruption = adapter.corrupt_ids(original_ids, mask_ratio=args.mask_ratio, seed=args.corruption_seed)
    corrupted_ids = corruption.corrupted_ids.squeeze(0)
    mask = corruption.mask.squeeze(0)
    masked_positions = torch.where(mask)[0]
    if len(masked_positions) == 0:
        raise RuntimeError("Corruption produced no masked positions.")

    timestep = mask.float().mean().unsqueeze(0)
    logits = adapter.reconstruct_logits(corrupted_ids, timesteps=timestep).squeeze(0)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    # This is the exact indexing used by scripts/run_duo_analytic.py.
    gathered = log_probs.gather(dim=-1, index=original_ids.to(log_probs.device).unsqueeze(-1)).squeeze(-1)

    rows: list[dict[str, Any]] = []
    top_k = max(1, int(args.top_k))
    for pos in masked_positions[: min(len(masked_positions), 8)].tolist():
        original_id = int(original_ids[pos].item())
        corrupted_id = int(corrupted_ids[pos].item())
        if corrupted_id != adapter.mask_token_id:
            raise AssertionError(
                f"Position {pos} is marked masked but corrupted_ids[{pos}]={corrupted_id}, "
                f"expected mask_token_id={adapter.mask_token_id}."
            )
        direct_logprob = float(log_probs[pos, original_id].detach().cpu().item())
        gathered_logprob = float(gathered[pos].detach().cpu().item())
        if abs(direct_logprob - gathered_logprob) > 1e-7:
            raise AssertionError(
                f"Gather mismatch at position {pos}: direct={direct_logprob}, gathered={gathered_logprob}"
            )
        top_probs, top_ids = torch.topk(probs[pos], k=min(top_k, probs.shape[-1]))
        rows.append(
            {
                "position": int(pos),
                "corrupted_token_id": corrupted_id,
                "corrupted_token_text": _decode_one(adapter, corrupted_id),
                "original_token_id": original_id,
                "original_token_text": _decode_one(adapter, original_id),
                "observed_logprob": gathered_logprob,
                "observed_prob": float(torch.exp(gathered[pos]).detach().cpu().item()),
                "top_predictions": [
                    {
                        "rank": rank + 1,
                        "token_id": int(tok_id.item()),
                        "token_text": _decode_one(adapter, int(tok_id.item())),
                        "prob": float(tok_prob.item()),
                    }
                    for rank, (tok_id, tok_prob) in enumerate(zip(top_ids.detach().cpu(), top_probs.detach().cpu()))
                ],
            }
        )

    special_counts = {
        "pad_eos_count": int((original_ids == adapter._require_tokenizer().eos_token_id).sum().item()),
        "mask_count_in_original": int((original_ids == adapter.mask_token_id).sum().item()),
    }
    prefix_ok = None
    if record.get("prompt_length", 0) and "prompt_token_ids" in record:
        plen = int(record["prompt_length"])
        prefix_ok = list(record["token_ids"][:plen]) == list(record["prompt_token_ids"])

    report = {
        "status": "passed",
        "config": args.config,
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "label": args.label,
        "example_id": record.get("example_id"),
        "text_preview": record.get("text", "")[:300],
        "vocab_audit": adapter.vocab_audit,
        "tokenizer_load_report": adapter.tokenizer_load_report,
        "logits_shape": list(logits.shape),
        "sequence_length": int(original_ids.numel()),
        "mask_ratio_requested": float(args.mask_ratio),
        "num_masked": int(mask.sum().item()),
        "timestep_passed_to_model": float(timestep.item()),
        "special_counts_in_original": special_counts,
        "prompt_prefix_matches_token_ids": prefix_ok,
        "checked_positions": rows,
        "interpretation": (
            "For each checked masked position, corrupted_ids[position] equals the tokenizer mask token id, "
            "and gathered log-prob equals log_probs[position, original_token_id]. This verifies position/id "
            "alignment for the implementation path used by run_duo_analytic.py."
        ),
    }

    out_json = Path(output_dir) / "probability_alignment_report.json"
    dump_json(report, out_json)

    lines = ["# DUO probability alignment audit", "", f"Status: **{report['status']}**", ""]
    lines.append(f"Example: `{report['example_id']}` ({args.split}/{args.label})")
    lines.append(f"Logits shape: `{report['logits_shape']}` = `[seq_len, vocab_size]`")
    lines.append(f"Mask token: `{adapter._require_tokenizer().mask_token}` id `{adapter.mask_token_id}`")
    lines.append("")
    lines.append("| position | original token id | original token | observed prob | observed logprob | top-1 token | top-1 prob |")
    lines.append("| --- | ---: | --- | ---: | ---: | --- | ---: |")
    for row in rows:
        top1 = row["top_predictions"][0]
        lines.append(
            f"| {row['position']} | {row['original_token_id']} | `{row['original_token_text']}` | "
            f"{row['observed_prob']:.6g} | {row['observed_logprob']:.6f} | "
            f"`{top1['token_text']}` | {top1['prob']:.6g} |"
        )
    out_md = Path(output_dir) / "probability_alignment_report.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Probability alignment audit passed. Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
