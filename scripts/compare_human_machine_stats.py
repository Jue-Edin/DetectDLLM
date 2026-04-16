import argparse
import re
import statistics
import string
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import roc_auc
from src.utils import configure_logging, dump_json, ensure_dir, load_json, load_jsonl


WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
SENTENCE_RE = re.compile(r"[.!?]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare human and machine text statistics and detector score distributions.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing dataset_*_metadata.json files.")
    parser.add_argument("--analytic-dir", default=None, help="Directory containing analytic_rows.jsonl.")
    parser.add_argument("--fastdetectgpt-dir", default=None, help="Directory containing fastdetectgpt_rows.jsonl.")
    parser.add_argument("--output-dir", default=None, help="Where to save comparison tables.")
    return parser.parse_args()


def _load_examples(dataset_dir: str | Path) -> list[dict[str, Any]]:
    dataset_dir = Path(dataset_dir)
    examples = []
    for metadata_path in sorted(dataset_dir.glob("dataset_*_metadata.json")):
        metadata = load_json(metadata_path)
        examples.extend(metadata.get("original_records", []))
        examples.extend(metadata.get("sampled_records", []))
    if not examples:
        raise FileNotFoundError(
            f"No dataset metadata found in '{dataset_dir}'. Run scripts/build_dataset.py first."
        )
    return examples


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _cohen_d(machine_values: list[float], human_values: list[float]) -> float:
    if not machine_values or not human_values:
        return 0.0
    x = np.asarray(machine_values, dtype=np.float64)
    y = np.asarray(human_values, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return 0.0
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    pooled_num = (x.size - 1) * sx * sx + (y.size - 1) * sy * sy
    pooled_den = x.size + y.size - 2
    if pooled_den <= 0:
        return 0.0
    pooled = float(np.sqrt(max(pooled_num / pooled_den, 0.0)))
    if pooled == 0.0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


def _cliffs_delta(machine_values: list[float], human_values: list[float]) -> float:
    if not machine_values or not human_values:
        return 0.0
    gt = 0
    lt = 0
    for x in machine_values:
        for y in human_values:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    total = len(machine_values) * len(human_values)
    if total == 0:
        return 0.0
    return float((gt - lt) / total)


def _format_value(value: float) -> str:
    return f"{value:.4f}"


def _write_markdown_table(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("_No rows._")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(_format_value(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _collect_text_features(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        text = example["text"]
        char_count = len(text)
        alpha_count = sum(ch.isalpha() for ch in text)
        punct_count = sum(ch in string.punctuation for ch in text)
        digit_count = sum(ch.isdigit() for ch in text)
        whitespace_count = sum(ch.isspace() for ch in text)
        words = WORD_RE.findall(text)
        word_count = len(words)
        unique_word_count = len({word.lower() for word in words})
        sentence_chunks = [chunk.strip() for chunk in SENTENCE_RE.split(text) if chunk.strip()]
        sentence_count = max(1, len(sentence_chunks)) if text.strip() else 0
        avg_word_length = float(np.mean([len(word) for word in words])) if words else 0.0
        rows.append(
            {
                "example_id": example["example_id"],
                "label": example["label"],
                "split": example["split"],
                "char_count": float(char_count),
                "word_count": float(word_count),
                "token_length": float(example.get("token_length", 0)),
                "sentence_count": float(sentence_count),
                "avg_word_length": float(avg_word_length),
                "type_token_ratio": float(unique_word_count / max(word_count, 1)),
                "punct_ratio": float(punct_count / max(char_count, 1)),
                "digit_ratio": float(digit_count / max(char_count, 1)),
                "uppercase_ratio": float(sum(ch.isupper() for ch in text) / max(alpha_count, 1)),
                "whitespace_ratio": float(whitespace_count / max(char_count, 1)),
            }
        )
    return rows


def _summarize_features_by_label(
    feature_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    numeric_fields = [
        "char_count",
        "word_count",
        "token_length",
        "sentence_count",
        "avg_word_length",
        "type_token_ratio",
        "punct_ratio",
        "digit_ratio",
        "uppercase_ratio",
        "whitespace_ratio",
    ]
    summary_rows = []
    comparison_rows = []
    for split_name in ["overall", "train", "val", "test"]:
        split_rows = feature_rows if split_name == "overall" else [row for row in feature_rows if row["split"] == split_name]
        if not split_rows:
            continue
        for label in ["human", "machine"]:
            label_rows = [row for row in split_rows if row["label"] == label]
            if not label_rows:
                continue
            for field in numeric_fields:
                stats = _summary([float(row[field]) for row in label_rows])
                summary_rows.append(
                    {
                        "split": split_name,
                        "label": label,
                        "feature": field,
                        **stats,
                    }
                )
        human_rows = [row for row in split_rows if row["label"] == "human"]
        machine_rows = [row for row in split_rows if row["label"] == "machine"]
        if not human_rows or not machine_rows:
            continue
        for field in numeric_fields:
            human_values = [float(row[field]) for row in human_rows]
            machine_values = [float(row[field]) for row in machine_rows]
            comparison_rows.append(
                {
                    "split": split_name,
                    "feature": field,
                    "human_mean": float(np.mean(human_values)),
                    "machine_mean": float(np.mean(machine_values)),
                    "machine_minus_human": float(np.mean(machine_values) - np.mean(human_values)),
                    "cohen_d": _cohen_d(machine_values, human_values),
                    "cliffs_delta": _cliffs_delta(machine_values, human_values),
                }
            )

    dump_json({"rows": summary_rows}, output_dir / "text_stats_by_label.json")
    dump_json({"rows": comparison_rows}, output_dir / "text_stats_comparison.json")
    _write_markdown_table(output_dir / "text_stats_by_label.md", "Text Statistics by Label", summary_rows)
    _write_markdown_table(output_dir / "text_stats_comparison.md", "Human vs Machine Text Statistics", comparison_rows)
    return {"summary_rows": summary_rows, "comparison_rows": comparison_rows}


def _choose_orientation(labels: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray, float]:
    auc_forward = roc_auc(labels, scores)
    auc_reverse = roc_auc(labels, -scores)
    if auc_reverse > auc_forward:
        return -1.0, -scores, float(auc_reverse)
    return 1.0, scores, float(auc_forward)


def _aggregate_analytic_rows(score_rows: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float, str], list[float]] = defaultdict(list)
    for row in score_rows:
        key = (row["example_id"], row["split"], float(row["mask_ratio"]), row["label"])
        grouped[key].append(float(row[score_field]))
    aggregated = []
    for key, values in grouped.items():
        example_id, split, mask_ratio, label = key
        aggregated.append(
            {
                "example_id": example_id,
                "split": split,
                "mask_ratio": float(mask_ratio),
                "label": label,
                "score": float(np.mean(values)),
                "score_std_across_seeds": float(np.std(values)),
            }
        )
    return aggregated


def _detector_label_stats(rows: list[dict[str, Any]], score_sign: float) -> list[dict[str, Any]]:
    oriented_rows = []
    for row in rows:
        oriented = dict(row)
        oriented["score_oriented"] = float(score_sign * float(row["score"]))
        oriented_rows.append(oriented)
    stats_rows = []
    for split_name in ["val", "test", "overall"]:
        split_rows = oriented_rows if split_name == "overall" else [row for row in oriented_rows if row["split"] == split_name]
        if not split_rows:
            continue
        for label in ["human", "machine"]:
            label_rows = [row for row in split_rows if row["label"] == label]
            if not label_rows:
                continue
            stats = _summary([float(row["score_oriented"]) for row in label_rows])
            stats_rows.append(
                {
                    "split": split_name,
                    "label": label,
                    **stats,
                }
            )
    return stats_rows


def _detector_comparison_rows(rows: list[dict[str, Any]], score_sign: float) -> list[dict[str, Any]]:
    oriented_rows = []
    for row in rows:
        oriented = dict(row)
        oriented["score_oriented"] = float(score_sign * float(row["score"]))
        oriented_rows.append(oriented)
    comparison_rows = []
    for split_name in ["val", "test", "overall"]:
        split_rows = oriented_rows if split_name == "overall" else [row for row in oriented_rows if row["split"] == split_name]
        human_rows = [row for row in split_rows if row["label"] == "human"]
        machine_rows = [row for row in split_rows if row["label"] == "machine"]
        if not human_rows or not machine_rows:
            continue
        human_values = [float(row["score_oriented"]) for row in human_rows]
        machine_values = [float(row["score_oriented"]) for row in machine_rows]
        comparison_rows.append(
            {
                "split": split_name,
                "human_mean": float(np.mean(human_values)),
                "machine_mean": float(np.mean(machine_values)),
                "machine_minus_human": float(np.mean(machine_values) - np.mean(human_values)),
                "cohen_d": _cohen_d(machine_values, human_values),
                "cliffs_delta": _cliffs_delta(machine_values, human_values),
            }
        )
    return comparison_rows


def _select_best_analytic_setting(score_rows: list[dict[str, Any]], score_field: str) -> dict[str, Any]:
    aggregated = _aggregate_analytic_rows(score_rows, score_field=score_field)
    best = None
    best_val_auc = -1.0
    for mask_ratio in sorted({float(row["mask_ratio"]) for row in aggregated}):
        val_rows = [row for row in aggregated if row["split"] == "val" and float(row["mask_ratio"]) == mask_ratio]
        if not val_rows:
            continue
        val_labels = np.asarray([1 if row["label"] == "machine" else 0 for row in val_rows], dtype=np.int64)
        val_scores_raw = np.asarray([float(row["score"]) for row in val_rows], dtype=np.float64)
        score_sign, _, val_auc = _choose_orientation(val_labels, val_scores_raw)
        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best = {
                "mask_ratio": float(mask_ratio),
                "score_sign": float(score_sign),
                "val_roc_auc": float(val_auc),
                "rows": [row for row in aggregated if float(row["mask_ratio"]) == mask_ratio],
            }
    if best is None:
        raise ValueError("Could not select an analytic setting from analytic_rows.jsonl.")
    return best


def _summarize_analytic_scores(analytic_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows_path = analytic_dir / "analytic_rows.jsonl"
    if not rows_path.exists():
        raise FileNotFoundError(f"Expected analytic rows at {rows_path}")
    score_rows = load_jsonl(rows_path)
    outputs = {}
    for detector_name, score_field in [("baseline", "baseline_score"), ("duo_analytic", "z_score")]:
        selected = _select_best_analytic_setting(score_rows, score_field=score_field)
        stats_rows = _detector_label_stats(selected["rows"], selected["score_sign"])
        comparison_rows = _detector_comparison_rows(selected["rows"], selected["score_sign"])
        payload = {
            "detector": detector_name,
            "selected_mask_ratio": selected["mask_ratio"],
            "score_sign": selected["score_sign"],
            "val_roc_auc": selected["val_roc_auc"],
            "stats_rows": stats_rows,
            "comparison_rows": comparison_rows,
        }
        outputs[detector_name] = payload
        dump_json(payload, output_dir / f"{detector_name}_score_stats.json")
        _write_markdown_table(
            output_dir / f"{detector_name}_score_stats.md",
            f"{detector_name} Score Statistics by Label",
            [
                {
                    "detector": detector_name,
                    "selected_mask_ratio": selected["mask_ratio"],
                    "score_sign": selected["score_sign"],
                    **row,
                }
                for row in stats_rows
            ],
        )
        _write_markdown_table(
            output_dir / f"{detector_name}_score_comparison.md",
            f"{detector_name} Human vs Machine Score Comparison",
            [
                {
                    "detector": detector_name,
                    "selected_mask_ratio": selected["mask_ratio"],
                    "score_sign": selected["score_sign"],
                    "val_roc_auc": selected["val_roc_auc"],
                    **row,
                }
                for row in comparison_rows
            ],
        )
    return outputs


def _summarize_fastdetectgpt_scores(fdg_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows_path = fdg_dir / "fastdetectgpt_rows.jsonl"
    if not rows_path.exists():
        raise FileNotFoundError(f"Expected Fast-DetectGPT rows at {rows_path}")
    rows = load_jsonl(rows_path)
    val_rows = [row for row in rows if row["split"] == "val"]
    if not val_rows:
        raise ValueError("Fast-DetectGPT rows do not contain a validation split.")
    val_labels = np.asarray([1 if row["label"] == "machine" else 0 for row in val_rows], dtype=np.int64)
    val_scores_raw = np.asarray([float(row["score"]) for row in val_rows], dtype=np.float64)
    score_sign, _, val_auc = _choose_orientation(val_labels, val_scores_raw)
    stats_rows = _detector_label_stats(rows, score_sign=score_sign)
    comparison_rows = _detector_comparison_rows(rows, score_sign=score_sign)
    payload = {
        "detector": "fastdetectgpt_surrogate",
        "score_sign": float(score_sign),
        "val_roc_auc": float(val_auc),
        "stats_rows": stats_rows,
        "comparison_rows": comparison_rows,
    }
    dump_json(payload, output_dir / "fastdetectgpt_score_stats.json")
    _write_markdown_table(
        output_dir / "fastdetectgpt_score_stats.md",
        "Fast-DetectGPT Score Statistics by Label",
        [
            {
                "detector": "fastdetectgpt_surrogate",
                "score_sign": float(score_sign),
                **row,
            }
            for row in stats_rows
        ],
    )
    _write_markdown_table(
        output_dir / "fastdetectgpt_score_comparison.md",
        "Fast-DetectGPT Human vs Machine Score Comparison",
        [
            {
                "detector": "fastdetectgpt_surrogate",
                "score_sign": float(score_sign),
                "val_roc_auc": float(val_auc),
                **row,
            }
            for row in comparison_rows
        ],
    )
    return payload


def _write_master_summary(output_dir: Path, text_stats: dict[str, Any], analytic_stats: dict[str, Any] | None, fdg_stats: dict[str, Any] | None) -> None:
    lines = [
        "# Human vs Machine Statistics Summary",
        "",
        "This report compares human and machine texts at two levels:",
        "",
        "1. raw text statistics computed from the dataset itself;",
        "2. detector score statistics computed from saved row-level outputs.",
        "",
        "Higher oriented detector scores always mean *more machine-like*.",
        "",
        "## Raw text statistics",
        "",
        f"- saved in `text_stats_by_label.md` and `text_stats_comparison.md`",
        "",
    ]
    if analytic_stats is not None:
        lines.extend(
            [
                "## Analytic DUO detector score statistics",
                "",
                f"- baseline selected mask ratio: {analytic_stats['baseline']['selected_mask_ratio']:.2f}",
                f"- baseline validation ROC-AUC used for orientation selection: {analytic_stats['baseline']['val_roc_auc']:.4f}",
                f"- duo_analytic selected mask ratio: {analytic_stats['duo_analytic']['selected_mask_ratio']:.2f}",
                f"- duo_analytic validation ROC-AUC used for orientation selection: {analytic_stats['duo_analytic']['val_roc_auc']:.4f}",
                "",
            ]
        )
    if fdg_stats is not None:
        lines.extend(
            [
                "## Fast-DetectGPT score statistics",
                "",
                f"- validation ROC-AUC used for orientation selection: {fdg_stats['val_roc_auc']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## How to read the tables",
            "",
            "- `machine_minus_human > 0` means the machine class tends to have larger values.",
            "- `cohen_d` measures standardized mean difference.",
            "- `cliffs_delta` is a rank-based effect size in [-1, 1].",
            "",
        ]
    )
    (output_dir / "human_machine_stats_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    configure_logging()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else ensure_dir(dataset_dir.parent.parent / "analysis" / f"{dataset_dir.name}_human_vs_machine")
    ensure_dir(output_dir)

    examples = _load_examples(dataset_dir)
    feature_rows = _collect_text_features(examples)
    text_stats = _summarize_features_by_label(feature_rows, output_dir)

    analytic_stats = None
    if args.analytic_dir:
        analytic_stats = _summarize_analytic_scores(Path(args.analytic_dir), output_dir)

    fdg_stats = None
    if args.fastdetectgpt_dir:
        fdg_stats = _summarize_fastdetectgpt_scores(Path(args.fastdetectgpt_dir), output_dir)

    _write_master_summary(output_dir, text_stats=text_stats, analytic_stats=analytic_stats, fdg_stats=fdg_stats)
    dump_json(
        {
            "dataset_dir": str(dataset_dir),
            "analytic_dir": str(args.analytic_dir) if args.analytic_dir else None,
            "fastdetectgpt_dir": str(args.fastdetectgpt_dir) if args.fastdetectgpt_dir else None,
            "output_dir": str(output_dir),
            "text_stats": text_stats,
            "analytic_stats": analytic_stats,
            "fastdetectgpt_stats": fdg_stats,
        },
        output_dir / "human_machine_stats_summary.json",
    )
    return {"output_dir": str(output_dir)}


def main() -> None:
    args = parse_args()
    result = run_comparison(args)
    print(f"Human vs machine comparison complete: {result['output_dir']}")


if __name__ == "__main__":
    main()
