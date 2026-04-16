from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.corruption import compute_length_bucket_boundaries, length_bucket
from src.utils import dump_json, stable_hash


SUPPORTED_SUFFIXES = {".txt", ".json", ".jsonl", ".csv"}
FIELD_PRIORITY = ["text", "content", "body"]
LIST_FIELD_PRIORITY = ["original", "original_only", "human", "texts", "documents", "data"]


def _choose_text_field(row: dict[str, Any], text_field: str | None) -> str:
    if text_field is not None:
        if text_field not in row:
            raise ValueError(f"Requested text field '{text_field}' not present in row keys {list(row.keys())}.")
        return text_field
    for candidate in FIELD_PRIORITY:
        if candidate in row and isinstance(row[candidate], str):
            return candidate
    raise ValueError(
        "Could not infer a text field. Provide --text-field explicitly. "
        f"Available keys: {list(row.keys())}"
    )


def _load_txt(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    if len(chunks) <= 1:
        chunks = [line.strip() for line in text.splitlines() if line.strip()]
    records = []
    for idx, chunk in enumerate(chunks):
        records.append({"source_id": f"{path.as_posix()}::{idx}", "text": chunk, "meta": {"path": str(path)}})
    return records


def _load_json_like(path: Path, text_field: str | None = None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = None
        for key in LIST_FIELD_PRIORITY:
            value = payload.get(key)
            if isinstance(value, list):
                items = value
                break
        if items is None:
            list_candidates = [value for value in payload.values() if isinstance(value, list)]
            if len(list_candidates) == 1:
                items = list_candidates[0]
        if items is None:
            raise ValueError(
                f"Unsupported JSON structure in {path}. Expected a list or a dict containing one of {LIST_FIELD_PRIORITY}."
            )
    else:
        raise ValueError(f"Unsupported JSON structure in {path}. Expected a list or dict-like payload.")
    records = []
    for idx, item in enumerate(items):
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            field = _choose_text_field(item, text_field=text_field)
            text = item[field]
        else:
            continue
        text = text.strip()
        if not text:
            continue
        records.append({"source_id": f"{path.as_posix()}::{idx}", "text": text, "meta": {"path": str(path)}})
    return records


def _load_jsonl(path: Path, text_field: str | None = None) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                field = _choose_text_field(item, text_field=text_field)
                text = item[field]
            else:
                continue
            text = text.strip()
            if text:
                records.append({"source_id": f"{path.as_posix()}::{idx}", "text": text, "meta": {"path": str(path)}})
    return records


def _load_csv(path: Path, text_field: str | None = None) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            field = _choose_text_field(row, text_field=text_field)
            text = row[field].strip()
            if text:
                records.append({"source_id": f"{path.as_posix()}::{idx}", "text": text, "meta": {"path": str(path)}})
    return records


def load_human_corpus(human_path: str | Path, text_field: str | None = None) -> list[dict[str, Any]]:
    root = Path(human_path)
    if not root.exists():
        raise FileNotFoundError(
            f"Human text path '{root}' does not exist. Provide a local corpus path with .txt/.json/.jsonl/.csv files."
        )
    files = [root] if root.is_file() else [path for path in root.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES]
    if not files:
        raise FileNotFoundError(
            f"No supported human text files found under '{root}'. Expected at least one of {sorted(SUPPORTED_SUFFIXES)}."
        )
    records: list[dict[str, Any]] = []
    for path in sorted(files):
        suffix = path.suffix.lower()
        if suffix == ".txt":
            records.extend(_load_txt(path))
        elif suffix == ".json":
            records.extend(_load_json_like(path, text_field=text_field))
        elif suffix == ".jsonl":
            records.extend(_load_jsonl(path, text_field=text_field))
        elif suffix == ".csv":
            records.extend(_load_csv(path, text_field=text_field))
    if not records:
        raise ValueError(f"Loaded zero usable text documents from '{root}'.")
    return records


def assign_split(source_id: str, seed: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    digest = stable_hash({"source_id": source_id, "seed": seed})
    value = int(digest[:8], 16) / float(0xFFFFFFFF)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def split_records(records: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    splits = {"train": [], "val": [], "test": []}
    for record in records:
        split = assign_split(record["source_id"], seed=seed)
        enriched = dict(record)
        enriched["split"] = split
        splits[split].append(enriched)
    return splits


def summarize_lengths(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"count": 0, "bucket_boundaries": [0, 0]}
    boundaries = compute_length_bucket_boundaries(lengths)
    buckets = {"short": 0, "medium": 0, "long": 0}
    for value in lengths:
        buckets[length_bucket(value, boundaries)] += 1
    return {"count": len(lengths), "bucket_boundaries": list(boundaries), "bucket_counts": buckets}


def save_dataset_split(
    split_name: str,
    original_rows: list[dict[str, Any]],
    sampled_rows: list[dict[str, Any]],
    output_dir: str | Path,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    dataset_path = output_dir / f"dataset_{split_name}.json"
    metadata_path = output_dir / f"dataset_{split_name}_metadata.json"
    dump_json(
        {
            "original": original_rows,
            "sampled": sampled_rows,
        },
        dataset_path,
    )
    dump_json(metadata, metadata_path)
    return dataset_path, metadata_path
