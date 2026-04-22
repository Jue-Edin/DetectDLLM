# Data

`data/human/` stores human-written source passages used to build the local
human-vs-DUO dataset.

In this repository we include one small local experiment file:

- `data/human/squad_gemma-2b-instruct_rewrite.original.json`

The loader accepts `.txt`, `.json`, `.jsonl`, and `.csv`. For JSON files it
supports top-level lists as well as dictionaries with keys such as
`original`, `original_only`, `human`, `texts`, `documents`, or `data`.

## Generated inspection views

When you run `scripts/build_dataset.py` or `scripts/run_full_pipeline.py`, the
builder now also writes a readable export under:

```text
data/generated/<dataset_tag>/
```

That folder contains:

- `human_train.json`, `human_val.json`, `human_test.json`
- `machine_train.json`, `machine_val.json`, `machine_test.json`
- `paired_train.json`, `paired_val.json`, `paired_test.json`
- `human_all.json`, `machine_all.json`, `paired_all.json`
- `pair_preview.json`
- `manifest.json`

These files are meant for easy human comparison and GitHub browsing.
The full experiment metadata still lives under `outputs/...`.
