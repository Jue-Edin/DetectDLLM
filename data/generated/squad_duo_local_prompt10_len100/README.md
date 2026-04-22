# Generated data view: squad_duo_local_prompt10_len100

This folder is a readable export of the local DetectDLLM dataset.
It is created automatically by `scripts/build_dataset.py`.

- source human corpus: `data/human/squad_gemma-2b-instruct_rewrite.original.json`
- source dataset artifact directory: `outputs\paper_prompt_local\datasets\squad_duo_local_prompt10_len100`
- prompt mode: `fixed_tokens`

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
