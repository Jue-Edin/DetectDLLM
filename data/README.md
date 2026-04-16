# Data

`data/human/` stores human-written source passages used to build the local
human-vs-DUO dataset.

In this repository we include one small local experiment file:

- `data/human/squad_gemma-2b-instruct_rewrite.original.json`

The loader accepts `.txt`, `.json`, `.jsonl`, and `.csv`. For JSON files it
now supports top-level lists as well as dictionaries with keys such as
`original`, `original_only`, `human`, `texts`, `documents`, or `data`.
