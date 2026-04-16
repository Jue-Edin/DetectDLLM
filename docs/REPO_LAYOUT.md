# Repository layout

```text
DetectDLLM/
  README.md
  LICENSE
  CITATION.cff
  requirements.txt
  environment.yml
  configs/
  data/
  docs/
  models/
  outputs/
  scripts/
  src/
  tests/
```

- `configs/`: experiment defaults.
- `data/human/`: human-written source passages.
- `docs/`: experiment notes and reproducibility instructions.
- `models/duo-distilled/`: local DUO checkpoint path expected by the code.
- `outputs/`: generated datasets, metrics, and plots.
- `scripts/`: command-line entry points.
- `src/`: reusable implementation code.
- `tests/`: lightweight tests that do not require the DUO weight file.
