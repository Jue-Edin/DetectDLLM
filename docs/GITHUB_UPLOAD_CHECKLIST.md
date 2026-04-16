# GitHub upload checklist

## Include in git

- source code under `scripts/` and `src/`
- configuration files under `configs/`
- documentation under `docs/`
- the small local human corpus under `data/human/`
- repository metadata: `README.md`, `LICENSE`, `CITATION.cff`, `.gitignore`
- dependency files: `requirements.txt`, `environment.yml`
- lightweight tests under `tests/`

## Do not include in git

- large DUO weight files under `models/duo-distilled/`
- generated outputs under `outputs/`
- local virtual environments or caches

## Local-only files expected outside git

The repository assumes a local directory:

```text
models/duo-distilled/
```

containing the DUO checkpoint files.
