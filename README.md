# DetectDLLM

A lightweight local repository for a first-pass experiment on **detecting DUO/DLLM-generated text**.

This repo is designed for a small, reproducible, local experiment before moving to larger server-side runs. It assumes that the DUO checkpoint is already available locally under `models/duo-distilled/` and does **not** attempt to store large model weights in git.

## What this repository does

1. builds a balanced **human vs DUO-generated** dataset from a local human corpus;
2. runs a **DLLM-specific analytic detector** based on masked-token reverse-conditionals under DUO;
3. compares against two baselines:
   - plain masked-token mean log-probability under DUO;
   - a Fast-DetectGPT surrogate baseline using a local or online AR model.

## Expected local model layout

```text
models/
  duo-distilled/
    config.json
    config.py
    merges.txt
    model.py
    model.safetensors
    README.md
    tokenizer.json
    tokenizer_config.json
    tokenizer_metadata.json
    vocab.json
```

The code loads the checkpoint from `models/duo-distilled/` by default.

## Included local human data

This repository includes one small human corpus for the local pilot experiment:

```text
data/human/squad_gemma-2b-instruct_rewrite.original.json
```

The loader reads this file directly.

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Verify local setup

```bash
python scripts/check_local_setup.py --config configs/paper_local.json
```

### 3. Run the full pipeline

```bash
python scripts/run_full_pipeline.py --config configs/paper_local.json
```

### 4. Main outputs

- dataset: `outputs/paper_local/datasets/squad_duo_local/`
- analytic detector summary: `outputs/paper_local/metrics/analytic_squad_duo_local/analytic_summary.json`
- analytic detector table: `outputs/paper_local/metrics/analytic_squad_duo_local/analytic_results.md`
- Fast-DetectGPT summary: `outputs/paper_local/metrics/fastdetectgpt_squad_duo_local_gpt2/fastdetectgpt_summary.json`

## Main scripts

- `scripts/build_dataset.py`: build the local human-vs-DUO dataset.
- `scripts/run_duo_analytic.py`: run the analytic DLLM detector and the plain DUO baseline.
- `scripts/run_fastdetectgpt_baseline.py`: run the Fast-DetectGPT surrogate baseline.
- `scripts/run_full_pipeline.py`: run the whole experiment end to end.
- `scripts/check_local_setup.py`: validate that the required local files exist.

## Method summary

For each text `X0`, we construct a one-step forward surrogate `X1'` by randomly masking a subset of tokens. DUO then provides reverse-process logits at the masked positions. We compute:

- an observed masked-token log-probability for the original text;
- the conditional mean and variance implied by DUO's own masked-position distribution;
- a normalized analytic score.

This is the DLLM-specific detector used in `scripts/run_duo_analytic.py`.

## Design choices in this repo

- The first public experiment is intentionally **small and local**.
- Hyperparameter choice is validation-only.
- The main detector uses **analytic normalization** rather than Monte Carlo reconstruction variance estimation.
- The repository keeps the large DUO checkpoint outside git.

## Documentation

- `docs/LOCAL_RUN.md`
- `docs/EXPERIMENT_DESIGN.md`
- `docs/REPO_LAYOUT.md`
- `docs/BORROWING_AND_MODIFICATIONS.md`

## Upstream references

- DUO official repository: see the upstream `s-sahoo/duo` project.
- DUO distilled checkpoint: see `s-sahoo/duo-distilled` on Hugging Face.
- Fast-DetectGPT: see `baoguangsheng/fast-detect-gpt`.

## License

MIT. See `LICENSE`.
