# DetectDLLM

A lightweight local repository for a first-pass experiment on **detecting DUO/DLLM-generated text**.

This repo is designed for a small, reproducible, local experiment before moving to larger server-side runs. It assumes that the DUO checkpoint is already available locally under `models/duo-distilled/` and does **not** attempt to store large model weights in git.

## What this repository does

1. builds a balanced **human vs DUO-generated** dataset from a local human corpus;
2. optionally builds a **prompt-paired** version where DUO is conditioned on a short human prefix;
3. runs three detectors on the **same DUO-generated texts**:
   - **Experiment A**: the proposed DLLM-specific analytic detector under DUO;
   - **Experiment B**: a Fast-DetectGPT **surrogate** baseline using a local AR model such as GPT-2;
   - **Experiment C**: a plain DUO masked-token mean log-probability baseline.
4. exports a readable copy of the generated pairs under `data/generated/` for easy inspection.

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
  gpt2/
    config.json
    generation_config.json
    merges.txt
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    vocab.json
```

The code loads the DUO checkpoint from `models/duo-distilled/` by default.
The GPT-2 files are only needed when you also want to run **Experiment B**.

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

### 3. Run the default local pipeline

```bash
python scripts/run_full_pipeline.py --config configs/paper_local.json
```

### 4. Run the prompt-paired pilot

```bash
python scripts/run_full_pipeline.py --config configs/paper_prompt_local.json
```

### 5. Run only Experiments A and C

```bash
python scripts/run_full_pipeline.py --config configs/paper_prompt_local.json --skip-fastdetectgpt
```

## Main outputs

- dataset artifacts: `outputs/.../datasets/<dataset_tag>/`
- DUO detector results: `outputs/.../metrics/analytic_<dataset_tag>/`
- Fast-DetectGPT baseline results: `outputs/.../metrics/fastdetectgpt_<dataset_tag>_gpt2/`
- experiment comparison table: `outputs/.../reports/<dataset_tag>/experiment_comparison.md`
- readable human/machine pairs: `data/generated/<dataset_tag>/`

## Main scripts

- `scripts/build_dataset.py`: build the local human-vs-DUO dataset and export readable pair files.
- `scripts/run_duo_analytic.py`: run Experiment A and Experiment C.
- `scripts/run_fastdetectgpt_baseline.py`: run Experiment B.
- `scripts/run_full_pipeline.py`: run the whole experiment end to end and write an A/B/C comparison table.
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
- The prompt-paired pilot uses a **human text prefix**, not a chat-style instruction prompt.

## Documentation

- `docs/LOCAL_RUN.md`
- `docs/EXPERIMENT_DESIGN.md`
- `docs/REPO_LAYOUT.md`
- `docs/PROMPT_PAIRED_LOCAL_PILOT.md`
- `docs/BORROWING_AND_MODIFICATIONS.md`

## Upstream references

- DUO official repository: see the upstream `s-sahoo/duo` project.
- DUO distilled checkpoint: see `s-sahoo/duo-distilled` on Hugging Face.
- Fast-DetectGPT: see `baoguangsheng/fast-detect-gpt`.

## License

MIT. See `LICENSE`.
