# Local run guide

## 1. Install dependencies

### PowerShell

```powershell
cd DetectDLLM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

### Bash

```bash
cd DetectDLLM
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2. Place the DUO model locally

Make sure the following path exists:

```text
models/duo-distilled/
```

with the checkpoint files described in `models/README.md`.

## 3. Check the setup

```bash
python scripts/check_local_setup.py --config configs/paper_local.json
```

## 4. Run the full local experiment

```bash
python scripts/run_full_pipeline.py --config configs/paper_local.json
```

This will:

1. build a local human-vs-DUO dataset,
2. run the analytic DUO detector,
3. run the Fast-DetectGPT surrogate baseline.

## 5. Optional smoke test

```bash
pytest -q
```

## 6. Important output paths

- dataset: `outputs/paper_local/datasets/squad_duo_local/`
- analytic detector: `outputs/paper_local/metrics/analytic_squad_duo_local/`
- Fast-DetectGPT baseline: `outputs/paper_local/metrics/fastdetectgpt_squad_duo_local_gpt2/`
