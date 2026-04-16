# Borrowing and modifications

This document explicitly records what was borrowed, what was adapted, and what
was newly added in this repository revision.

## 1. Fast-DetectGPT baseline

The file `scripts/run_fastdetectgpt_baseline.py` contains an adapted version of
the analytic discrepancy function from the Fast-DetectGPT repository
(`baoguangsheng/fast-detect-gpt`, function
`get_sampling_discrepancy_analytic` in `scripts/fast_detect_gpt.py`).

What was kept from Fast-DetectGPT:

- the analytic normalization idea,
- the use of `log_softmax` on the scoring logits,
- the use of `softmax` on the reference logits,
- the analytic computation of the conditional mean and variance,
- the final normalized discrepancy score.

What was changed for this repository:

- it was wrapped into a standalone local script that reads the dataset format
  produced by `scripts/build_dataset.py`;
- the code now selects score orientation on the validation split only;
- threshold selection is also done on the validation split only;
- outputs are written into repository-local metrics folders with JSON, JSONL,
  Markdown, and SVG artifacts.

## 2. DUO analytic detector

The file `scripts/run_duo_analytic.py` is not a line-by-line copy of the DUO
repository or the Fast-DetectGPT repository.

It is a detector built for this project's DLLM setting. The central change is:

- replacing AR next-token scoring with masked-position scoring under DUO;
- replacing Monte Carlo reconstruction variance estimation with an analytic
  mean/variance computed directly from DUO logits at the masked positions;
- averaging across multiple corruption seeds;
- selecting hyperparameters using validation ROC-AUC only.

## 3. Repository completion work added in this revision

Newly added for usability and reproducibility:

- `README.md` (public, English paper-repo style)
- `requirements.txt`
- `environment.yml`
- `configs/default.json`, `configs/paper_local.json`, `configs/smoke.json`
- `docs/LOCAL_RUN.md`
- `docs/EXPERIMENT_DESIGN.md`
- `docs/REPO_LAYOUT.md`
- `scripts/check_local_setup.py`
- `scripts/run_full_pipeline.py`
- `tests/test_data_loading.py`
- `tests/test_metrics.py`
- support for JSON human corpora whose top-level key is `original`

## 4. What was not copied wholesale

- I did not transplant the full DUO training/sampling framework into this
  detector repository.
- I did not copy the full Fast-DetectGPT experiment pipeline.
- I did not add the large DUO checkpoint files into git.

## 5. Upstream references

- DUO official codebase and sampling examples: `s-sahoo/duo`
- DUO distilled Hugging Face checkpoint: `s-sahoo/duo-distilled`
- Fast-DetectGPT baseline: `baoguangsheng/fast-detect-gpt`
