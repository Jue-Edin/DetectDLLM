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

## 3. Repository completion work added in the earlier revision

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

## 6. Prompt-conditioned data construction for the DLLM pilot

The prompt-conditioned dataset construction added for the local DLLM pilot is
*inspired by* the data-generation setup used in Fast-DetectGPT, especially the
logic in `baoguangsheng/fast-detect-gpt/scripts/data_builder.py`.

What was borrowed conceptually:

- the idea of creating paired data by taking a short human prefix as the prompt;
- the idea of conditioning the source model on only that prefix and letting it
  complete the remaining tokens;
- the goal of matching the human and machine examples by construction so that
  each pair shares the same prefix and approximately the same total length.

What was **not** copied line-by-line:

- the DUO generation code here does **not** reuse Fast-DetectGPT's
  autoregressive `model.generate(...)` code path;
- the DUO code here does **not** keep Fast-DetectGPT's minimum-word retry loop;
- the DUO code here does **not** use OpenAI/chat prompting templates from that
  repository.

What was changed for this repository:

- the source model is a DLLM (DUO), so continuation is produced by iterative
  masked denoising rather than autoregressive next-token sampling;
- the prefix can now be specified exactly by token count (`prompt_tokens`) or by
  fraction (`prompt_fraction`);
- the human text can be truncated into fixed token windows
  (`target_total_tokens`) before generation, which makes it easy to build
  experiments such as "keep 10 tokens, generate the remaining 90";
- the builder now stores `prompt_text`, `prompt_token_ids`, window boundaries,
  and continuation metadata in each example record so the provenance of each
  pair is explicit;
- generation can now consume exact prompt token ids
  (`prompt_token_id_seqs`) to avoid decode/re-encode drift and guarantee that
  the DLLM is conditioned on the intended prefix tokens exactly.

In short: the **experimental protocol idea** (human prefix -> model completion ->
paired comparison) is borrowed from Fast-DetectGPT, but the **actual DUO
generation implementation** and the fixed-token prompt/window controls are newly
implemented for this repository.

## 7. New changes in this revision

This revision adds several project-specific usability changes.
These are **newly written for this repository** and are **not copied** from the
Fast-DetectGPT or DUO repositories.

### 7.1 Explicit A/B/C experiment organization

Files changed:

- `scripts/run_duo_analytic.py`
- `scripts/run_full_pipeline.py`
- `docs/EXPERIMENT_DESIGN.md`
- `README.md`

What was added:

- explicit naming of the same-source DUO plain baseline as
  `duo_plain_meanlogprob`;
- saving artifacts for both the plain DUO baseline and the analytic DUO
  detector;
- writing an A/B/C comparison report under
  `outputs/.../reports/<dataset_tag>/experiment_comparison.{json,md}`.

What was **not** borrowed:

- there is no copied experiment-comparison reporter from upstream;
- the A/B/C naming and report layout are local project decisions.

### 7.2 Readable dataset export under `data/generated/`

Files changed:

- `scripts/build_dataset.py`
- `data/README.md`
- `README.md`

What was added:

- automatic export of simplified human records, machine records, and paired
  records under `data/generated/<dataset_tag>/`;
- combined files such as `human_all.json`, `machine_all.json`, and
  `paired_all.json`;
- a readable preview file and a small folder-level README.

Purpose of this change:

- make it easy to inspect human/machine pairs side by side;
- keep the repository layout understandable when uploaded to GitHub;
- separate readable data views from the heavier `outputs/...` experiment
  artifacts.

What was **not** borrowed:

- no upstream repo provided this exact export format;
- the folder layout and JSON schemas are newly designed for this repository.

### 7.3 Small configuration cleanup

Files changed:

- `configs/paper_local.json`
- `configs/paper_prompt_local.json`
- `configs/default.json`

What was added or adjusted:

- default readable data export settings;
- clearer local-file handling for the local GPT-2 baseline config;
- cleaner propagation of dataset tags through the full pipeline.

These are local engineering changes only, not borrowed algorithmic content.
