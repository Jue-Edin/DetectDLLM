# Prompt-paired local pilot experiment

This local pilot is intended to answer the advisor's question:
"Can we generate around 100 DLLM examples, and can we make the prompt simple and explicit?"

## Proposed setting

- Human source corpus: `data/human/squad_gemma-2b-instruct_rewrite.original.json`
- For each selected human passage:
  - tokenize with the DUO tokenizer;
  - keep the first `prompt_tokens=10` tokens as the prompt;
  - keep the full passage length fixed at `target_total_tokens=100`;
  - generate the remaining 90 tokens with DUO conditioned only on that 10-token prefix.
- This yields a natural pair:
  - **human**: the original 100-token human window;
  - **machine**: the DUO continuation of the same 10-token prompt to 100 tokens.

## Why this setting is useful

- It directly mirrors the advisor's requested protocol.
- Human and machine texts are matched by prompt and total length.
- The prompt is simple, short, and easy to explain.
- The metadata explicitly stores the prompt and the segment boundaries.

## Command

```bash
python scripts/run_full_pipeline.py --config configs/paper_prompt_local.json
```

## Main dataset artifacts

- dataset directory:
  `outputs/paper_prompt_local/datasets/squad_duo_local_prompt10_len100/`
- manifest with generation settings:
  `outputs/paper_prompt_local/datasets/squad_duo_local_prompt10_len100/manifest.json`
- preview of prompt/human/machine pairs:
  `outputs/paper_prompt_local/datasets/squad_duo_local_prompt10_len100/prompt_pair_preview.json`

## Notes

If you end up with fewer than 100 usable examples because some human passages are
too short after tokenization, increase `max_windows_per_source` to `2` and set
`window_stride` to a smaller value such as `50`. This will extract multiple
100-token windows from longer human passages while still keeping all windows
from the same source text in the same split.
