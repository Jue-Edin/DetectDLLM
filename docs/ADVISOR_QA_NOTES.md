# Advisor Q&A notes for the current DUO detector pilot

## 1. How do we know the probability is gathered at the right position and token id?

The implementation uses the same tokenizer to create the dataset token ids and to gather DUO logits. For each corrupted example:

1. `mask_random_positions` creates a boolean mask with the same shape as `input_ids`.
2. The corrupted sequence is a clone of the original ids where only `mask == True` positions are replaced by `tokenizer.mask_token_id`.
3. DUO returns logits with shape `[batch, seq_len, vocab_size]`.
4. `run_duo_analytic.py` computes `log_probs = log_softmax(logits, dim=-1)` and gathers
   `log_probs[position, original_ids[position]]`.
5. It then multiplies by the boolean mask, so only the masked positions contribute.

Run this explicit audit locally:

```bash
python scripts/audit_duo_probability_alignment.py \
  --config configs/paper_prompt_local.json \
  --split val \
  --label machine \
  --example-index 0
```

The script writes:

```text
outputs/debug/probability_alignment/probability_alignment_report.json
outputs/debug/probability_alignment/probability_alignment_report.md
```

It checks the exact equality between direct indexing
`log_probs[position, original_token_id]` and the gathered value used by the detector.

## 2. Mask token consistency

The code uses the tokenizer's actual `mask_token_id`; it does not manually type `<MASK>`. In the current manifest the tokenizer/model audit is:

```json
{
  "model_vocab_size": 50258,
  "tokenizer_total_vocab_size": 50258,
  "mask_token": "<|mask|>",
  "mask_token_id": 50257,
  "expected_mask_token_id": 50257,
  "mask_token_matches_expected_last_id": true,
  "status": "aligned"
}
```

The updated adapter hard-fails when this audit is not aligned.

## 3. Why can plain log-prob detect corrupted-looking machine text while the analytic statistic fails?

Plain mean log-probability asks whether DUO finds the observed token sequence likely. It strongly penalizes garbled continuations.

The analytic statistic subtracts the model's own expected log-probability and divides by its own variance under the same corrupted context. If the context is already strange or high entropy, the reference mean decreases and the reference variance increases. This can normalize away part of the obvious "garbage" signal. Therefore, the current result is not mainly a probability-gathering bug; it suggests the analytic statistic, as currently adapted, is not yet the right detector for this DUO setup.

## 4. Temperature=0 / greedy generation

The adapter treats `strategy="greedy"` or `temperature <= 0` as argmax decoding. Run:

```bash
python scripts/run_full_pipeline.py --config configs/paper_prompt_local_temp0.json
```

This also uses `window_stride=50` and `max_windows_per_source=2` to make it easier to get exactly 100 pairs.
