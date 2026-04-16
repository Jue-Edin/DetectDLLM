# Human vs Machine Statistics Summary

This report compares human and machine texts at two levels:

1. raw text statistics computed from the dataset itself;
2. detector score statistics computed from saved row-level outputs.

Higher oriented detector scores always mean *more machine-like*.

## Raw text statistics

- saved in `text_stats_by_label.md` and `text_stats_comparison.md`

## Analytic DUO detector score statistics

- baseline selected mask ratio: 0.30
- baseline validation ROC-AUC used for orientation selection: 1.0000
- duo_analytic selected mask ratio: 0.30
- duo_analytic validation ROC-AUC used for orientation selection: 0.8700

## Fast-DetectGPT score statistics

- validation ROC-AUC used for orientation selection: 1.0000

## How to read the tables

- `machine_minus_human > 0` means the machine class tends to have larger values.
- `cohen_d` measures standardized mean difference.
- `cliffs_delta` is a rank-based effect size in [-1, 1].
