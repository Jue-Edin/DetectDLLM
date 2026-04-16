# Fast-DetectGPT Human vs Machine Score Comparison

| detector | score_sign | val_roc_auc | split | human_mean | machine_mean | machine_minus_human | cohen_d | cliffs_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fastdetectgpt_surrogate | -1.0000 | 1.0000 | val | -1.1264 | 5.5932 | 6.7197 | 5.4521 | 1.0000 |
| fastdetectgpt_surrogate | -1.0000 | 1.0000 | test | -0.7808 | 6.3178 | 7.0986 | 10.5788 | 1.0000 |
| fastdetectgpt_surrogate | -1.0000 | 1.0000 | overall | -0.9020 | 6.5718 | 7.4738 | 5.7013 | 1.0000 |