from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import compute_metrics


def test_compute_metrics_keys() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    metrics = compute_metrics(labels=labels, scores=scores, threshold=0.5, bootstrap_samples=4, bootstrap_seed=0)
    assert 'roc_auc' in metrics
    assert 'pr_auc' in metrics
    assert 'balanced_accuracy' in metrics
    assert 'tpr_at_fpr_1pct' in metrics
