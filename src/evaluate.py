from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.utils import dump_json, ensure_dir


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    labels = labels[order]
    scores = scores[order]
    positives = max(1, int(labels.sum()))
    negatives = max(1, int((labels == 0).sum()))
    tps = np.cumsum(labels == 1)
    fps = np.cumsum(labels == 0)
    tpr = np.concatenate([[0.0], tps / positives, [1.0]])
    fpr = np.concatenate([[0.0], fps / negatives, [1.0]])
    thresholds = np.concatenate([[scores[0] + 1e-6], scores, [scores[-1] - 1e-6]]) if len(scores) else np.array([1.0, 0.0])
    return fpr, tpr, thresholds


def pr_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    labels = labels[order]
    positives = max(1, int(labels.sum()))
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    precision = np.concatenate([[1.0], tp / np.maximum(tp + fp, 1)])
    recall = np.concatenate([[0.0], tp / positives])
    return recall, precision


def auc(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc(fpr, tpr)


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    recall, precision = pr_curve(labels, scores)
    return auc(recall, precision)


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float((labels == predictions).mean())


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    tpr = float((predictions[positives] == 1).mean()) if positives.any() else 0.0
    tnr = float((predictions[negatives] == 0).mean()) if negatives.any() else 0.0
    return (tpr + tnr) / 2.0


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = tpr[fpr <= target_fpr]
    if len(valid) == 0:
        return 0.0
    return float(valid.max())


def bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    if n_bootstrap <= 1 or len(labels) == 0:
        value = metric_fn(labels, scores)
        return value, value
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels), size=len(labels))
        sample_labels = labels[indices]
        sample_scores = scores[indices]
        if sample_labels.min() == sample_labels.max():
            continue
        values.append(metric_fn(sample_labels, sample_scores))
    if not values:
        value = metric_fn(labels, scores)
        return value, value
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(np.int64)
    metrics = {
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": pr_auc(labels, scores),
        "accuracy": accuracy(labels, predictions),
        "balanced_accuracy": balanced_accuracy(labels, predictions),
        "tpr_at_fpr_1pct": tpr_at_fpr(labels, scores, 0.01),
        "tpr_at_fpr_5pct": tpr_at_fpr(labels, scores, 0.05),
        "tpr_at_fpr_10pct": tpr_at_fpr(labels, scores, 0.10),
        "threshold": float(threshold),
    }
    for metric_name in ["roc_auc", "pr_auc"]:
        metric_fn = roc_auc if metric_name == "roc_auc" else pr_auc
        low, high = bootstrap_ci(labels, scores, metric_fn, bootstrap_samples, bootstrap_seed)
        metrics[f"{metric_name}_ci95"] = [low, high]
    return metrics


def save_metrics_json(metrics: dict[str, Any], path: str | Path) -> None:
    dump_json(metrics, path)


def save_markdown_table(rows: list[dict[str, Any]], path: str | Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        rendered = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _svg_canvas(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" fill="white" />'
        f"{body}</svg>"
    )


def _plot_frame(width: int, height: int, title: str) -> tuple[float, float, float, float, list[str]]:
    margin_left = 60.0
    margin_right = 20.0
    margin_top = 40.0
    margin_bottom = 45.0
    inner_width = width - margin_left - margin_right
    inner_height = height - margin_top - margin_bottom
    body = [
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-family="Arial" font-size="16">{title}</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="black" />',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="black" />',
    ]
    return margin_left, margin_top, inner_width, inner_height, body


def save_line_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    path: str | Path,
    title: str,
    x_label: str,
    y_label: str,
    color: str = "#1f77b4",
) -> None:
    width, height = 720, 480
    margin_left, margin_top, inner_width, inner_height, body = _plot_frame(width, height, title)
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)
    points = []
    for x_value, y_value in zip(x_values, y_values):
        px = margin_left + ((x_value - x_min) / x_span) * inner_width
        py = margin_top + inner_height - ((y_value - y_min) / y_span) * inner_height
        points.append(f"{px:.2f},{py:.2f}")
    body.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}" />')
    body.append(
        f'<text x="{width / 2:.1f}" y="{height - 10}" text-anchor="middle" font-family="Arial" font-size="12">{x_label}</text>'
    )
    body.append(
        f'<text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="12">{y_label}</text>'
    )
    Path(path).write_text(_svg_canvas(width, height, "".join(body)), encoding="utf-8")


def save_histogram(
    human_scores: np.ndarray,
    machine_scores: np.ndarray,
    path: str | Path,
    title: str,
    bins: int = 20,
) -> None:
    width, height = 720, 480
    margin_left, margin_top, inner_width, inner_height, body = _plot_frame(width, height, title)
    all_scores = np.concatenate([human_scores, machine_scores]) if len(human_scores) and len(machine_scores) else np.array([0.0, 1.0])
    counts_h, edges = np.histogram(human_scores, bins=bins, range=(all_scores.min(), all_scores.max()))
    counts_m, _ = np.histogram(machine_scores, bins=edges)
    max_count = max(1, int(max(counts_h.max(initial=0), counts_m.max(initial=0))))
    bar_width = inner_width / bins
    for idx in range(bins):
        x = margin_left + idx * bar_width
        human_height = (counts_h[idx] / max_count) * inner_height
        machine_height = (counts_m[idx] / max_count) * inner_height
        body.append(
            f'<rect x="{x:.2f}" y="{margin_top + inner_height - human_height:.2f}" width="{bar_width * 0.45:.2f}" height="{human_height:.2f}" fill="#4c78a8" fill-opacity="0.65" />'
        )
        body.append(
            f'<rect x="{x + bar_width * 0.5:.2f}" y="{margin_top + inner_height - machine_height:.2f}" width="{bar_width * 0.45:.2f}" height="{machine_height:.2f}" fill="#f58518" fill-opacity="0.65" />'
        )
    body.append('<text x="570" y="70" font-family="Arial" font-size="12" fill="#4c78a8">human</text>')
    body.append('<text x="570" y="90" font-family="Arial" font-size="12" fill="#f58518">machine</text>')
    Path(path).write_text(_svg_canvas(width, height, "".join(body)), encoding="utf-8")


def save_bar_plot(labels: list[str], values: list[float], path: str | Path, title: str, y_label: str) -> None:
    width, height = 720, 480
    margin_left, margin_top, inner_width, inner_height, body = _plot_frame(width, height, title)
    max_value = max(values) if values else 1.0
    bar_width = inner_width / max(1, len(labels))
    for idx, (label, value) in enumerate(zip(labels, values)):
        bar_height = (value / max_value) * inner_height if max_value > 0 else 0
        x = margin_left + idx * bar_width + 8
        y = margin_top + inner_height - bar_height
        body.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 16:.2f}" height="{bar_height:.2f}" fill="#54a24b" />'
        )
        body.append(
            f'<text x="{x + (bar_width - 16) / 2:.2f}" y="{height - 20}" text-anchor="middle" font-family="Arial" font-size="11">{label}</text>'
        )
        body.append(
            f'<text x="{x + (bar_width - 16) / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-family="Arial" font-size="10">{value:.3f}</text>'
        )
    body.append(
        f'<text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="12">{y_label}</text>'
    )
    Path(path).write_text(_svg_canvas(width, height, "".join(body)), encoding="utf-8")


def save_detector_artifacts(
    output_dir: str | Path,
    detector_name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    metrics: dict[str, Any],
    score_key: str = "score",
) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    plots_dir = ensure_dir(Path(output_dir) / "plots")
    fpr, tpr, _ = roc_curve(labels, scores)
    recall, precision = pr_curve(labels, scores)
    human_scores = scores[labels == 0]
    machine_scores = scores[labels == 1]
    roc_path = plots_dir / f"{detector_name}_roc.svg"
    pr_path = plots_dir / f"{detector_name}_pr.svg"
    hist_path = plots_dir / f"{detector_name}_hist.svg"
    save_line_plot(fpr, tpr, roc_path, f"{detector_name} ROC", "FPR", "TPR")
    save_line_plot(recall, precision, pr_path, f"{detector_name} PR", "Recall", "Precision")
    save_histogram(human_scores, machine_scores, hist_path, f"{detector_name} {score_key} histogram")
    metrics_path = Path(output_dir) / f"{detector_name}_metrics.json"
    save_metrics_json(metrics, metrics_path)
    return {
        "roc_plot": str(roc_path),
        "pr_plot": str(pr_path),
        "hist_plot": str(hist_path),
        "metrics_json": str(metrics_path),
    }
