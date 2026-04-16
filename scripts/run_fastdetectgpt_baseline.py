import argparse
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import compute_metrics, save_detector_artifacts, save_markdown_table
from src.utils import configure_logging, dump_json, dump_jsonl, ensure_dir, load_json


# The core function below is adapted from Fast-DetectGPT's MIT-licensed implementation:
# baoguangsheng/fast-detect-gpt, file scripts/fast_detect_gpt.py, function
# get_sampling_discrepancy_analytic. We keep the same analytic normalization idea
# but wrap it in a standalone local script that can evaluate your DUO dataset.
def get_sampling_discrepancy_analytic(
    logits_ref: torch.Tensor,
    logits_score: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    if logits_ref.shape[0] != 1 or logits_score.shape[0] != 1 or labels.shape[0] != 1:
        raise ValueError('This helper expects a single example at a time.')
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]
    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).clamp_min(1e-6).sqrt()
    discrepancy = discrepancy.mean()
    return float(discrepancy.item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a Fast-DetectGPT baseline on the local DUO dataset.')
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--sampling-model-name-or-path', default=None)
    parser.add_argument('--scoring-model-name-or-path', default=None)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--max-length', type=int, default=None)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--bootstrap-samples', type=int, default=None)
    return parser.parse_args()


def _resolve_device(device: str) -> str:
    if device != 'auto':
        return device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _sanitize_name(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', name)


def _load_examples(dataset_dir: str | Path) -> list[dict[str, Any]]:
    dataset_dir = Path(dataset_dir)
    examples = []
    for metadata_path in sorted(dataset_dir.glob('dataset_*_metadata.json')):
        metadata = load_json(metadata_path)
        examples.extend(metadata.get('original_records', []))
        examples.extend(metadata.get('sampled_records', []))
    if not examples:
        raise FileNotFoundError(
            f"No dataset metadata found in '{dataset_dir}'. Run scripts/build_dataset.py first."
        )
    return examples


def _best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(scores)
    if len(candidates) == 0:
        return 0.0
    best_threshold = float(candidates[0])
    best_value = -1.0
    for threshold in candidates:
        predictions = (scores >= threshold).astype(np.int64)
        positives = labels == 1
        negatives = labels == 0
        tpr = float((predictions[positives] == 1).mean()) if positives.any() else 0.0
        tnr = float((predictions[negatives] == 0).mean()) if negatives.any() else 0.0
        value = (tpr + tnr) / 2.0
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order]
    scores = scores[order]
    positives = max(1, int(labels.sum()))
    negatives = max(1, int((labels == 0).sum()))
    tps = np.cumsum(labels == 1)
    fps = np.cumsum(labels == 0)
    tpr = np.concatenate([[0.0], tps / positives, [1.0]])
    fpr = np.concatenate([[0.0], fps / negatives, [1.0]])
    return float(np.trapezoid(tpr, fpr))


def _choose_orientation(labels: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray, float]:
    auc_forward = _roc_auc(labels, scores)
    auc_reverse = _roc_auc(labels, -scores)
    if auc_reverse > auc_forward:
        return -1.0, -scores, float(auc_reverse)
    return 1.0, scores, float(auc_forward)


def _score_text(
    text: str,
    scoring_tokenizer,
    scoring_model,
    device: str,
    max_length: int,
    sampling_tokenizer=None,
    sampling_model=None,
) -> float:
    tokenized = scoring_tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
        padding=True,
        return_token_type_ids=False,
    ).to(device)
    labels = tokenized.input_ids[:, 1:]
    with torch.inference_mode():
        logits_score = scoring_model(**tokenized).logits[:, :-1]
    if sampling_model is None:
        logits_ref = logits_score
    else:
        tokenized_ref = sampling_tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
            padding=True,
            return_token_type_ids=False,
        ).to(device)
        if not torch.all(tokenized_ref.input_ids[:, 1:] == labels):
            raise ValueError('Sampling/scoring tokenizers produced different labels; use matching tokenization.')
        with torch.inference_mode():
            logits_ref = sampling_model(**tokenized_ref).logits[:, :-1]
    return get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)


def run_fastdetectgpt_baseline(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    configure_logging()
    dataset_dir = Path(args.dataset_dir)
    output_root = Path(config.get('output_root', 'outputs'))
    baseline_cfg = config.get('fastdetectgpt', {})
    sampling_name = args.sampling_model_name_or_path or baseline_cfg.get('sampling_model_name_or_path', 'gpt2')
    scoring_name = args.scoring_model_name_or_path or baseline_cfg.get('scoring_model_name_or_path') or sampling_name
    local_files_only = bool(args.local_files_only or baseline_cfg.get('local_files_only', False))
    max_length = int(args.max_length if args.max_length is not None else baseline_cfg.get('max_length', 512))
    model_tag = _sanitize_name(Path(scoring_name).name or str(scoring_name))
    output_dir = Path(args.output_dir) if args.output_dir else output_root / 'metrics' / f'fastdetectgpt_{dataset_dir.name}_{model_tag}'
    ensure_dir(output_dir)
    device = _resolve_device(args.device)
    bootstrap_samples = int(args.bootstrap_samples if args.bootstrap_samples is not None else config.get('bootstrap_samples', 200))

    scoring_tokenizer = AutoTokenizer.from_pretrained(scoring_name, local_files_only=local_files_only)
    if scoring_tokenizer.pad_token is None:
        scoring_tokenizer.pad_token = scoring_tokenizer.eos_token
    scoring_model = AutoModelForCausalLM.from_pretrained(scoring_name, local_files_only=local_files_only).to(device)
    scoring_model.eval()

    sampling_tokenizer = None
    sampling_model = None
    if sampling_name != scoring_name:
        sampling_tokenizer = AutoTokenizer.from_pretrained(sampling_name, local_files_only=local_files_only)
        if sampling_tokenizer.pad_token is None:
            sampling_tokenizer.pad_token = sampling_tokenizer.eos_token
        sampling_model = AutoModelForCausalLM.from_pretrained(sampling_name, local_files_only=local_files_only).to(device)
        sampling_model.eval()

    examples = _load_examples(dataset_dir)
    rows = []
    progress = tqdm(examples, desc='fast-detectgpt baseline', leave=False)
    for example in progress:
        score = _score_text(
            text=example['text'],
            scoring_tokenizer=scoring_tokenizer,
            scoring_model=scoring_model,
            device=device,
            max_length=max_length,
            sampling_tokenizer=sampling_tokenizer,
            sampling_model=sampling_model,
        )
        rows.append(
            {
                'example_id': example['example_id'],
                'label': example['label'],
                'split': example['split'],
                'score': float(score),
                'text_length': int(len(example['token_ids'])),
            }
        )

    rows_path = output_dir / 'fastdetectgpt_rows.jsonl'
    dump_jsonl(rows, rows_path)

    val_rows = [row for row in rows if row['split'] == 'val']
    test_rows = [row for row in rows if row['split'] == 'test']
    if not val_rows or not test_rows:
        raise ValueError('Fast-DetectGPT baseline needs non-empty val and test splits.')
    val_labels = np.array([1 if row['label'] == 'machine' else 0 for row in val_rows], dtype=np.int64)
    val_scores_raw = np.array([row['score'] for row in val_rows], dtype=np.float64)
    score_sign, val_scores, val_auc = _choose_orientation(val_labels, val_scores_raw)
    threshold = _best_threshold(val_labels, val_scores)
    test_labels = np.array([1 if row['label'] == 'machine' else 0 for row in test_rows], dtype=np.int64)
    test_scores_raw = np.array([row['score'] for row in test_rows], dtype=np.float64)
    test_scores = score_sign * test_scores_raw
    metrics = compute_metrics(
        labels=test_labels,
        scores=test_scores,
        threshold=threshold,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=13,
    )
    summary = {
        'detector': 'fastdetectgpt_surrogate',
        'sampling_model_name_or_path': sampling_name,
        'scoring_model_name_or_path': scoring_name,
        'score_sign': float(score_sign),
        'val_roc_auc': float(val_auc),
        'threshold': float(threshold),
        'metrics': metrics,
        'artifacts': save_detector_artifacts(
            output_dir=output_dir,
            detector_name='fastdetectgpt_surrogate',
            labels=test_labels,
            scores=test_scores,
            metrics=metrics,
            score_key='fastdetectgpt_score',
        ),
    }
    dump_json(summary, output_dir / 'fastdetectgpt_summary.json')
    save_markdown_table([{
        'detector': 'fastdetectgpt_surrogate',
        'score_sign': float(score_sign),
        'val_roc_auc': float(val_auc),
        'roc_auc': float(metrics['roc_auc']),
        'pr_auc': float(metrics['pr_auc']),
        'accuracy': float(metrics['accuracy']),
        'balanced_accuracy': float(metrics['balanced_accuracy']),
        'tpr@1%fpr': float(metrics['tpr_at_fpr_1pct']),
        'tpr@5%fpr': float(metrics['tpr_at_fpr_5pct']),
        'tpr@10%fpr': float(metrics['tpr_at_fpr_10pct']),
        'threshold': float(threshold),
    }], output_dir / 'fastdetectgpt_results.md', 'Fast-DetectGPT Baseline Results')
    return {'output_dir': str(output_dir), 'rows_path': str(rows_path), 'summary_path': str(output_dir / 'fastdetectgpt_summary.json')}


def main() -> None:
    args = parse_args()
    result = run_fastdetectgpt_baseline(args)
    print(f"Fast-DetectGPT baseline complete: {result['output_dir']}")


if __name__ == '__main__':
    main()
