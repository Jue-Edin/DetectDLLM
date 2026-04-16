import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.duo_adapter import DuoAdapter
from src.evaluate import compute_metrics, roc_auc, save_bar_plot, save_detector_artifacts, save_markdown_table
from src.utils import configure_logging, dump_json, dump_jsonl, ensure_dir, load_json, stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the analytic DUO detector.')
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--checkpoint-dir', default=None)
    parser.add_argument('--tokenizer-dir', default=None)
    parser.add_argument('--allow-online-tokenizer', action='store_true')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--mask-ratios', nargs='*', type=float, default=None)
    parser.add_argument('--corruption-seeds', nargs='*', type=int, default=None)
    parser.add_argument('--bootstrap-samples', type=int, default=None)
    parser.add_argument('--eps', type=float, default=1e-6)
    return parser.parse_args()


def _resolve(args: argparse.Namespace, config: dict[str, Any], key: str, default: Any) -> Any:
    value = getattr(args, key.replace('-', '_'), None)
    if value is not None:
        return value
    return config.get(key, default)


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


def _cache_path(cache_dir: str | Path, payload: dict[str, Any]) -> Path:
    return Path(cache_dir) / f"{stable_hash(payload)}.json"


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


def _choose_orientation(labels: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray, float]:
    auc_forward = roc_auc(labels, scores)
    auc_reverse = roc_auc(labels, -scores)
    if auc_reverse > auc_forward:
        return -1.0, -scores, float(auc_reverse)
    return 1.0, scores, float(auc_forward)


def compute_analytic_record(
    adapter: DuoAdapter,
    token_ids: list[int],
    example_id: str,
    label: str,
    split: str,
    mask_ratio: float,
    corruption_seed: int,
    cache_dir: str | Path,
    eps: float = 1e-6,
) -> dict[str, Any]:
    cache_key = {
        'example_id': example_id,
        'label': label,
        'split': split,
        'mask_ratio': mask_ratio,
        'corruption_seed': corruption_seed,
        'token_hash': stable_hash(token_ids),
        'detector_method': 'analytic',
    }
    cache_path = _cache_path(cache_dir, cache_key)
    if cache_path.exists():
        return load_json(cache_path)

    original_ids = torch.tensor(token_ids, dtype=torch.long)
    corruption = adapter.corrupt_ids(original_ids, mask_ratio=mask_ratio, seed=corruption_seed)
    corrupted_ids = corruption.corrupted_ids.squeeze(0)
    mask = corruption.mask.squeeze(0)
    if int(mask.sum().item()) <= 0:
        raise ValueError('The corruption mask is empty; analytic detector requires at least one masked token.')

    timesteps = mask.unsqueeze(0).sum(dim=1).float() / max(1, mask.shape[0])
    logits = adapter.reconstruct_logits(corrupted_ids, timesteps=timesteps).squeeze(0)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    device = log_probs.device
    original_ids_device = original_ids.to(device)
    mask_device = mask.to(device)
    mask_float = mask_device.float()
    mask_count = mask_float.sum().clamp_min(1.0)

    observed_token_logprob = log_probs.gather(dim=-1, index=original_ids_device.unsqueeze(-1)).squeeze(-1)
    observed_sum_logprob = float((observed_token_logprob * mask_float).sum().detach().cpu().item())
    observed_mean_logprob = float((observed_token_logprob * mask_float).sum().div(mask_count).detach().cpu().item())

    ref_token_mean = (probs * log_probs).sum(dim=-1)
    ref_token_second_moment = (probs * torch.square(log_probs)).sum(dim=-1)
    ref_token_var = (ref_token_second_moment - torch.square(ref_token_mean)).clamp_min(0.0)

    reference_sum_mean = float((ref_token_mean * mask_float).sum().detach().cpu().item())
    reference_sum_var = float((ref_token_var * mask_float).sum().detach().cpu().item())
    reference_sum_std = float(np.sqrt(max(reference_sum_var, eps)))
    z_score = float((observed_sum_logprob - reference_sum_mean) / reference_sum_std)

    entropy = -(probs * log_probs).sum(dim=-1)
    mean_entropy = float((entropy * mask_float).sum().div(mask_count).detach().cpu().item())
    mean_max_prob = float((probs.max(dim=-1).values * mask_float).sum().div(mask_count).detach().cpu().item())

    record = {
        'example_id': example_id,
        'label': label,
        'split': split,
        'text_length': len(token_ids),
        'mask_ratio': float(mask_ratio),
        'corruption_seed': int(corruption_seed),
        'num_masked': int(mask.sum().item()),
        'baseline_score': observed_mean_logprob,
        'original_sum_logprob': observed_sum_logprob,
        'original_mean_logprob': observed_mean_logprob,
        'reference_sum_mean': reference_sum_mean,
        'reference_sum_var': reference_sum_var,
        'reference_sum_std': reference_sum_std,
        'mean_entropy': mean_entropy,
        'mean_max_prob': mean_max_prob,
        'z_score': z_score,
        'detector_method': 'analytic',
    }
    dump_json(record, cache_path)
    return record


def score_examples(
    adapter: DuoAdapter,
    examples: list[dict[str, Any]],
    mask_ratios: list[float],
    corruption_seeds: list[int],
    cache_dir: str | Path,
    eps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(examples) * len(mask_ratios) * len(corruption_seeds)
    progress = tqdm(total=total, desc='analytic zero-shot scoring', leave=False)
    for example in examples:
        for mask_ratio in mask_ratios:
            for corruption_seed in corruption_seeds:
                rows.append(
                    compute_analytic_record(
                        adapter=adapter,
                        token_ids=example['token_ids'],
                        example_id=example['example_id'],
                        label=example['label'],
                        split=example['split'],
                        mask_ratio=mask_ratio,
                        corruption_seed=corruption_seed,
                        cache_dir=cache_dir,
                        eps=eps,
                    )
                )
                progress.update(1)
    progress.close()
    return rows


def _aggregate_rows(score_rows: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float, str], list[float]] = defaultdict(list)
    for row in score_rows:
        key = (row['example_id'], row['split'], float(row['mask_ratio']), row['label'])
        grouped[key].append(float(row[score_field]))
    aggregated = []
    for key, values in grouped.items():
        example_id, split, mask_ratio, label = key
        aggregated.append(
            {
                'example_id': example_id,
                'split': split,
                'mask_ratio': mask_ratio,
                'label': label,
                'score': float(np.mean(values)),
                'score_std_across_seeds': float(np.std(values)),
            }
        )
    return aggregated


def run_duo_analytic(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    configure_logging()
    dataset_dir = Path(args.dataset_dir)
    output_root = Path(_resolve(args, config, 'output_root', 'outputs'))
    output_dir = Path(args.output_dir) if args.output_dir else output_root / 'metrics' / f'analytic_{dataset_dir.name}'
    mask_ratios = args.mask_ratios or _resolve(args, config, 'mask_ratios', [0.15, 0.30, 0.50])
    corruption_seeds = args.corruption_seeds or _resolve(args, config, 'corruption_seeds', [11, 17, 23])
    bootstrap_samples = int(_resolve(args, config, 'bootstrap_samples', 200))
    checkpoint_dir = _resolve(args, config, 'checkpoint_dir', 'models/duo-distilled')
    tokenizer_dir = _resolve(args, config, 'tokenizer_dir', 'assets/tokenizer/gpt2')
    allow_online_tokenizer = bool(_resolve(args, config, 'allow_online_tokenizer', False))
    if args.allow_online_tokenizer:
        allow_online_tokenizer = True

    adapter = DuoAdapter(
        checkpoint_dir=checkpoint_dir,
        tokenizer_dir=tokenizer_dir,
        allow_online_tokenizer=allow_online_tokenizer,
    ).load(require_tokenizer=True)
    examples = _load_examples(dataset_dir)
    ensure_dir(output_dir)
    cache_dir = ensure_dir(output_dir / 'cache')

    score_rows = score_examples(
        adapter=adapter,
        examples=examples,
        mask_ratios=mask_ratios,
        corruption_seeds=corruption_seeds,
        cache_dir=cache_dir,
        eps=float(args.eps),
    )
    rows_path = output_dir / 'analytic_rows.jsonl'
    dump_jsonl(score_rows, rows_path)

    results_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for detector_name, score_field in [('baseline', 'baseline_score'), ('duo_analytic', 'z_score')]:
        aggregated = _aggregate_rows(score_rows, score_field=score_field)
        best_setting = None
        best_val_auc = -1.0
        for mask_ratio in sorted(set(row['mask_ratio'] for row in aggregated)):
            val_rows = [row for row in aggregated if row['split'] == 'val' and row['mask_ratio'] == mask_ratio]
            test_rows = [row for row in aggregated if row['split'] == 'test' and row['mask_ratio'] == mask_ratio]
            if not val_rows or not test_rows:
                continue
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
            row = {
                'detector': detector_name,
                'mask_ratio': float(mask_ratio),
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
            }
            results_rows.append(row)
            if detector_name == 'duo_analytic' and val_auc > best_val_auc:
                best_val_auc = float(val_auc)
                best_setting = (mask_ratio, score_sign, threshold, test_labels, test_scores, metrics)
        if detector_name == 'duo_analytic' and best_setting is not None:
            mask_ratio, score_sign, threshold, labels, scores, metrics = best_setting
            summary['best_duo_analytic'] = {
                'mask_ratio': float(mask_ratio),
                'score_sign': float(score_sign),
                'threshold': float(threshold),
                'selection': 'highest validation ROC-AUC',
                'artifacts': save_detector_artifacts(
                    output_dir=output_dir,
                    detector_name='duo_analytic',
                    labels=labels,
                    scores=scores,
                    metrics=metrics,
                    score_key='duo_analytic_score',
                ),
            }

    summary_path = output_dir / 'analytic_summary.json'
    dump_json(
        {
            'rows': results_rows,
            'summary': summary,
            'detector_method': 'analytic',
            'tokenizer_load_report': adapter.tokenizer_load_report,
            'tokenizer_vocab_audit': adapter.vocab_audit,
        },
        summary_path,
    )
    save_markdown_table(results_rows, output_dir / 'analytic_results.md', 'Analytic DUO Detector Results')

    duo_rows = [row for row in results_rows if row['detector'] == 'duo_analytic']
    if duo_rows:
        save_bar_plot(
            labels=[f"{row['mask_ratio']:.2f}" for row in duo_rows],
            values=[row['roc_auc'] for row in duo_rows],
            path=output_dir / 'plots' / 'analytic_mask_ratio_ablation.svg',
            title='Analytic DUO ROC-AUC by mask ratio',
            y_label='ROC-AUC',
        )
    return {'output_dir': str(output_dir), 'rows_path': str(rows_path), 'summary_path': str(summary_path)}


def main() -> None:
    args = parse_args()
    result = run_duo_analytic(args)
    print(f"Analytic DUO scoring complete: {result['output_dir']}")


if __name__ == '__main__':
    main()
