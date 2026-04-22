import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_dataset import run_build_dataset
from scripts.run_duo_analytic import run_duo_analytic
from scripts.run_fastdetectgpt_baseline import run_fastdetectgpt_baseline
from src.evaluate import save_markdown_table
from src.utils import dump_json, ensure_dir, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the full local DetectDLLM pipeline.')
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--skip-fastdetectgpt', action='store_true')
    return parser.parse_args()


def _best_row(summary_payload: dict[str, Any], detector_name: str) -> dict[str, Any] | None:
    rows = summary_payload.get('rows', [])
    candidates = [row for row in rows if row.get('detector') == detector_name]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row.get('val_roc_auc', float('-inf'))))


def _build_experiment_rows(
    *,
    config: dict[str, Any],
    dataset_dir: Path,
    build_result: dict[str, Any],
    duo_result: dict[str, Any],
    fdg_result: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    duo_summary = load_json(duo_result['summary_path'])

    duo_analytic_row = _best_row(duo_summary, 'duo_analytic')
    if duo_analytic_row is not None:
        rows.append(
            {
                'experiment': 'A',
                'detector': 'our_duo_analytic',
                'generator_model': Path(config.get('checkpoint_dir', 'models/duo-distilled')).name,
                'scoring_model': Path(config.get('checkpoint_dir', 'models/duo-distilled')).name,
                'reference_model': Path(config.get('checkpoint_dir', 'models/duo-distilled')).name,
                'dataset_dir': str(dataset_dir),
                'data_export_dir': build_result.get('data_export_dir') or '',
                'val_roc_auc': float(duo_analytic_row['val_roc_auc']),
                'roc_auc': float(duo_analytic_row['roc_auc']),
                'pr_auc': float(duo_analytic_row['pr_auc']),
                'accuracy': float(duo_analytic_row['accuracy']),
                'balanced_accuracy': float(duo_analytic_row['balanced_accuracy']),
                'tpr@1%fpr': float(duo_analytic_row['tpr@1%fpr']),
                'tpr@5%fpr': float(duo_analytic_row['tpr@5%fpr']),
                'tpr@10%fpr': float(duo_analytic_row['tpr@10%fpr']),
                'threshold': float(duo_analytic_row['threshold']),
                'selected_mask_ratio': float(duo_analytic_row['mask_ratio']),
                'notes': 'DLLM-specific white-box detector under DUO.',
            }
        )

    if fdg_result is not None:
        fdg_summary = load_json(fdg_result['summary_path'])
        fdg_metrics = fdg_summary['metrics']
        rows.append(
            {
                'experiment': 'B',
                'detector': 'fastdetectgpt_surrogate',
                'generator_model': Path(config.get('checkpoint_dir', 'models/duo-distilled')).name,
                'scoring_model': Path(str(fdg_summary['scoring_model_name_or_path'])).name,
                'reference_model': Path(str(fdg_summary['sampling_model_name_or_path'])).name,
                'dataset_dir': str(dataset_dir),
                'data_export_dir': build_result.get('data_export_dir') or '',
                'val_roc_auc': float(fdg_summary['val_roc_auc']),
                'roc_auc': float(fdg_metrics['roc_auc']),
                'pr_auc': float(fdg_metrics['pr_auc']),
                'accuracy': float(fdg_metrics['accuracy']),
                'balanced_accuracy': float(fdg_metrics['balanced_accuracy']),
                'tpr@1%fpr': float(fdg_metrics['tpr_at_fpr_1pct']),
                'tpr@5%fpr': float(fdg_metrics['tpr_at_fpr_5pct']),
                'tpr@10%fpr': float(fdg_metrics['tpr_at_fpr_10pct']),
                'threshold': float(fdg_summary['threshold']),
                'selected_mask_ratio': '',
                'notes': 'Fast-DetectGPT surrogate baseline; GPT-2 is used as the AR scoring/reference model, not as the generator of the machine texts.',
            }
        )

    duo_plain_row = _best_row(duo_summary, 'duo_plain_meanlogprob')
    if duo_plain_row is not None:
        rows.append(
            {
                'experiment': 'C',
                'detector': 'duo_plain_meanlogprob',
                'generator_model': Path(config.get('checkpoint_dir', 'models/duo-distilled')).name,
                'scoring_model': Path(config.get('checkpoint_dir', 'models/duo-distilled')).name,
                'reference_model': Path(config.get('checkpoint_dir', 'models/duo-distilled')).name,
                'dataset_dir': str(dataset_dir),
                'data_export_dir': build_result.get('data_export_dir') or '',
                'val_roc_auc': float(duo_plain_row['val_roc_auc']),
                'roc_auc': float(duo_plain_row['roc_auc']),
                'pr_auc': float(duo_plain_row['pr_auc']),
                'accuracy': float(duo_plain_row['accuracy']),
                'balanced_accuracy': float(duo_plain_row['balanced_accuracy']),
                'tpr@1%fpr': float(duo_plain_row['tpr@1%fpr']),
                'tpr@5%fpr': float(duo_plain_row['tpr@5%fpr']),
                'tpr@10%fpr': float(duo_plain_row['tpr@10%fpr']),
                'threshold': float(duo_plain_row['threshold']),
                'selected_mask_ratio': float(duo_plain_row['mask_ratio']),
                'notes': 'Same-source DUO ablation without analytic normalization.',
            }
        )

    return rows


def _write_experiment_report(
    *,
    output_root: Path,
    dataset_tag: str,
    config_path: str,
    dataset_dir: Path,
    build_result: dict[str, Any],
    duo_result: dict[str, Any],
    fdg_result: dict[str, Any] | None,
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    report_dir = ensure_dir(output_root / 'reports' / dataset_tag)
    report_payload = {
        'config_path': str(config_path),
        'dataset_tag': dataset_tag,
        'dataset_dir': str(dataset_dir),
        'data_export_dir': build_result.get('data_export_dir'),
        'build_result': build_result,
        'duo_result': duo_result,
        'fastdetectgpt_result': fdg_result,
        'experiments': rows,
    }
    json_path = report_dir / 'experiment_comparison.json'
    md_path = report_dir / 'experiment_comparison.md'
    dump_json(report_payload, json_path)
    save_markdown_table(rows, md_path, 'Experiment comparison (A/B/C)')
    return {'report_dir': str(report_dir), 'json_path': str(json_path), 'markdown_path': str(md_path)}


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    dataset_tag = config.get('dataset_tag', 'default')
    output_root = Path(config.get('output_root', 'outputs'))
    dataset_dir = output_root / 'datasets' / dataset_tag

    build_args = argparse.Namespace(
        config=args.config,
        human_path=config.get('human_path'),
        text_field=None,
        checkpoint_dir=config.get('checkpoint_dir'),
        tokenizer_dir=config.get('tokenizer_dir'),
        allow_online_tokenizer=bool(config.get('allow_online_tokenizer', False)),
        output_dir=str(dataset_dir),
        dataset_tag=dataset_tag,
        data_export_root=config.get('data_export_root'),
        no_data_export=not bool(config.get('export_data_views', True)),
        seed=config.get('seed'),
        generation_steps=config.get('generation_steps'),
        generation_strategy=config.get('generation_strategy'),
        generation_batch_size=config.get('generation_batch_size'),
        temperature=config.get('generation_temperature'),
        top_p=config.get('generation_top_p'),
        prompt_fraction=config.get('prompt_fraction'),
        prompt_tokens=config.get('prompt_tokens'),
        target_total_tokens=config.get('target_total_tokens'),
        window_stride=config.get('window_stride'),
        max_windows_per_source=config.get('max_windows_per_source'),
        max_examples=config.get('max_examples'),
        unconditional_only=bool(config.get('dataset_unconditional_only', False)),
    )
    build_result = run_build_dataset(build_args)
    print(f'[pipeline] dataset built at {build_result["output_dir"]}')
    if build_result.get('data_export_dir'):
        print(f'[pipeline] readable data export written to {build_result["data_export_dir"]}')

    duo_args = argparse.Namespace(
        config=args.config,
        dataset_dir=str(dataset_dir),
        checkpoint_dir=config.get('checkpoint_dir'),
        tokenizer_dir=config.get('tokenizer_dir'),
        allow_online_tokenizer=bool(config.get('allow_online_tokenizer', False)),
        output_dir=None,
        mask_ratios=config.get('mask_ratios'),
        corruption_seeds=config.get('corruption_seeds'),
        bootstrap_samples=config.get('bootstrap_samples'),
        eps=1e-6,
    )
    duo_result = run_duo_analytic(duo_args)
    print(f'[pipeline] DUO analytic and plain baselines finished at {duo_result["output_dir"]}')

    fdg_result = None
    if not args.skip_fastdetectgpt:
        baseline_cfg = config.get('fastdetectgpt', {})
        fdg_args = argparse.Namespace(
            config=args.config,
            dataset_dir=str(dataset_dir),
            sampling_model_name_or_path=baseline_cfg.get('sampling_model_name_or_path', 'gpt2'),
            scoring_model_name_or_path=baseline_cfg.get('scoring_model_name_or_path', None),
            local_files_only=bool(baseline_cfg.get('local_files_only', False)),
            output_dir=None,
            max_length=int(baseline_cfg.get('max_length', 512)),
            device='auto',
            bootstrap_samples=config.get('bootstrap_samples'),
        )
        fdg_result = run_fastdetectgpt_baseline(fdg_args)
        print(f'[pipeline] Fast-DetectGPT baseline finished at {fdg_result["output_dir"]}')
    else:
        print('[pipeline] Fast-DetectGPT baseline skipped.')

    rows = _build_experiment_rows(
        config=config,
        dataset_dir=dataset_dir,
        build_result=build_result,
        duo_result=duo_result,
        fdg_result=fdg_result,
    )
    report_paths = _write_experiment_report(
        output_root=output_root,
        dataset_tag=dataset_tag,
        config_path=args.config,
        dataset_dir=dataset_dir,
        build_result=build_result,
        duo_result=duo_result,
        fdg_result=fdg_result,
        rows=rows,
    )
    print(f'[pipeline] experiment comparison written to {report_paths["markdown_path"]}')


if __name__ == '__main__':
    main()
