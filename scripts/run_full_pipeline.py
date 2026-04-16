import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_dataset import run_build_dataset
from scripts.run_duo_analytic import run_duo_analytic
from scripts.run_fastdetectgpt_baseline import run_fastdetectgpt_baseline
from src.utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the full local DetectDLLM pipeline.')
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--skip-fastdetectgpt', action='store_true')
    return parser.parse_args()


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
        seed=config.get('seed'),
        generation_steps=config.get('generation_steps'),
        generation_strategy=config.get('generation_strategy'),
        temperature=config.get('generation_temperature'),
        top_p=config.get('generation_top_p'),
        prompt_fraction=config.get('prompt_fraction'),
        unconditional_only=bool(config.get('dataset_unconditional_only', False)),
    )
    build_result = run_build_dataset(build_args)
    print(f'[pipeline] dataset built at {build_result["output_dir"]}')

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
    print(f'[pipeline] analytic detector finished at {duo_result["output_dir"]}')

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


if __name__ == '__main__':
    main()
