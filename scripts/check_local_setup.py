import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_utils import load_human_corpus
from src.tokenizer_utils import load_local_tokenizer
from src.utils import load_json


REQUIRED_MODEL_FILES = [
    'config.json',
    'config.py',
    'merges.txt',
    'model.py',
    'model.safetensors',
    'tokenizer.json',
    'tokenizer_config.json',
    'vocab.json',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate the local DetectDLLM setup.')
    parser.add_argument('--config', default='configs/default.json')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    checkpoint_dir = Path(config.get('checkpoint_dir', 'models/duo-distilled'))
    tokenizer_dir = Path(config.get('tokenizer_dir', str(checkpoint_dir)))
    human_path = Path(config.get('human_path', 'data/human'))

    print(f'[check] checkpoint_dir = {checkpoint_dir}')
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f'Missing checkpoint directory: {checkpoint_dir}')
    missing = [name for name in REQUIRED_MODEL_FILES if not (checkpoint_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f'Missing required DUO files: {missing}')
    print('[check] DUO checkpoint files found.')

    print(f'[check] human_path = {human_path}')
    records = load_human_corpus(human_path)
    print(f'[check] loaded {len(records)} human passages.')

    tokenizer, report = load_local_tokenizer(
        checkpoint_dir=checkpoint_dir,
        tokenizer_dir=tokenizer_dir,
        model_max_length=1024,
        allow_online=False,
    )
    print(f'[check] tokenizer loader = {report["loader"]} @ {report["path"]}')
    print(f'[check] tokenizer vocab size = {len(tokenizer)}')
    print('[check] local setup looks ready.')


if __name__ == '__main__':
    main()
