from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_utils import load_human_corpus


def test_load_original_json_dict() -> None:
    path = REPO_ROOT / 'data' / 'human' / 'squad_gemma-2b-instruct_rewrite.original.json'
    records = load_human_corpus(path)
    assert len(records) == 100
    assert all('text' in record for record in records)
    assert all(record['text'].strip() for record in records)
