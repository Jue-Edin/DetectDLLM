import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, GPT2TokenizerFast


MASK_TOKEN = "<|mask|>"
GPT2_END_OF_TEXT = "<|endoftext|>"
DEFAULT_FALLBACK_TOKENIZER_DIR = Path("assets/tokenizer/gpt2")


def tokenizer_asset_paths(tokenizer_dir: str | Path) -> tuple[Path, Path, Path]:
    tokenizer_dir = Path(tokenizer_dir)
    return tokenizer_dir / "vocab.json", tokenizer_dir / "merges.txt", tokenizer_dir / "tokenizer.json"


def _has_local_tokenizer_assets(tokenizer_dir: str | Path) -> bool:
    vocab_path, merges_path, tokenizer_json_path = tokenizer_asset_paths(tokenizer_dir)
    return tokenizer_json_path.exists() or (vocab_path.exists() and merges_path.exists())


def _build_gpt2_fast_tokenizer(tokenizer_dir: Path, model_max_length: int) -> GPT2TokenizerFast:
    vocab_path, merges_path, tokenizer_json_path = tokenizer_asset_paths(tokenizer_dir)
    kwargs: dict[str, Any] = {
        "model_max_length": model_max_length,
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
    }
    if tokenizer_json_path.exists():
        kwargs["tokenizer_file"] = str(tokenizer_json_path)
    elif vocab_path.exists() and merges_path.exists():
        kwargs["vocab_file"] = str(vocab_path)
        kwargs["merges_file"] = str(merges_path)
    else:
        raise FileNotFoundError(
            f"No local tokenizer assets found in '{tokenizer_dir}'. Expected tokenizer.json "
            "or vocab.json + merges.txt."
        )
    tokenizer = GPT2TokenizerFast(**kwargs)
    tokenizer = _ensure_gpt2_special_tokens(tokenizer)
    return tokenizer


def _ensure_gpt2_special_tokens(tokenizer):
    if tokenizer.eos_token is None:
        tokenizer.eos_token = GPT2_END_OF_TEXT
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": MASK_TOKEN})
    return tokenizer


def _persist_tokenizer_metadata(tokenizer, tokenizer_dir: str | Path, source: dict[str, Any]) -> Path:
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = tokenizer_dir / "tokenizer_metadata.json"
    metadata = {
        "mask_token": tokenizer.mask_token,
        "mask_token_id": tokenizer.mask_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "base_vocab_size": tokenizer.vocab_size,
        "total_vocab_size": len(tokenizer),
        "source": source,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata_path


def load_local_tokenizer(
    checkpoint_dir: str | Path,
    tokenizer_dir: str | Path | None = DEFAULT_FALLBACK_TOKENIZER_DIR,
    model_max_length: int = 1024,
    allow_online: bool = False,
):
    checkpoint_dir = Path(checkpoint_dir)
    candidate_dirs: list[Path] = []
    for candidate in [checkpoint_dir, Path(tokenizer_dir) if tokenizer_dir is not None else None]:
        if candidate is None:
            continue
        candidate = Path(candidate)
        if candidate not in candidate_dirs:
            candidate_dirs.append(candidate)

    attempts: list[dict[str, str]] = []

    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists():
            attempts.append({"source": str(candidate_dir), "loader": "exists", "error": "directory not found"})
            continue

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(candidate_dir),
                local_files_only=True,
                trust_remote_code=False,
            )
            tokenizer = _ensure_gpt2_special_tokens(tokenizer)
            _persist_tokenizer_metadata(
                tokenizer,
                candidate_dir,
                source={"loader": "AutoTokenizer.from_pretrained", "path": str(candidate_dir), "mode": "offline"},
            )
            return tokenizer, {
                "loader": "AutoTokenizer.from_pretrained",
                "path": str(candidate_dir),
                "mode": "offline",
                "attempts": attempts,
            }
        except Exception as exc:
            attempts.append({"source": str(candidate_dir), "loader": "AutoTokenizer.from_pretrained", "error": str(exc)})

        if _has_local_tokenizer_assets(candidate_dir):
            tokenizer = _build_gpt2_fast_tokenizer(candidate_dir, model_max_length=model_max_length)
            _persist_tokenizer_metadata(
                tokenizer,
                candidate_dir,
                source={"loader": "GPT2TokenizerFast(local_files)", "path": str(candidate_dir), "mode": "offline"},
            )
            return tokenizer, {
                "loader": "GPT2TokenizerFast(local_files)",
                "path": str(candidate_dir),
                "mode": "offline",
                "attempts": attempts,
            }
        attempts.append(
            {
                "source": str(candidate_dir),
                "loader": "GPT2TokenizerFast(local_files)",
                "error": "missing tokenizer.json and vocab.json/merges.txt",
            }
        )

    if allow_online:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=False)
        tokenizer = _ensure_gpt2_special_tokens(tokenizer)
        persist_dir = Path(tokenizer_dir) if tokenizer_dir is not None else DEFAULT_FALLBACK_TOKENIZER_DIR
        _persist_tokenizer_metadata(
            tokenizer,
            persist_dir,
            source={"loader": "GPT2TokenizerFast.from_pretrained", "path": "gpt2", "mode": "online"},
        )
        return tokenizer, {
            "loader": "GPT2TokenizerFast.from_pretrained",
            "path": "gpt2",
            "mode": "online",
            "attempts": attempts,
        }

    attempts_text = "\n".join(
        f"- {item['loader']} @ {item['source']}: {item['error']}" for item in attempts
    )
    raise FileNotFoundError(
        "Unable to load a tokenizer offline.\n"
        f"Tried, in order:\n{attempts_text}\n"
        "Priority order is: checkpoint directory first, assets/tokenizer/gpt2 second, and online gpt2 only when "
        "--allow-online-tokenizer is explicitly enabled."
    )
