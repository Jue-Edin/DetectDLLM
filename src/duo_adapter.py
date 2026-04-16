from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.corruption import CorruptionResult, build_linear_unmask_plan, mask_random_positions
from src.duo_local_model import DUOLocal
from src.tokenizer_utils import load_local_tokenizer
from src.utils import select_device, set_seed


class DuoAdapter:
    def __init__(
        self,
        checkpoint_dir: str | Path = "models/duo-distilled",
        tokenizer_dir: str | Path = "assets/tokenizer/gpt2",
        allow_online_tokenizer: bool = False,
        device: str = "auto",
        max_length: int | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.allow_online_tokenizer = allow_online_tokenizer
        self.device = select_device(device)
        self.model: DUOLocal | None = None
        self.tokenizer = None
        self.max_length = max_length
        self.tokenizer_load_report: dict[str, Any] | None = None
        self.vocab_audit: dict[str, Any] | None = None

    def load(self, require_tokenizer: bool = False) -> "DuoAdapter":
        if self.model is None:
            self.model = DUOLocal.from_checkpoint(
                checkpoint_dir=self.checkpoint_dir,
                device=self.device,
                dtype=torch.float32,
            )
            self.max_length = self.max_length or self.model.config.model_length
        if require_tokenizer and self.tokenizer is None:
            self.tokenizer, self.tokenizer_load_report = load_local_tokenizer(
                checkpoint_dir=self.checkpoint_dir,
                tokenizer_dir=self.tokenizer_dir,
                model_max_length=self.max_length or 1024,
                allow_online=self.allow_online_tokenizer,
            )
            self.vocab_audit = self._build_vocab_audit()
        return self

    def _require_model(self) -> DUOLocal:
        if self.model is None:
            self.load(require_tokenizer=False)
        assert self.model is not None
        return self.model

    def _require_tokenizer(self):
        if self.tokenizer is None:
            self.load(require_tokenizer=True)
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer failed to load.")
        return self.tokenizer

    def _build_vocab_audit(self) -> dict[str, Any]:
        model = self._require_model()
        tokenizer = self._require_tokenizer()
        audit = {
            "model_vocab_size": int(model.config.vocab_size),
            "tokenizer_base_vocab_size": int(tokenizer.vocab_size),
            "tokenizer_total_vocab_size": int(len(tokenizer)),
            "mask_token": tokenizer.mask_token,
            "mask_token_id": tokenizer.mask_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "aligned_total_vocab": int(len(tokenizer)) == int(model.config.vocab_size),
            "expected_mask_token_id": int(model.config.vocab_size) - 1,
            "mask_token_matches_expected_last_id": tokenizer.mask_token_id == (int(model.config.vocab_size) - 1),
        }
        audit["status"] = "aligned" if audit["aligned_total_vocab"] and audit["mask_token_matches_expected_last_id"] else "mismatch"
        return audit

    @property
    def mask_token_id(self) -> int:
        tokenizer = self._require_tokenizer()
        return int(tokenizer.mask_token_id)

    @property
    def special_token_ids(self) -> list[int]:
        tokenizer = self._require_tokenizer()
        ids = [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]
        return [int(token_id) for token_id in ids if token_id is not None]

    def tokenize_texts(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        tokenizer = self._require_tokenizer()
        return tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length or self.max_length,
            return_tensors=return_tensors,
        )

    def decode_ids(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str | list[str]:
        tokenizer = self._require_tokenizer()
        if isinstance(token_ids, torch.Tensor) and token_ids.ndim == 2:
            return tokenizer.batch_decode(token_ids.tolist(), skip_special_tokens=skip_special_tokens)
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            return tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def corrupt_ids(self, input_ids: torch.Tensor, mask_ratio: float, seed: int) -> CorruptionResult:
        tokenizer = self._require_tokenizer()
        return mask_random_positions(
            input_ids=input_ids,
            mask_ratio=mask_ratio,
            mask_token_id=int(tokenizer.mask_token_id),
            seed=seed,
            special_token_ids=self.special_token_ids,
        )

    def reconstruct_logits(self, input_ids: torch.Tensor, timesteps: torch.Tensor | float) -> torch.Tensor:
        model = self._require_model()
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor([timesteps], dtype=torch.float32, device=self.device)
        timesteps = timesteps.to(self.device).float()
        with torch.inference_mode():
            logits = model(input_ids=input_ids, timesteps=timesteps)
        return logits

    def score_masked_positions(
        self,
        original_ids: torch.Tensor,
        corrupted_ids: torch.Tensor,
        mask: torch.Tensor,
        timesteps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if original_ids.ndim == 1:
            original_ids = original_ids.unsqueeze(0)
        if corrupted_ids.ndim == 1:
            corrupted_ids = corrupted_ids.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if timesteps is None:
            timesteps = mask.sum(dim=1).float() / max(1, mask.shape[1])
        logits = self.reconstruct_logits(corrupted_ids, timesteps=timesteps)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        gathered = log_probs.gather(dim=-1, index=original_ids.to(self.device).unsqueeze(-1)).squeeze(-1)
        mask_device = mask.to(self.device)
        mask_counts = mask_device.sum(dim=1).clamp_min(1)
        sum_logprob = (gathered * mask_device).sum(dim=1)
        mean_logprob = sum_logprob / mask_counts
        entropy = -(probs * log_probs).sum(dim=-1)
        mean_entropy = (entropy * mask_device).sum(dim=1) / mask_counts
        mean_max_prob = (probs.max(dim=-1).values * mask_device).sum(dim=1) / mask_counts
        return {
            "sum_logprob": sum_logprob.detach().cpu(),
            "mean_logprob": mean_logprob.detach().cpu(),
            "mean_entropy": mean_entropy.detach().cpu(),
            "mean_max_prob": mean_max_prob.detach().cpu(),
            "mask_count": mask_counts.detach().cpu(),
        }

    def _sample_rows(
        self,
        logits: torch.Tensor,
        strategy: str,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> torch.Tensor:
        if strategy == "greedy" or temperature <= 0:
            return logits.argmax(dim=-1)
        rng = np.random.default_rng(seed)
        rows = []
        for row in logits.detach().float().cpu().numpy():
            scaled = row / max(temperature, 1e-6)
            shifted = scaled - np.max(scaled)
            probs = np.exp(shifted)
            probs = probs / probs.sum()
            if strategy == "top_p":
                order = np.argsort(-probs)
                sorted_probs = probs[order]
                cumulative = np.cumsum(sorted_probs)
                cutoff = cumulative <= top_p
                if not np.any(cutoff):
                    cutoff[0] = True
                else:
                    cutoff[np.argmax(cutoff)] = True
                kept_indices = order[cutoff]
                kept_probs = probs[kept_indices]
                kept_probs = kept_probs / kept_probs.sum()
                sampled = rng.choice(kept_indices, p=kept_probs)
            else:
                sampled = rng.choice(len(probs), p=probs)
            rows.append(int(sampled))
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def _iterative_fill(
        self,
        current_ids: torch.Tensor,
        current_mask: torch.Tensor,
        lengths: torch.Tensor,
        num_steps: int,
        strategy: str,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> torch.Tensor:
        plan = build_linear_unmask_plan(current_mask.cpu(), num_steps=num_steps, seed=seed)
        current_ids = current_ids.clone().to(self.device)
        current_mask = current_mask.clone().to(self.device)
        lengths = lengths.to(self.device)
        for step_idx in range(num_steps):
            remaining = current_mask.sum(dim=1)
            if int(remaining.sum().item()) == 0:
                break
            timesteps = remaining.float() / lengths.float().clamp_min(1)
            logits = self.reconstruct_logits(current_ids, timesteps=timesteps)
            for row_idx in range(current_ids.shape[0]):
                positions = plan[row_idx][step_idx]
                if not positions:
                    continue
                position_tensor = torch.tensor(positions, dtype=torch.long, device=self.device)
                row_logits = logits[row_idx, position_tensor]
                sampled = self._sample_rows(
                    row_logits,
                    strategy=strategy,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed + (row_idx * 1009) + step_idx,
                )
                current_ids[row_idx, position_tensor] = sampled
                current_mask[row_idx, position_tensor] = False
        return current_ids

    def sample_reconstructions(
        self,
        corrupted_ids: torch.Tensor,
        mask: torch.Tensor,
        num_samples: int,
        num_steps: int = 1,
        strategy: str = "top_p",
        temperature: float = 1.0,
        top_p: float = 0.9,
        seed: int = 0,
    ) -> torch.Tensor:
        if corrupted_ids.ndim == 1:
            corrupted_ids = corrupted_ids.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        lengths = torch.full((corrupted_ids.shape[0],), corrupted_ids.shape[1], dtype=torch.long)
        samples = []
        for sample_idx in range(num_samples):
            sampled_ids = self._iterative_fill(
                current_ids=corrupted_ids,
                current_mask=mask,
                lengths=lengths,
                num_steps=num_steps,
                strategy=strategy,
                temperature=temperature,
                top_p=top_p,
                seed=seed + sample_idx,
            )
            samples.append(sampled_ids.detach().cpu())
        return torch.stack(samples, dim=0)

    def generate_texts(
        self,
        target_lengths: list[int],
        prompt_texts: list[str] | None = None,
        num_steps: int = 8,
        strategy: str = "top_p",
        temperature: float = 1.0,
        top_p: float = 0.9,
        seed: int = 0,
    ) -> dict[str, Any]:
        tokenizer = self._require_tokenizer()
        set_seed(seed)
        batch_size = len(target_lengths)
        max_length = max(target_lengths)
        pad_token_id = int(tokenizer.pad_token_id)
        mask_token_id = int(tokenizer.mask_token_id)
        current_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
        current_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        prompt_lengths = []
        prompt_token_ids: list[list[int]] = []
        if prompt_texts is None:
            prompt_texts = [""] * batch_size
        for text, target_length in zip(prompt_texts, target_lengths):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            prompt_token_ids.append(token_ids[:target_length])
            prompt_lengths.append(min(len(token_ids), target_length))
        for row_idx, target_length in enumerate(target_lengths):
            prompt_ids = prompt_token_ids[row_idx]
            prompt_length = prompt_lengths[row_idx]
            if prompt_length:
                current_ids[row_idx, :prompt_length] = torch.tensor(prompt_ids, dtype=torch.long)
            if target_length > prompt_length:
                current_ids[row_idx, prompt_length:target_length] = mask_token_id
                current_mask[row_idx, prompt_length:target_length] = True
        generated = self._iterative_fill(
            current_ids=current_ids,
            current_mask=current_mask,
            lengths=torch.tensor(target_lengths, dtype=torch.long),
            num_steps=num_steps,
            strategy=strategy,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        ).cpu()
        texts = []
        sequences = []
        for row_idx, target_length in enumerate(target_lengths):
            sequence = generated[row_idx, :target_length].tolist()
            texts.append(tokenizer.decode(sequence, skip_special_tokens=True))
            sequences.append(sequence)
        return {
            "texts": texts,
            "token_ids": sequences,
            "target_lengths": target_lengths,
            "prompt_lengths": prompt_lengths,
            "num_steps": num_steps,
            "sampling": strategy,
        }

    def export_corruption(self, result: CorruptionResult) -> dict[str, Any]:
        return asdict(result)
