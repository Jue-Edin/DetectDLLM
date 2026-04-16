from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


DEFAULT_MASK_RATIOS = [0.15, 0.30, 0.50]


@dataclass
class CorruptionResult:
    corrupted_ids: torch.Tensor
    mask: torch.Tensor
    mask_counts: torch.Tensor
    mask_ratio: float


def _valid_positions(input_ids: torch.Tensor, special_token_ids: Iterable[int] | None = None) -> torch.Tensor:
    valid = torch.ones_like(input_ids, dtype=torch.bool)
    if special_token_ids is None:
        return valid
    for token_id in special_token_ids:
        if token_id is not None and token_id >= 0:
            valid &= input_ids.ne(token_id)
    return valid


def mask_random_positions(
    input_ids: torch.Tensor,
    mask_ratio: float,
    mask_token_id: int,
    seed: int,
    special_token_ids: Iterable[int] | None = None,
) -> CorruptionResult:
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    generator = np.random.default_rng(seed)
    valid = _valid_positions(input_ids, special_token_ids=special_token_ids)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row_idx in range(input_ids.shape[0]):
        indices = torch.where(valid[row_idx])[0].cpu().numpy()
        if len(indices) == 0:
            continue
        mask_count = max(1, int(round(len(indices) * mask_ratio)))
        chosen = generator.choice(indices, size=min(mask_count, len(indices)), replace=False)
        mask[row_idx, torch.tensor(chosen, dtype=torch.long)] = True
    corrupted_ids = input_ids.clone()
    corrupted_ids[mask] = mask_token_id
    return CorruptionResult(
        corrupted_ids=corrupted_ids,
        mask=mask,
        mask_counts=mask.sum(dim=1),
        mask_ratio=mask_ratio,
    )


def mask_ratio_tensor(mask: torch.Tensor) -> torch.Tensor:
    seq_len = mask.shape[1]
    return mask.sum(dim=1).float() / max(1, seq_len)


def build_linear_unmask_plan(mask: torch.Tensor, num_steps: int, seed: int) -> list[list[list[int]]]:
    rng = np.random.default_rng(seed)
    plans: list[list[list[int]]] = []
    for row in mask.cpu().numpy():
        positions = np.flatnonzero(row).tolist()
        rng.shuffle(positions)
        total = len(positions)
        step_plan: list[list[int]] = []
        start = 0
        for step_idx in range(num_steps):
            target_revealed = int(round(total * (step_idx + 1) / max(1, num_steps)))
            end = max(start, min(target_revealed, total))
            if step_idx == num_steps - 1:
                end = total
            step_positions = positions[start:end]
            if not step_positions and start < total:
                step_positions = [positions[start]]
                end = start + 1
            step_plan.append(step_positions)
            start = end
        plans.append(step_plan)
    return plans


def length_bucket(length: int, boundaries: tuple[int, int]) -> str:
    low, high = boundaries
    if length <= low:
        return "short"
    if length <= high:
        return "medium"
    return "long"


def compute_length_bucket_boundaries(lengths: list[int]) -> tuple[int, int]:
    if not lengths:
        return (0, 0)
    values = sorted(lengths)
    low = values[len(values) // 3]
    high = values[(2 * len(values)) // 3]
    return low, high
