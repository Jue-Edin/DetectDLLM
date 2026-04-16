import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from src.utils import load_json


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def regular_attention_multi_headed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    attention_output = F.scaled_dot_product_attention(
        query=q.transpose(1, 2),
        key=k.transpose(1, 2),
        value=v.transpose(1, 2),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )
    attention_output = attention_output.transpose(1, 2)
    batch_size, seq_len, n_heads, head_dim = attention_output.shape
    return attention_output.reshape(batch_size, seq_len, n_heads * head_dim)


@dataclass
class DUOConfig:
    vocab_size: int
    model_length: int
    causal: bool
    hidden_dim: int
    cond_dim: int
    n_blocks: int
    n_heads: int
    dropout: float
    var_min: bool

    @classmethod
    def from_json(cls, path: str | Path) -> "DUOConfig":
        data = load_json(path)
        return cls(
            vocab_size=int(data["vocab_size"]),
            model_length=int(data["model_length"]),
            causal=bool(data["causal"]),
            hidden_dim=int(data["hidden_dim"]),
            cond_dim=int(data["cond_dim"]),
            n_blocks=int(data["n_blocks"]),
            n_heads=int(data["n_heads"]),
            dropout=float(data["dropout"]),
            var_min=bool(data["var_min"]),
        )


class LayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 10_000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached: Optional[int] = None
        self.cos_cached: Optional[torch.Tensor] = None
        self.sin_cached: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        device = x.device
        dtype = x.dtype
        if self.seq_len_cached != seq_len or self.cos_cached is None or self.cos_cached.device != device:
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)
            self.cos_cached = emb.cos().to(dtype=dtype)
            self.sin_cached = emb.sin().to(dtype=dtype)
            self.seq_len_cached = seq_len
        return self.cos_cached, self.sin_cached


class EmbeddingLayer(nn.Module):
    def __init__(self, dim: int, vocab_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return self.embedding[x]
        return torch.einsum(
            "blv,ve->ble",
            torch.nn.functional.softmax(x, dim=-1).float(),
            self.embedding.float(),
        ).to(x.dtype)


class DDiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, cond_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim, bias=True),
        )
        self.dropout = dropout
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, rotary_cos_sin: tuple[torch.Tensor, torch.Tensor], c: torch.Tensor) -> torch.Tensor:
        x_skip = x
        x = self.norm1(x)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )
        x = modulate(x, shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        batch_size, seq_len, three_dim = qkv.shape
        head_dim = three_dim // (3 * self.n_heads)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, head_dim)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
        cos, sin = rotary_cos_sin
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        attn_out = regular_attention_multi_headed(q, k, v)
        x = x_skip + gate_msa * self.attn_out(attn_out)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.norm_final(x)
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(x, shift, scale)
        return self.linear(x)


class HFDIT(nn.Module):
    def __init__(self, config: DUOConfig) -> None:
        super().__init__()
        if config.causal:
            raise ValueError("This local runtime only supports the non-causal DUO checkpoint.")
        self.causal = config.causal
        self.vocab_size = config.vocab_size
        dim = config.hidden_dim
        self.vocab_embed = EmbeddingLayer(dim, self.vocab_size)
        self.sigma_map = TimestepEmbedder(config.cond_dim)
        self.rotary_emb = Rotary(dim // config.n_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim=dim,
                    n_heads=config.n_heads,
                    cond_dim=config.cond_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.n_blocks)
            ]
        )
        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=self.vocab_size,
            cond_dim=config.cond_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states: list[torch.Tensor] = []
        x = self.vocab_embed(x)
        if output_hidden_states:
            hidden_states.append(x)
        t_cond = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)
        for block in self.blocks:
            x = block(x, rotary_cos_sin, c=t_cond)
            if output_hidden_states:
                hidden_states.append(x)
        x = self.output_layer(x, c=t_cond)
        return x, hidden_states


class DUOLocal(nn.Module):
    def __init__(self, config: DUOConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = HFDIT(config)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "DUOLocal":
        checkpoint_dir = Path(checkpoint_dir)
        config = DUOConfig.from_json(checkpoint_dir / "config.json")
        model = cls(config)
        state_dict = load_file(str(checkpoint_dir / "model.safetensors"))
        model.load_state_dict(state_dict, strict=True)
        model.to(device=device, dtype=dtype)
        model.eval()
        return model

    def forward(
        self,
        input_ids: torch.LongTensor,
        timesteps: torch.FloatTensor,
        output_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.shape[0] != input_ids.shape[0]:
            timesteps = timesteps.expand(input_ids.shape[0])
        logits, hidden_states = self.backbone(
            x=input_ids,
            sigma=timesteps.float(),
            output_hidden_states=output_hidden_states,
        )
        if output_hidden_states:
            return logits, hidden_states
        return logits
