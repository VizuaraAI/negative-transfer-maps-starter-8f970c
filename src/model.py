"""A small decoder-only transformer, deliberately plain.

Nothing here is novel, and that is the point: the contribution of this project
is the measurement, so the model has to be a boring, well-understood baseline
that a reviewer will not question. Pre-LayerNorm blocks, learned positional
embeddings, GELU MLP with 4x expansion, optional weight tying.

The one non-obvious piece is `param_groups`, which keeps LayerNorm and bias
parameters out of weight decay, and `n_nonembed_params`, which is the capacity
axis the whole paper is organised around.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError(
                f"n_embd={cfg.n_embd} is not divisible by n_head={cfg.n_head}"
            )
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Post-activation hidden units, used by the polysemanticity diagnostic."""
        return F.gelu(self.fc(x), approximate="tanh")


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scaled init on residual projections (GPT-2 recipe): keeps activation
        # variance stable as depth grows, which matters because depth varies
        # across the capacity grid.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction=reduction
        )
        return logits, loss

    def hidden_states(self, idx: torch.Tensor) -> list[torch.Tensor]:
        """Residual-stream activations after each block, for the CKA diagnostic."""
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        states = []
        for block in self.blocks:
            x = block(x)
            states.append(x)
        return states

    def n_params(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not non_embedding:
            return total
        total -= self.pos_emb.weight.numel()
        total -= self.tok_emb.weight.numel()
        if not self.cfg.tie_embeddings:
            total -= self.head.weight.numel()
        return total

    def param_groups(self, weight_decay: float) -> list[dict]:
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]


def build_model(cfg: ModelConfig) -> TinyLM:
    return TinyLM(cfg)


if __name__ == "__main__":
    from .config import CAPACITY_PRESETS

    print(f"{'preset':8s} {'layers':>7s} {'width':>7s} {'non-embed':>12s} {'total':>12s}")
    for name in CAPACITY_PRESETS:
        cfg = ModelConfig.from_capacity(name)
        model = build_model(cfg)
        print(
            f"{name:8s} {cfg.n_layer:7d} {cfg.n_embd:7d} "
            f"{model.n_params(non_embedding=True):12,d} {model.n_params():12,d}"
        )
