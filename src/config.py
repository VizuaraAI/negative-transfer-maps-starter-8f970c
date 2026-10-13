"""Configuration objects for the negative transfer mapping experiments.

Every run in this project is fully described by three things: a capacity preset
(how big the model is), a mixture (which domains it sees and in what ratio), and
a token budget. Keeping those three in one serializable object is what makes the
180-run grid reproducible and what lets `transfer.py` line runs up afterwards.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Capacity presets. The label is nominal, `nonembed_params()` reports the real
# count. Non-embedding parameters (not total) are the capacity axis of this
# project, because embedding parameters do not participate in the shared
# computation where domains compete.
CAPACITY_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"n_layer": 6, "n_embd": 240, "n_head": 6},
    "small": {"n_layer": 8, "n_embd": 384, "n_head": 6},
    "base": {"n_layer": 12, "n_embd": 560, "n_head": 10},
    "medium": {"n_layer": 16, "n_embd": 912, "n_head": 12},
    "large": {"n_layer": 24, "n_embd": 1200, "n_head": 16},
}


@dataclass
class ModelConfig:
    capacity: str = "tiny"
    n_layer: int = 6
    n_embd: int = 240
    n_head: int = 6
    block_size: int = 512
    vocab_size: int = 50304  # gpt-neox-20b tokenizer, padded to a multiple of 64
    dropout: float = 0.0
    tie_embeddings: bool = True

    @classmethod
    def from_capacity(cls, capacity: str, **overrides) -> "ModelConfig":
        if capacity not in CAPACITY_PRESETS:
            raise ValueError(
                f"unknown capacity {capacity!r}; choose from {sorted(CAPACITY_PRESETS)}"
            )
        return cls(capacity=capacity, **CAPACITY_PRESETS[capacity], **overrides)

    def nonembed_params(self) -> int:
        """Analytic non-embedding parameter count (attention + MLP + norms)."""
        d, layers = self.n_embd, self.n_layer
        attn = 4 * d * d + 4 * d  # qkv + out projection, with biases
        mlp = 8 * d * d + 5 * d  # 4d expansion, two matrices, with biases
        norms = 4 * d  # two LayerNorms per block
        return layers * (attn + mlp + norms) + 2 * d  # + final LayerNorm


@dataclass
class OptimConfig:
    # Base LR is specified at the `base` width and rescaled as 1/width (muP-style
    # maximal update scaling) so that no capacity in the grid is accidentally
    # mistuned. A mistuned small model looks exactly like capacity competition,
    # which is the single most dangerous confound in this project.
    base_lr: float = 6e-4
    base_width: int = 560
    lr_scaling: str = "mup"  # "mup" | "constant"
    min_lr_ratio: float = 0.1
    warmup_frac: float = 0.02
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    def lr_for_width(self, width: int) -> float:
        if self.lr_scaling == "constant":
            return self.base_lr
        return self.base_lr * (self.base_width / width)


@dataclass
class DataConfig:
    tokenizer: str = "EleutherAI/gpt-neox-20b"
    # Mixture weights over domain keys registered in `data.py`. A singleton run
    # has one key at weight 1.0; a pair run has two keys at 0.5 each.
    mixture: dict[str, float] = field(default_factory=lambda: {"wiki": 1.0})
    # Token budget accounting. "per_domain" holds tokens-seen-per-domain fixed
    # between singleton and pair runs (so a 2-domain run trains for 2x total
    # steps); "total" holds total tokens fixed. The paper reports both.
    budget_mode: str = "per_domain"
    tokens_per_domain: int = 50_000_000
    eval_tokens_per_domain: int = 2_000_000
    seed: int = 0
    cache_dir: str = "data_cache"
    streaming: bool = True

    def total_tokens(self) -> int:
        if self.budget_mode == "total":
            return self.tokens_per_domain
        return self.tokens_per_domain * len(self.mixture)


@dataclass
class TrainConfig:
    batch_tokens: int = 65_536  # tokens per optimizer step
    micro_batch_size: int = 8
    log_every: int = 50
    eval_every: int = 500
    diagnostics_every: int = 0  # 0 disables; set to eval_every to collect them
    device: str = "auto"
    dtype: str = "bfloat16"
    compile_model: bool = False
    out_dir: str = "runs"
    run_name: str | None = None
    save_checkpoint: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def run_id(self) -> str:
        """Stable identifier used as the run directory name and join key."""
        if self.train.run_name:
            return self.train.run_name
        mix = "+".join(sorted(self.data.mixture))
        return f"{self.model.capacity}__{mix}__s{self.data.seed}"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def load_config(path: str | Path, **overrides) -> ExperimentConfig:
    """Load a YAML config, then apply dotted overrides like model.capacity='base'."""
    raw = yaml.safe_load(Path(path).read_text()) or {}

    capacity = raw.get("model", {}).get("capacity", "tiny")
    model_kwargs = {k: v for k, v in raw.get("model", {}).items() if k != "capacity"}
    cfg = ExperimentConfig(
        model=ModelConfig.from_capacity(capacity, **model_kwargs),
        optim=OptimConfig(**raw.get("optim", {})),
        data=DataConfig(**raw.get("data", {})),
        train=TrainConfig(**raw.get("train", {})),
    )

    for dotted, value in overrides.items():
        if value is None:
            continue
        section, _, attr = dotted.partition(".")
        if not attr or not hasattr(cfg, section):
            raise ValueError(f"bad override {dotted!r}")
        target = getattr(cfg, section)
        if not hasattr(target, attr):
            raise ValueError(f"bad override {dotted!r}")
        setattr(target, attr, value)

    # Re-derive the architecture if capacity was overridden on the command line.
    if "model.capacity" in overrides and overrides["model.capacity"]:
        cfg.model = ModelConfig.from_capacity(
            cfg.model.capacity,
            block_size=cfg.model.block_size,
            vocab_size=cfg.model.vocab_size,
            dropout=cfg.model.dropout,
            tie_embeddings=cfg.model.tie_embeddings,
        )
    return cfg
