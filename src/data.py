"""Domain corpora: registry, streaming loaders, and token-budget bookkeeping.

Eight public domains are registered below. All of them stream from the Hugging
Face Hub, so nothing large is committed to this repository. The tokenizer is
pinned across every run in the grid: if the tokenizer changes, cross-domain loss
numbers stop being comparable and the whole transfer map becomes meaningless.

Run `python -m src.data --check` to confirm every domain still resolves on the
Hub before you launch a long sweep.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

DOMAIN_ORDER = [
    "web",
    "wiki",
    "code",
    "math",
    "legal",
    "biomed",
    "dialog",
    "multiling",
]


@dataclass(frozen=True)
class DomainSpec:
    key: str
    hf_path: str
    hf_name: str | None
    split: str
    text_field: str
    blurb: str


# Each entry is a real public dataset on the Hugging Face Hub. If one of these
# is ever gated or renamed, swap it here: nothing else in the codebase hardcodes
# a dataset path.
DOMAINS: dict[str, DomainSpec] = {
    "web": DomainSpec(
        "web", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text",
        "filtered educational web text, the generic-English control domain",
    ),
    "wiki": DomainSpec(
        "wiki", "wikimedia/wikipedia", "20231101.en", "train", "text",
        "encyclopedic prose, expected to be near-free with web text",
    ),
    "code": DomainSpec(
        "code", "codeparrot/codeparrot-clean", None, "train", "content",
        "Python source, the most syntactically distant domain in the pool",
    ),
    "math": DomainSpec(
        "math", "open-web-math/open-web-math", None, "train", "text",
        "LaTeX-heavy mathematical web text",
    ),
    "legal": DomainSpec(
        "legal", "HFforLegal/case-law", None, "us", "document",
        "US case law, long-range and highly formulaic",
    ),
    "biomed": DomainSpec(
        "biomed", "MedRAG/pubmed", None, "train", "content",
        "PubMed abstracts, dense biomedical terminology",
    ),
    "dialog": DomainSpec(
        "dialog", "HuggingFaceH4/ultrachat_200k", None, "train_sft", "messages",
        "multi-turn conversational text, flattened into a transcript",
    ),
    "multiling": DomainSpec(
        "multiling", "HuggingFaceFW/fineweb-2", "hin_Deva", "train", "text",
        "non-English (Hindi) web text, shares no script with the others",
    ),
}


def get_tokenizer(name: str = "EleutherAI/gpt-neox-20b"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _extract_text(record: dict, field: str) -> str:
    value = record.get(field, "")
    if isinstance(value, str):
        return value
    # ultrachat stores a list of {role, content} dicts; flatten to a transcript.
    if isinstance(value, list):
        parts = []
        for turn in value:
            if isinstance(turn, dict):
                role = turn.get("role", "")
                content = turn.get("content", "")
                parts.append(f"{role}: {content}" if role else str(content))
            else:
                parts.append(str(turn))
        return "\n".join(parts)
    return str(value)


def iter_domain_text(
    key: str, seed: int = 0, shuffle_buffer: int = 10_000, streaming: bool = True,
    cache_dir: str | None = None, split: str | None = None,
) -> Iterator[str]:
    """Yield raw documents from one domain, indefinitely (loops at exhaustion)."""
    from datasets import load_dataset

    spec = DOMAINS[key]
    while True:
        ds = load_dataset(
            spec.hf_path,
            spec.hf_name,
            split=split or spec.split,
            streaming=streaming,
            cache_dir=cache_dir,
        )
        if streaming:
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
        for record in ds:
            text = _extract_text(record, spec.text_field)
            if text and text.strip():
                yield text


def iter_domain_tokens(
    key: str, tokenizer, block_size: int, seed: int = 0, streaming: bool = True,
    cache_dir: str | None = None, split: str | None = None,
) -> Iterator[np.ndarray]:
    """Yield fixed-length token blocks of `block_size + 1` ids from one domain.

    Documents are concatenated with an EOS separator and chopped into blocks, the
    standard pretraining packing scheme. The extra token is the shifted label.
    """
    eos = tokenizer.eos_token_id
    buffer: list[int] = []
    need = block_size + 1
    for text in iter_domain_text(
        key, seed=seed, streaming=streaming, cache_dir=cache_dir, split=split
    ):
        buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        buffer.append(eos)
        while len(buffer) >= need:
            yield np.asarray(buffer[:need], dtype=np.int64)
            buffer = buffer[block_size:]


class MixtureStream:
    """Interleaves per-domain token streams according to fixed mixture weights.

    Sampling is stochastic per block but the realised per-domain token counts are
    tracked exactly in `tokens_seen`, which is what gets logged. That log is the
    ground truth for "matched tokens on domain i", the control that makes the
    pairwise delta interpretable.
    """

    def __init__(
        self, mixture: dict[str, float], tokenizer, block_size: int, seed: int = 0,
        streaming: bool = True, cache_dir: str | None = None,
    ):
        unknown = set(mixture) - set(DOMAINS)
        if unknown:
            raise ValueError(f"unknown domains in mixture: {sorted(unknown)}")
        total = float(sum(mixture.values()))
        if total <= 0:
            raise ValueError("mixture weights must sum to a positive number")

        self.keys = sorted(mixture)
        self.weights = np.array([mixture[k] / total for k in self.keys])
        self.block_size = block_size
        self.tokens_seen: dict[str, int] = {k: 0 for k in self.keys}
        self.blocks_seen: dict[str, int] = {k: 0 for k in self.keys}
        self.rng = np.random.default_rng(seed)
        self.streams = {
            k: iter_domain_tokens(
                k, tokenizer, block_size, seed=seed + i,
                streaming=streaming, cache_dir=cache_dir,
            )
            for i, k in enumerate(self.keys)
        }

    def next_block(self) -> tuple[np.ndarray, str]:
        key = self.keys[int(self.rng.choice(len(self.keys), p=self.weights))]
        block = next(self.streams[key])
        self.tokens_seen[key] += self.block_size
        self.blocks_seen[key] += 1
        return block, key

    def next_batch(self, batch_size: int, device: str = "cpu"):
        blocks, keys = [], []
        for _ in range(batch_size):
            block, key = self.next_block()
            blocks.append(block)
            keys.append(key)
        arr = torch.from_numpy(np.stack(blocks))
        x = arr[:, :-1].contiguous().to(device)
        y = arr[:, 1:].contiguous().to(device)
        return x, y, keys

    def min_tokens_seen(self) -> int:
        return min(self.tokens_seen.values())


def build_eval_batches(
    key: str, tokenizer, block_size: int, n_blocks: int, batch_size: int = 8,
    seed: int = 1234, streaming: bool = True, cache_dir: str | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Materialise a FIXED held-out evaluation set for one domain.

    The seed here is deliberately different from any training seed and is held
    constant across the entire grid: every run is scored on byte-identical eval
    batches, so a delta of 0.01 nats is a real delta and not eval-set noise.
    """
    stream = iter_domain_tokens(
        key, tokenizer, block_size, seed=seed, streaming=streaming,
        cache_dir=cache_dir,
    )
    blocks = [b for b in itertools.islice(stream, n_blocks)]
    batches = []
    for i in range(0, len(blocks) - batch_size + 1, batch_size):
        arr = torch.from_numpy(np.stack(blocks[i : i + batch_size]))
        batches.append((arr[:, :-1].contiguous(), arr[:, 1:].contiguous()))
    return batches


def _check_domains(streaming: bool = True) -> int:
    """Smoke-test every registered domain. Returns the number of failures."""
    failures = 0
    for key in DOMAIN_ORDER:
        spec = DOMAINS[key]
        try:
            text = next(iter_domain_text(key, streaming=streaming))
            preview = " ".join(text.split())[:70]
            print(f"  ok    {key:10s} {spec.hf_path:35s} | {preview}")
        except Exception as exc:  # noqa: BLE001 - we want the reason printed
            failures += 1
            print(f"  FAIL  {key:10s} {spec.hf_path:35s} | {type(exc).__name__}: {exc}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the domain registry.")
    parser.add_argument("--check", action="store_true", help="pull one doc per domain")
    parser.add_argument("--no-streaming", action="store_true")
    args = parser.parse_args()

    print(f"{len(DOMAINS)} domains registered:")
    for key in DOMAIN_ORDER:
        print(f"  {key:10s} {DOMAINS[key].blurb}")
    if args.check:
        print("\nchecking Hub availability (this downloads a few KB per domain):")
        failures = _check_domains(streaming=not args.no_streaming)
        print(f"\n{len(DOMAINS) - failures}/{len(DOMAINS)} domains reachable")


if __name__ == "__main__":
    main()
