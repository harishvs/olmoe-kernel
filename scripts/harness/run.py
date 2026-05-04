"""Experiment runner: launches server, runs workload, captures JSON."""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from .client import BenchResult, ConcurrentResult, Request, run_concurrent_workload, run_workload
from .server import ServerConfig, serve


_TOKENIZER_CACHE: dict[str, object] = {}


def _get_tokenizer(model: str):
    if model not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        _TOKENIZER_CACHE[model] = AutoTokenizer.from_pretrained(model)
    return _TOKENIZER_CACHE[model]


def make_synthetic_prompt(n_tokens: int, *, model: str, seed: int = 42) -> str:
    """Build a prompt that tokenizes to exactly n_tokens under the model's tokenizer.

    Generates a random word stream, tokenizes it, slices to exactly n_tokens of
    token IDs, and decodes back to text. The returned string is guaranteed to
    re-tokenize to n_tokens against the same tokenizer.
    """
    import random
    rng = random.Random(seed)
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "pack", "my", "box", "with", "five", "dozen", "liquor", "jugs",
        "how", "vexingly", "zebras", "jump", "sphinx", "of", "black", "quartz",
        "judge", "crwth", "vyce", "ytterbium", "xenon",
    ]
    tok = _get_tokenizer(model)

    # Generate ~1.5x target word count to leave headroom for BPE expansion,
    # then slice the token stream to exactly n_tokens.
    raw = " ".join(rng.choice(words) for _ in range(int(n_tokens * 1.5) + 16))
    ids = tok(raw, add_special_tokens=False)["input_ids"]
    if len(ids) < n_tokens:
        # Rare: word pool too compressible. Fall back to repeating.
        raw = (raw + " ") * ((n_tokens // max(len(ids), 1)) + 2)
        ids = tok(raw, add_special_tokens=False)["input_ids"]
    ids = ids[:n_tokens]
    return tok.decode(ids, skip_special_tokens=True)


def run_experiment(
    *,
    engine: str,
    model: str,
    engine_args: list[str],
    requests: list[Request],
    warmup: int,
    seed: int,
    label: str,
    results_dir: Path,
    port: int = 8000,
) -> BenchResult:
    cfg = ServerConfig(engine=engine, model=model, port=port, extra_args=engine_args)
    with serve(cfg) as url:
        result = run_workload(
            url=url,
            model=model,
            requests=requests,
            warmup=warmup,
            seed=seed,
            label=label,
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"{label}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "label": result.label,
                "engine": engine,
                "model": model,
                "engine_args": engine_args,
                "n_iters": result.n_iters,
                "warmup": warmup,
                "seed": seed,
                "ttft_ms": result.ttft_ms,
                "tpot_ms": result.tpot_ms,
                "total_ms": result.total_ms,
                "raw": [dataclasses.asdict(r) for r in result.raw],
            },
            f,
            indent=2,
        )
    print(f"[harness.run] wrote {out_path}")
    return result
