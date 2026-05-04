"""Dataset loaders for benchmark prompts."""
from __future__ import annotations

import json
import random
from pathlib import Path

from .client import Request

SHAREGPT_PATH = Path(__file__).resolve().parents[2] / "data" / "sharegpt.json"


def load_sharegpt_prompts(
    *,
    n: int,
    min_tokens: int,
    max_tokens: int,
    tokenizer,
    seed: int = 42,
    max_output_tokens: int = 128,
) -> list[Request]:
    """Return n Request objects drawn from ShareGPT, filtered to a token-length band.

    The first human message of each conversation is the prompt.
    """
    if not SHAREGPT_PATH.exists():
        raise FileNotFoundError(
            f"ShareGPT dataset not found at {SHAREGPT_PATH}. "
            "Download it first."
        )

    with open(SHAREGPT_PATH) as f:
        conversations = json.load(f)

    rng = random.Random(seed)
    rng.shuffle(conversations)

    selected: list[Request] = []
    for i, conv in enumerate(conversations):
        if not conv.get("conversations"):
            continue
        turn = conv["conversations"][0]
        if turn.get("from") != "human":
            continue
        text = turn.get("value", "")
        if not text:
            continue
        n_tok = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if min_tokens <= n_tok <= max_tokens:
            selected.append(Request(
                prompt=text,
                max_tokens=max_output_tokens,
                label=f"sharegpt_{conv['id']}_tok{n_tok}",
            ))
        if len(selected) >= n:
            break

    if len(selected) < n:
        raise RuntimeError(
            f"Only found {len(selected)}/{n} ShareGPT prompts in "
            f"[{min_tokens}, {max_tokens}] tokens. Loosen the filter or "
            "increase the dataset scan."
        )

    return selected
