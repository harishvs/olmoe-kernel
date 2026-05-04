"""Plot baseline TPOT vs input length.

Shows that decode cost per token is essentially independent of prompt length
- strong evidence that decode has a different (bandwidth-bound) bottleneck
than prefill (which scales with length).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
OUT = Path(__file__).resolve().parents[2] / "talk" / "figures" / "tpot_vs_length.png"

INPUT_LENS = [32, 128, 512, 1024]
TIMESTAMP = "20260503_051642"


def load_tpot(input_len: int) -> float:
    """Return TPOT in ms for the given input length, derived by subtracting
    the O=1 (prefill-only) run from the O=128 run and dividing by 127."""
    f1 = RESULTS_DIR / f"baseline_{TIMESTAMP}_L{input_len}_O1_seed42.json"
    f2 = RESULTS_DIR / f"baseline_{TIMESTAMP}_L{input_len}_O128_seed42.json"
    t1 = json.load(open(f1))["avg_latency"] * 1000.0
    t2 = json.load(open(f2))["avg_latency"] * 1000.0
    return (t2 - t1) / 127.0


def main():
    tpot = np.array([load_tpot(L) for L in INPUT_LENS])
    lens = np.array(INPUT_LENS, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(lens, tpot, "o-", color="tab:blue", linewidth=2.2, markersize=10,
            label="Measured TPOT (OLMoE on L4)", zorder=3)

    # Reference line: what would "grows with prompt length" look like?
    # Anchor at L=32, extrapolate linearly in input length (i.e. if decode
    # cost scaled the way prefill does).
    anchor_T = tpot[0]
    linear_ref = anchor_T * (lens / lens[0])
    ax.plot(lens, linear_ref, "--", color="tab:red", alpha=0.8,
            label="Hypothetical: if decode scaled like prefill")

    for L, T in zip(lens, tpot):
        ax.annotate(f"{T:.1f} ms", xy=(L, T), xytext=(6, 12),
                    textcoords="offset points", fontsize=10, color="tab:blue")

    ax.set_xscale("log")
    ax.set_xticks(INPUT_LENS)
    ax.set_xticklabels(INPUT_LENS)
    ax.set_xlabel("Input length at start of decode (tokens, log scale)")
    ax.set_ylabel("Time per output token (ms)")
    ax.set_title("TPOT is nearly flat — decode doesn't care about prompt length")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.set_ylim(0, max(linear_ref.max(), tpot.max()) * 1.1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print(f"  32 -> 1024 growth: {tpot[-1] / tpot[0]:.2f}x (measured) "
          f"vs {1024 / 32:.0f}x (if decode scaled with prompt)")


if __name__ == "__main__":
    main()
