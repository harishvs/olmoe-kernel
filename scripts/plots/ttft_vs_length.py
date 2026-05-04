"""Plot baseline TTFT vs input length, with linear and quadratic reference lines.

Shows that OLMoE TTFT grows roughly linearly with input length on L4 -
NOT quadratically, which attention-dominated intuition would predict.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
OUT = Path(__file__).resolve().parents[2] / "talk" / "figures" / "ttft_vs_length.png"

INPUT_LENS = [32, 128, 512, 1024]
TIMESTAMP = "20260503_051642"


def load_ttft(input_len: int) -> float:
    """Return mean TTFT in ms for the given input length from the O=1 sweep."""
    f = RESULTS_DIR / f"baseline_{TIMESTAMP}_L{input_len}_O1_seed42.json"
    return json.load(open(f))["avg_latency"] * 1000.0


def main():
    ttft = np.array([load_ttft(L) for L in INPUT_LENS])
    lens = np.array(INPUT_LENS, dtype=float)

    # Anchor both reference curves at L=32 so they all start at the same point.
    anchor_L = lens[0]
    anchor_T = ttft[0]

    linear_ref = anchor_T * (lens / anchor_L)
    quadratic_ref = anchor_T * (lens / anchor_L) ** 2

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax in (ax_full, ax_zoom):
        ax.plot(lens, ttft, "o-", color="tab:blue", linewidth=2.2, markersize=9,
                label="Measured TTFT (OLMoE on L4)", zorder=3)
        ax.plot(lens, linear_ref, "--", color="tab:green", alpha=0.9,
                label="Linear reference (anchored at L=32)")
        ax.plot(lens, quadratic_ref, ":", color="tab:red", alpha=0.9,
                label="Quadratic reference (if attention dominated)")
        ax.set_xscale("log")
        ax.set_xticks(INPUT_LENS)
        ax.set_xticklabels(INPUT_LENS)
        ax.set_xlabel("Input length (tokens, log scale)")
        ax.set_ylabel("TTFT (ms)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=9)

    # Left panel: full y-range, shows quadratic blowup
    ax_full.set_title("Full range — quadratic blows up")
    ax_full.set_ylim(bottom=0)

    # Right panel: zoomed to the measured data, shows sub-linear growth
    ax_zoom.set_title("Zoomed — measured is even below linear")
    ax_zoom.set_ylim(0, 200)
    for L, T in zip(lens, ttft):
        ax_zoom.annotate(f"{T:.1f} ms", xy=(L, T), xytext=(6, 10),
                         textcoords="offset points", fontsize=9, color="tab:blue")

    fig.suptitle("TTFT grows sub-linearly with input length on OLMoE",
                 fontsize=13, y=1.02)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print(f"  32 -> 1024 growth: {ttft[-1] / ttft[0]:.2f}x (measured) "
          f"vs {1024 / 32:.0f}x (linear) vs {(1024 / 32)**2:.0f}x (quadratic)")


if __name__ == "__main__":
    main()
