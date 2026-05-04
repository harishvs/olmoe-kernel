"""Plot the effect of prefix caching on TTFT vs prefix length.

Shows two curves:
  - Cache OFF: TTFT grows with total prompt length (P + 64)
  - Cache ON:  TTFT stays flat regardless of P — prefix is skipped
The gap between them widens with prefix length. That's the lever.
"""
import json
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SUMMARY_GLOB = str(REPO / "results/harness/prefix_caching/summary_*.json")
OUT = REPO / "talk" / "figures" / "prefix_caching.png"


def main():
    summary_path = sorted(glob(SUMMARY_GLOB))[-1]
    summary = json.load(open(summary_path))

    # Keys come back as strings from JSON
    prefixes = sorted(int(k) for k in summary.keys())
    off_ms = np.array([summary[str(P)]["off_ttft_ms"]["mean"] for P in prefixes])
    on_ms = np.array([summary[str(P)]["on_ttft_ms"]["mean"] for P in prefixes])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(prefixes, off_ms, "o-", color="tab:red", linewidth=2.2, markersize=10,
            label="Cache OFF — full prefill every request", zorder=3)
    ax.plot(prefixes, on_ms, "o-", color="tab:green", linewidth=2.2, markersize=10,
            label="Cache ON — prefix skipped on hit", zorder=3)

    # Shade the savings between the two curves
    ax.fill_between(prefixes, on_ms, off_ms, alpha=0.12, color="tab:orange",
                    label="TTFT saved by cache hit")

    # Per-point annotations
    for P, off, on in zip(prefixes, off_ms, on_ms):
        savings_pct = 100.0 * (off - on) / off
        ax.annotate(f"{off:.0f} ms", xy=(P, off), xytext=(8, 6),
                    textcoords="offset points", fontsize=10, color="tab:red")
        ax.annotate(f"{on:.0f} ms", xy=(P, on), xytext=(8, -14),
                    textcoords="offset points", fontsize=10, color="tab:green")
        # Put savings % at the widest gap
        mid_y = (off + on) / 2
        ax.annotate(f"−{savings_pct:.0f}%", xy=(P, mid_y), xytext=(-32, 0),
                    textcoords="offset points", fontsize=11, color="tab:orange",
                    fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks(prefixes)
    ax.set_xticklabels(prefixes)
    ax.set_xlabel("Shared prefix length (tokens, log scale)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Prefix caching: benefit scales with prefix length\n"
                 "Suffix=64 tokens, batch=1, bf16 OLMoE on L4")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)
    ax.set_ylim(0, max(off_ms) * 1.15)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
