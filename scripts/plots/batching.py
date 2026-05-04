"""Plot the batching Pareto: throughput up, latency up — pick your side.

Two-panel figure:
  Left: output tokens/sec vs concurrency (throughput curve)
  Right: per-request TTFT and TPOT vs concurrency (latency curves)
"""
import json
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SUMMARY_GLOB = str(REPO / "results/harness/batching/summary_*.json")
OUT = REPO / "talk" / "figures" / "batching.png"


def main():
    summary_path = sorted(glob(SUMMARY_GLOB))[-1]
    summary = json.load(open(summary_path))

    Bs = sorted(int(k) for k in summary.keys())
    tok_s = np.array([summary[str(B)]["output_tokens_per_s"] for B in Bs])
    req_s = np.array([summary[str(B)]["requests_per_s"] for B in Bs])
    ttft_mean = np.array([summary[str(B)]["ttft_ms"]["mean"] for B in Bs])
    ttft_p90 = np.array([summary[str(B)]["ttft_ms"]["p90"] for B in Bs])
    tpot_mean = np.array([summary[str(B)]["tpot_ms"]["mean"] for B in Bs])

    fig, (ax_tp, ax_lat) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left panel: throughput ---
    ax_tp.plot(Bs, tok_s, "o-", color="tab:blue", linewidth=2.2, markersize=10,
               label="Output tokens/sec", zorder=3)
    # Linear-scaling reference line anchored at B=1
    anchor = tok_s[0]
    linear_ref = anchor * np.array(Bs, dtype=float) / Bs[0]
    ax_tp.plot(Bs, linear_ref, "--", color="tab:gray", alpha=0.7,
               label="Linear reference (from B=1)")

    for B, t in zip(Bs, tok_s):
        ax_tp.annotate(f"{t:.0f} tok/s", xy=(B, t), xytext=(8, 8),
                       textcoords="offset points", fontsize=10, color="tab:blue")

    ax_tp.set_xscale("log")
    ax_tp.set_xticks(Bs)
    ax_tp.set_xticklabels(Bs)
    ax_tp.set_xlabel("Concurrency (in-flight requests)")
    ax_tp.set_ylabel("Output tokens / second")
    ax_tp.set_title("Throughput grows sub-linearly with concurrency")
    ax_tp.grid(True, alpha=0.3)
    ax_tp.legend(loc="upper left", fontsize=10)
    ax_tp.set_ylim(bottom=0)

    # --- Right panel: latency ---
    ax_lat.plot(Bs, ttft_mean, "o-", color="tab:red", linewidth=2.2, markersize=10,
                label="TTFT mean", zorder=3)
    ax_lat.plot(Bs, ttft_p90, "s--", color="tab:red", linewidth=1.5, markersize=8,
                alpha=0.6, label="TTFT p90")
    ax_lat.plot(Bs, tpot_mean, "o-", color="tab:purple", linewidth=2.2, markersize=10,
                label="TPOT mean", zorder=3)

    for B, v in zip(Bs, ttft_mean):
        ax_lat.annotate(f"{v:.0f} ms", xy=(B, v), xytext=(8, 8),
                        textcoords="offset points", fontsize=10, color="tab:red")
    for B, v in zip(Bs, tpot_mean):
        ax_lat.annotate(f"{v:.1f}", xy=(B, v), xytext=(8, -14),
                        textcoords="offset points", fontsize=10, color="tab:purple")

    ax_lat.set_xscale("log")
    ax_lat.set_xticks(Bs)
    ax_lat.set_xticklabels(Bs)
    ax_lat.set_xlabel("Concurrency (in-flight requests)")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title("Per-request latency grows with concurrency")
    ax_lat.grid(True, alpha=0.3)
    ax_lat.legend(loc="upper left", fontsize=10)
    ax_lat.set_ylim(bottom=0)

    fig.suptitle("Batching Pareto on OLMoE: throughput↑, latency↑\n"
                 "(ShareGPT prompts 128–512 tok, 128-token outputs, bf16 on L4)",
                 fontsize=12, y=1.02)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
