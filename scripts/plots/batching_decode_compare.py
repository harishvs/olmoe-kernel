"""Compare ShareGPT batching vs pure-decode batching.

Both experiments swept concurrency B in {1, 4, 16, 32}. The gap between the
two TPOT curves shows the effect of prefill-mixing; their shared slope shows
the effect of MoE routing at high batch.
"""
import json
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SG_SUMMARY = sorted(glob(str(REPO / "results/harness/batching/summary_*.json")))[-1]
PD_SUMMARY = sorted(glob(str(REPO / "results/harness/batching_decode/summary_*.json")))[-1]
OUT = REPO / "talk" / "figures" / "batching_compare.png"


def load(path):
    s = json.load(open(path))
    Bs = sorted(int(k) for k in s.keys())
    tok_s = np.array([s[str(B)]["output_tokens_per_s"] for B in Bs])
    tpot = np.array([s[str(B)]["tpot_ms"]["mean"] for B in Bs])
    return np.array(Bs), tok_s, tpot


def main():
    sg_Bs, sg_tps, sg_tpot = load(SG_SUMMARY)
    pd_Bs, pd_tps, pd_tpot = load(PD_SUMMARY)
    # Union of Bs for axis ticks
    Bs = sorted(set(sg_Bs.tolist()) | set(pd_Bs.tolist()))

    fig, (ax_tp, ax_lat) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left panel: throughput ---
    ax_tp.plot(pd_Bs, pd_tps, "o-", color="tab:blue", linewidth=2.2, markersize=10,
               label="Pure decode (1-tok prompt)", zorder=3)
    ax_tp.plot(sg_Bs, sg_tps, "s-", color="tab:orange", linewidth=2.2, markersize=10,
               label="ShareGPT (128–512 tok prompt)", zorder=3)
    anchor = pd_tps[0]
    linear_ref = anchor * pd_Bs.astype(float) / pd_Bs[0]
    ax_tp.plot(pd_Bs, linear_ref, "--", color="tab:gray", alpha=0.6,
               label="Linear reference (from B=1)")

    for B, t in zip(pd_Bs, pd_tps):
        ax_tp.annotate(f"{t:.0f}", xy=(B, t), xytext=(8, 6),
                       textcoords="offset points", fontsize=9, color="tab:blue")
    for B, t in zip(sg_Bs, sg_tps):
        ax_tp.annotate(f"{t:.0f}", xy=(B, t), xytext=(8, -14),
                       textcoords="offset points", fontsize=9, color="tab:orange")

    ax_tp.set_xscale("log")
    ax_tp.set_xticks(Bs)
    ax_tp.set_xticklabels(Bs)
    ax_tp.set_xlabel("Concurrency")
    ax_tp.set_ylabel("Output tokens / second")
    ax_tp.set_title("Throughput vs concurrency")
    ax_tp.grid(True, alpha=0.3)
    ax_tp.legend(loc="upper left", fontsize=10)
    ax_tp.set_ylim(bottom=0)

    # --- Right panel: TPOT ---
    ax_lat.plot(pd_Bs, pd_tpot, "o-", color="tab:blue", linewidth=2.2, markersize=10,
                label="Pure decode (1-tok prompt)", zorder=3)
    ax_lat.plot(sg_Bs, sg_tpot, "s-", color="tab:orange", linewidth=2.2, markersize=10,
                label="ShareGPT (128–512 tok prompt)", zorder=3)

    for B, t in zip(pd_Bs, pd_tpot):
        ax_lat.annotate(f"{t:.1f}", xy=(B, t), xytext=(8, -14),
                        textcoords="offset points", fontsize=9, color="tab:blue")
    for B, t in zip(sg_Bs, sg_tpot):
        ax_lat.annotate(f"{t:.1f}", xy=(B, t), xytext=(8, 6),
                        textcoords="offset points", fontsize=9, color="tab:orange")

    ax_lat.set_xscale("log")
    ax_lat.set_xticks(Bs)
    ax_lat.set_xticklabels(Bs)
    ax_lat.set_xlabel("Concurrency")
    ax_lat.set_ylabel("TPOT (ms/token)")
    ax_lat.set_title("Per-user decode time vs concurrency")
    ax_lat.grid(True, alpha=0.3)
    ax_lat.legend(loc="upper left", fontsize=10)
    ax_lat.set_ylim(bottom=0)

    fig.suptitle("MoE batching: pure decode isn't much better than mixed\n"
                 "Gap between curves = prefill-mixing cost. Shared slope = MoE tax.",
                 fontsize=12, y=1.02)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
