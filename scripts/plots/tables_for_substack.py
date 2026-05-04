"""Render the substack post's tables as PNG images.

Substack's editor mangles HTML tables on paste. Upload these PNGs instead.
Uses matplotlib table rendering with consistent styling.
"""
from pathlib import Path

import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parents[2] / "talk" / "figures" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared styling constants
HEADER_FACE = "#2b4a6b"
HEADER_TEXT = "white"
ROW_ALT = "#f7f7f7"
ROW_FACE = "white"
FONT_SIZE = 11
HEADER_FONT_SIZE = 11


def render_table(rows: list[list[str]], out_name: str, col_weights=None,
                 highlight_col: int | None = None, highlight_rows: set[int] | None = None):
    """
    rows[0] is header. Remaining rows are body.
    col_weights: relative widths per column. Default uniform.
    highlight_col: if set, bold that column across body rows.
    highlight_rows: set of body-row indices (0-based in the BODY) to highlight (e.g. GSM8K row).
    """
    n_cols = len(rows[0])
    n_rows = len(rows)
    if col_weights is None:
        col_weights = [1.0] * n_cols

    # Bigger figure. Extra width for roomy columns, extra height for row padding.
    fig_w = max(1.4 * sum(col_weights) + 1.5, 7.0)
    fig_h = 0.75 * n_rows + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    # Generous margins so nothing clips at the edges
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)

    # Normalize column widths
    total = sum(col_weights)
    col_w = [w / total for w in col_weights]

    tbl = ax.table(
        cellText=rows,
        cellLoc="center",
        loc="center",
        colWidths=col_w,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FONT_SIZE)

    # Set EXPLICIT cell heights (otherwise matplotlib shrinks the header).
    # height is a fraction of axes height. 1/n_rows gives uniform sizing.
    for (r, c), cell in tbl.get_celld().items():
        cell.set_height(1.0 / n_rows)
        cell.PAD = 0.06  # more breathing room around text

    # Style header
    for c in range(n_cols):
        cell = tbl[0, c]
        cell.set_facecolor(HEADER_FACE)
        cell.set_text_props(color=HEADER_TEXT, weight="bold", fontsize=HEADER_FONT_SIZE)
        cell.set_edgecolor("#1a3550")

    # Style body rows (zebra + highlights)
    highlight_rows = highlight_rows or set()
    for r in range(1, n_rows):
        body_r = r - 1  # 0-based body index
        for c in range(n_cols):
            cell = tbl[r, c]
            if body_r in highlight_rows:
                cell.set_facecolor("#fff2cc")
                cell.set_text_props(weight="bold")
            elif body_r % 2 == 0:
                cell.set_facecolor(ROW_FACE)
            else:
                cell.set_facecolor(ROW_ALT)
            cell.set_edgecolor("#d0d0d0")
            if highlight_col is not None and c == highlight_col:
                cur = cell.get_text()
                cur.set_weight("bold")

    out_path = OUT_DIR / out_name
    # Use a fixed pad instead of "tight" so nothing clips
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.2,
                facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---- Table 1: baseline measured ----
render_table(
    rows=[
        ["Input length", "TTFT mean", "TTFT p90", "End-to-end @ O=128", "TPOT / token"],
        ["32",   "41.5 ms", "41.6 ms", "1314 ms", "10.0 ms"],
        ["128",  "51.1 ms", "51.2 ms", "1342 ms", "10.2 ms"],
        ["512",  "67.0 ms", "67.7 ms", "1386 ms", "10.4 ms"],
        ["1024", "94.1 ms", "96.1 ms", "1447 ms", "10.7 ms"],
    ],
    out_name="baseline.png",
    col_weights=[1.2, 1.0, 1.0, 1.7, 1.2],
)

# ---- Table 2: prefill kernel breakdown ----
render_table(
    rows=[
        ["Kernel", "CUDA time", "% of TTFT"],
        ["fused_moe_kernel",        "66.9 ms", "79.5%"],
        ["attention projections",   "7.9 ms",  "9.4%"],
        ["activation (SiLU)",       "2.3 ms",  "2.7%"],
        ["flash attention math",    "1.9 ms",  "2.2%"],
        ["topk_softmax (router)",   "0.08 ms", "0.09%"],
    ],
    out_name="prefill_kernels.png",
    col_weights=[2.0, 1.0, 1.0],
    highlight_rows={0, 4},  # fused_moe_kernel and the router row
)

# ---- Table 3: FP8 accuracy ----
render_table(
    rows=[
        ["Task", "bf16", "FP8", "Δ"],
        ["MMLU (5-shot)",   "52.34%", "52.19%", "-0.15 pp"],
        ["HellaSwag",       "60.62%", "60.25%", "-0.37 pp"],
        ["ARC-Challenge",   "49.49%", "49.57%", "+0.08 pp"],
        ["GSM8K (math)",    "35.03%", "33.13%", "-1.90 pp (-5.4%)"],
    ],
    out_name="fp8_accuracy.png",
    col_weights=[1.5, 0.9, 0.9, 1.5],
    highlight_rows={3},  # GSM8K row, the material finding
)

# ---- Table 4: batching sweep ----
render_table(
    rows=[
        ["Batch (B)", "tok/s", "TPOT", "TTFT p90"],
        ["1",   "101",  "10.0 ms", "34 ms"],
        ["4",   "163",  "23.3 ms", "87 ms"],
        ["16",  "387",  "42.9 ms", "269 ms"],
        ["32",  "581",  "51.1 ms", "389 ms"],
        ["64",  "1128", "56.7 ms", "679 ms"],
        ["128", "2055", "66.1 ms", "1873 ms"],
    ],
    out_name="batching.png",
    col_weights=[1.0, 0.9, 1.0, 1.1],
    highlight_rows={5},  # B=128 peak
)

print("\nAll tables written to", OUT_DIR)
