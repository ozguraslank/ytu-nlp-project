import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import numpy as np

GREEN, RED = "tab:green", "tab:red"


def moving_average(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        w = max(1, len(x))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def zoom_ylim(curves, floor=0.0):
    """Tight Y-limits so near-1.0 structure is visible."""
    lo = min(c.min() for c in curves)
    lo = max(floor, lo - 0.05 * (1.0 - lo) - 1e-4)
    return lo, 1.0 + 0.02 * (1.0 - lo) + 1e-4


def draw_question(ax, rows, window, title=None):
    curves = []
    for r in rows:
        ma = moving_average(r["top10_prob_sums"], window)
        curves.append(ma)
        ax.plot(ma, color=GREEN if r["label"] == 1 else RED,
                alpha=0.65, linewidth=1.2)
    ax.set_ylim(*zoom_ylim(curves))
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="answers.jsonl")
    ap.add_argument("--question_id", type=int, default=None,
                    help="question for figure 1; default = first mixed one")
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--grid_n", type=int, default=9,
                    help="number of questions in the small-multiples grid")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    print(f"Length of rows: {len(rows)}")
    by_q = {}
    for r in rows:
        by_q.setdefault(r["question_id"], []).append(r)
    qids = sorted(by_q)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    saved = []

    # ---------- Figure 1: single question, zoomed ----------------------
    qid = args.question_id if args.question_id is not None else qids[0]
    if qid not in by_q:
        raise SystemExit(f"question_id {qid} not found in {args.data}")
    fig, ax = plt.subplots(figsize=(11, 6))
    draw_question(ax, by_q[qid], args.window)
    ax.plot([], [], color=GREEN, label="Correct answers")
    ax.plot([], [], color=RED, label="Incorrect answers")
    ax.set_xlabel("Token index")
    ax.set_ylabel(f"Sum of top-10 token probabilities (MA, w={args.window})")
    nc = sum(r["label"] for r in by_q[qid])
    ax.set_title(f"question_id={qid}  ({nc} correct / "
                 f"{len(by_q[qid]) - nc} incorrect) — Y-axis zoomed")
    ax.legend(loc="lower left")
    fig.tight_layout()
    p = os.path.join(outdir, f"top10_ma_q{qid}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    saved.append(p)

    # ---------- Figure 2: small multiples grid -------------------------
    n = min(args.grid_n, len(qids))
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, q in zip(axes, qids[:n]):
        nc = sum(r["label"] for r in by_q[q])
        draw_question(ax, by_q[q], args.window,
                      title=f"q={q} ({nc}✓ {len(by_q[q]) - nc}✗)")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Top-10 probability mass (MA) — green=correct, "
                 "red=incorrect, Y-axes zoomed per question", y=1.0)
    fig.supxlabel("Token index"); fig.supylabel("Top-10 prob. sum (MA)")
    fig.tight_layout()
    p = os.path.join(outdir, "top10_ma_grid.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    saved.append(p)

    for p in saved:
        if not os.path.isfile(p):
            raise SystemExit(f"ERROR: figure was not written to {p}")
        print(f"Saved -> {p}")


if __name__ == "__main__":
    main()
