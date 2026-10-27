"""
plots.py — turn runs/*/metrics.jsonl + summary.json into figures and a results table.

Run:  ../04-diffusion/.venv/bin/python plots.py [--runs runs] [--out figures]

Seeds of the same run name are averaged (lines) or shown as mean ± range (table).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

# Color follows the strategy in every figure (validated categorical slots, fixed order).
COLORS = {
    "topk_softmax": "#2a78d6",
    "topk_softmax_cap": "#eb6834",
    "topk_aux": "#1baf7a",
    "topk_aux_nocap": "#eda100",
    "expert_choice": "#e87ba4",
    "expert_choice_seq": "#008300",
    "dense": "#8a8984",  # neutral reference
}
LABELS = {
    "dense": "dense (FLOP-matched)",
    "topk_softmax": "top-2 softmax",
    "topk_softmax_cap": "top-2 softmax, cap 1.25",
    "topk_aux": "top-2 + aux, cap 1.25",
    "topk_aux_nocap": "top-2 + aux, no cap",
    "expert_choice": "expert-choice (batch)",
    "expert_choice_seq": "expert-choice (per-seq)",
    "topk_aux_0.001": "top-2 + aux 0.001, cap 1.25",
    "topk_aux_0.1": "top-2 + aux 0.1, cap 1.25",
}
ORDER = list(LABELS)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def load(runs_dir):
    metrics, summaries = defaultdict(list), defaultdict(list)
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        s_path, m_path = os.path.join(d, "summary.json"), os.path.join(d, "metrics.jsonl")
        if not os.path.exists(s_path):
            continue
        s = json.load(open(s_path))
        summaries[s["name"]].append(s)
        metrics[s["name"]].append([json.loads(l) for l in open(m_path)])
    return metrics, summaries


def curve(runs, key):
    """Mean over seeds of one metric vs step."""
    steps = [r["step"] for r in runs[0]]
    vals = np.array([[r[key] for r in run] for run in runs], dtype=float)
    return np.array(steps), vals.mean(0)


def direct_label(ax, x, y, name):
    ax.annotate(LABELS[name], (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK2)


def line_plot(ax, metrics, key, names, title, ylabel, log=False, skip_first=0):
    ends = []
    for n in names:
        if n not in metrics:
            continue
        x, y = curve(metrics[n], key)
        x, y = x[skip_first:], y[skip_first:]
        ax.plot(x, y, color=COLORS[n], lw=2, label=LABELS[n])
        ax.plot(x[-1], y[-1], "o", ms=5, color=COLORS[n], mec=SURFACE, mew=1.5)
        ends.append((y[-1], x[-1], n))
    # Spread end labels so they don't collide, then re-center the stack on the data
    # so every label stays next to its own line.
    ends.sort()
    if len(ends) > 4:
        # Too many series to label legibly at the line ends — the legend (and the
        # results table) carry identity instead.
        ends = []
    if ends and not log:
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.055
        placed = []
        for y, _, _ in ends:
            placed.append(y if not placed else max(y, placed[-1] + gap))
        shift = np.mean(placed) - np.mean([e[0] for e in ends])
        placed = [p - shift for p in placed]
        for i in range(1, len(placed)):  # re-enforce the gap after the shift
            placed[i] = max(placed[i], placed[i - 1] + gap)
    else:
        placed = [e[0] for e in ends]
    for (_, x, n), yy in zip(ends, placed):
        direct_label(ax, x, yy, n)
    if ends:
        max_step = max(e[1] for e in ends)
        ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= max_step])
        ax.set_xlim(right=max_step * 1.35)
    ax.set_title(title, loc="left")
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=8)


def fig_val_ppl(metrics, out):
    names = ["dense", "topk_softmax", "topk_softmax_cap", "topk_aux", "topk_aux_nocap",
             "expert_choice", "expert_choice_seq"]
    fig, ax = plt.subplots(figsize=(9, 5))
    line_plot(ax, metrics, "val_ppl", names, "Validation perplexity (teacher-forced)",
              "perplexity", skip_first=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "val_ppl.png"), dpi=150)
    plt.close(fig)


def fig_load(metrics, out):
    names = ["topk_softmax", "topk_softmax_cap", "topk_aux", "topk_aux_nocap",
             "expert_choice", "expert_choice_seq"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    line_plot(axes[0], metrics, "load_cv", names, "Processed expert load: coefficient of variation",
              "load CV (0 = balanced)")
    line_plot(axes[1], metrics, "pref_cv", names, "Router preference (plain top-2 on probs): CV",
              "preference CV")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.legend().remove()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    fig.savefig(os.path.join(out, "load_cv.png"), dpi=150)
    plt.close(fig)


def fig_drops(metrics, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    line_plot(axes[0], metrics, "dropped_frac", ["topk_softmax_cap", "topk_aux"],
              "Token-choice: assignments dropped by capacity", "fraction of assignments")
    line_plot(axes[1], metrics, "unrouted_frac",
              ["topk_softmax_cap", "topk_aux", "expert_choice", "expert_choice_seq"],
              "Tokens processed by ZERO experts", "fraction of tokens")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "drops.png"), dpi=150)
    plt.close(fig)


def fig_heatmaps(metrics, out):
    """Final-window processed load per (layer, expert), seed 0, as a multiple of fair share."""
    names = [n for n in ["topk_softmax", "topk_softmax_cap", "topk_aux", "topk_aux_nocap",
                         "expert_choice"] if n in metrics]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names) + 1, 2.9), squeeze=False)
    blues = matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    for ax, n in zip(axes[0], names):
        mat = np.array([l["load_frac"] for l in metrics[n][0][-1]["layers"]])
        mat = mat * mat.shape[1]
        im = ax.imshow(mat, cmap=blues, vmin=0, vmax=4, aspect="auto")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 2.2 else INK)
        ax.set_title(LABELS[n], fontsize=9, loc="left")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))
        ax.set_xlabel("expert")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0][0].set_ylabel("layer")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="load ÷ fair share (1.0 = balanced)")
    fig.savefig(os.path.join(out, "expert_load_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_coverage(metrics, out):
    if not any(n in metrics for n in ["expert_choice", "expert_choice_seq"]):
        return
    fig, ax = plt.subplots(figsize=(7, 3.8))
    width = 0.38
    for i, n in enumerate(["expert_choice", "expert_choice_seq"]):
        if n not in metrics:
            continue
        cov = np.mean([np.mean([l["coverage"] for l in run[-1]["layers"]], 0) for run in metrics[n]], 0)
        xs = np.arange(len(cov)) + (i - 0.5) * width
        ax.bar(xs, cov, width - 0.04, color=COLORS[n], label=LABELS[n])
        for x, v in zip(xs, cov):
            if v > 0.01:
                ax.text(x, v + 0.01, f"{v:.0%}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(cov)))
    ax.set_xlabel("number of experts that processed the token")
    ax.set_ylabel("fraction of tokens")
    ax.set_title("Expert-choice: experts per token (token-choice is always 2)", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ec_coverage.png"), dpi=150)
    plt.close(fig)


def agg(summaries, key):
    v = np.array([s[key] for s in summaries], dtype=float)
    return v.mean(), v.min(), v.max()


def fig_bars(summaries, out, bench=None):
    names = [n for n in ORDER if n in summaries]
    panels = [
        ("final_val_ppl", "Val perplexity (teacher-forced)", "{:.3f}"),
        ("est_causal_ppl", "Est. causal ppl (scored as generated)", "{:.3f}"),
        ("train_tok_per_s", "Training throughput (tok/s)", "{:,.0f}"),
        ("infer_tok_per_s", "Inference throughput (tok/s)", "{:,.0f}"),
    ]
    if bench:
        panels[2] = ("train_tok_per_s", "Training tok/s (interleaved bench)", "{:,.0f}")
        panels[3] = ("infer_tok_per_s", "Inference tok/s (interleaved bench)", "{:,.0f}")

    def value(n, key):
        if key == "est_causal_ppl":
            v = [s["final_val_ppl"] * s["autoregressive_ppl_causal_set"] / s["teacher_forced_ppl_causal_set"]
                 for s in summaries[n]]
            return float(np.mean(v)), min(v), max(v)
        if bench and "tok_per_s" in key:
            return (bench[n][key],) * 3 if n in bench else None
        return agg(summaries[n], key)

    fig, axes = plt.subplots(1, 4, figsize=(17, 0.42 * len(names) + 1.4), sharey=True)
    y = np.arange(len(names))
    for ax, (key, title, fmt) in zip(axes, panels):
        for i, n in enumerate(names):
            v = value(n, key)
            if v is None:
                ax.text(0, i, " not benchmarked", va="center", fontsize=8, color=INK2)
                continue
            m, lo, hi = v
            base = n if n in COLORS else "topk_aux"
            color, alpha = COLORS[base], 1.0 if n in COLORS else 0.55
            if "ppl" in key:
                # Dot + seed range: perplexities differ by a few %, and a bar on a
                # truncated axis would exaggerate that.
                ax.plot([lo, hi], [i, i], color=color, lw=2, alpha=alpha, solid_capstyle="round")
                ax.plot(m, i, "o", ms=8, color=color, alpha=alpha, mec=SURFACE, mew=2)
            else:
                ax.barh(i, m, height=0.7, color=color, alpha=alpha)
            ax.text(max(m, hi) if "ppl" in key else m, i, " " + fmt.format(m), va="center", fontsize=8, color=INK2)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_axisbelow(True)
        ax.grid(axis="y", visible=False)
        if "tok_per_s" in key:
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}k"))
        vals = [v[0] for v in (value(n, key) for n in names) if v is not None]
        if "ppl" in key:
            ax.set_xlim(min(vals) * 0.97, max(vals) * 1.03)
        else:
            ax.set_xlim(0, max(vals) * 1.25)
    axes[0].set_yticks(y, [LABELS[n] for n in names])
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "summary_bars.png"), dpi=150)
    plt.close(fig)


def est_causal(S):
    return float(np.mean([s["final_val_ppl"] * s["autoregressive_ppl_causal_set"]
                          / s["teacher_forced_ppl_causal_set"] for s in S]))


def results_table(metrics, summaries, out, bench=None):
    rows = ["| strategy | seeds | val ppl (teacher-forced) | est. causal ppl | AR − TF gap | load CV | pref CV | dropped | zero-expert tokens "
            "| leaks future? | train tok/s | infer tok/s |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    # est. causal ppl: rescale the big-sample val ppl by the paired AR/TF ratio measured
    # on the small causal-eval set. Equal to val ppl for strictly causal routers.
    for n in [n for n in ORDER if n in summaries]:
        S = summaries[n]
        last = [run[-1] for run in metrics[n]]
        lm = lambda k: np.mean([r[k] for r in last])  # noqa: E731
        ppl, lo, hi = agg(S, "final_val_ppl")
        gap = np.mean([s["autoregressive_ppl_causal_set"] - s["teacher_forced_ppl_causal_set"] for s in S])
        leak = np.mean([s["frac_positions_changed"] for s in S])
        if bench:
            tr = f"{bench[n]['train_tok_per_s']:,.0f}" if n in bench else "–"
            inf = f"{bench[n]['infer_tok_per_s']:,.0f}" if n in bench else "–"
        else:
            tr, inf = f"{agg(S, 'train_tok_per_s')[0]:,.0f}", f"{agg(S, 'infer_tok_per_s')[0]:,.0f}"
        rows.append(
            f"| {LABELS[n]} | {len(S)} | {ppl:.3f} ({lo:.3f}–{hi:.3f}) | {est_causal(S):.3f} "
            f"| {gap:+.3f} | {lm('load_cv'):.2f} | {lm('pref_cv'):.2f} | {lm('dropped_frac'):.1%} "
            f"| {lm('unrouted_frac'):.1%} | {leak:.0%} of positions | {tr} | {inf} |")
    text = "\n".join(rows) + "\n"
    with open(os.path.join(out, "results.md"), "w") as f:
        f.write(text)
    print(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    metrics, summaries = load(args.runs)
    fig_val_ppl(metrics, args.out)
    fig_load(metrics, args.out)
    fig_drops(metrics, args.out)
    fig_heatmaps(metrics, args.out)
    fig_coverage(metrics, args.out)
    bench_path = os.path.join(args.out, "throughput_bench.json")
    bench = json.load(open(bench_path)) if os.path.exists(bench_path) else None
    fig_bars(summaries, args.out, bench)
    results_table(metrics, summaries, args.out, bench)
    print(f"figures -> {args.out}/")


if __name__ == "__main__":
    main()
