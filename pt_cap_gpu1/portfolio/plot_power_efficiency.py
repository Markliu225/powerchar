#!/usr/bin/env python3
"""Overview figures from a portfolio data dir (e.g. data_h200):
  fig_h200_power_vs_throughput.png   power vs tok/s   (prefill linear | decode log)
  fig_h200_power_vs_tokperJ.png      power vs tok/J   (prefill linear | decode log)
x = power cap (cap_w), the control variable — NOT measured power.
tok/J is computed here as throughput_tok_s / power_sample_avg_w; the CSV's
tok_per_joule column (energy-counter based, persistence mode was off) is ignored.
Color = model, marker = workload within model; every line direct-labeled.
"""
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed order: color slot per model, marker order within model
GROUPS = [
    ("Phi-3-mini-4k", "#2a78d6", ["chat-phi3", "rag-phi3", "code-phi3", "longform-phi3"]),
    ("Qwen2.5-7B",    "#1baf7a", ["summarize-qwen7b", "classify-qwen7b"]),
    ("Qwen2.5-3B",    "#eda100", ["translate-qwen3b", "fastchat-qwen15b"]),
    ("Qwen3-4B",      "#008300", ["qwen3chat-4b", "qwen3think-4b"]),
]
MARKERS = ["o", "s", "^", "D"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e8e8e6"


def load(data_dir, wid, phase):
    path = os.path.join(data_dir, f"{wid}_{phase}.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    pts = sorted((float(r["power_sample_avg_w"]), float(r["throughput_tok_s"]),
                  float(r["throughput_tok_s"]) / float(r["power_sample_avg_w"]))
                 for r in rows)
    return ([p for p, _, _ in pts], [t for _, t, _ in pts], [e for _, _, e in pts])


def style(ax):
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def end_labels(ax, items, min_gap_px=11):
    """items: (x_end, y_end, text, color). Spread labels vertically in display
    space at a common x just right of the data; leader line if displaced."""
    if not items:
        return
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + 0.30 * (xmax - xmin))
    x_lab = max(x for x, _, _, _ in items) + 0.03 * (ax.get_xlim()[1] - xmin)
    disp = ax.transData.transform([(x, y) for x, y, _, _ in items])
    order = sorted(range(len(items)), key=lambda i: disp[i][1])
    ys = [disp[i][1] for i in order]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + min_gap_px)
    inv = ax.transData.inverted()
    for k, i in enumerate(order):
        x_end, y_end, text, color = items[i]
        y_lab = inv.transform((disp[i][0], ys[k]))[1]
        if abs(ys[k] - disp[i][1]) > 6:
            ax.plot([x_end, x_lab - 0.01 * (ax.get_xlim()[1] - xmin)],
                    [y_end, y_lab], color=GRID, lw=0.6, zorder=1)
        ax.annotate(text, (x_lab, y_lab), fontsize=7.5, color=INK,
                    va="center", ha="left", annotation_clip=False)


def panel(ax, data_dir, phase, ycol, title, ylab, logy):
    ends = []
    for model, color, wids in GROUPS:
        for j, wid in enumerate(wids):
            d = load(data_dir, wid, phase)
            if d is None:
                continue
            P, T, E = d
            Y = T if ycol == "throughput" else E
            ax.plot(P, Y, color=color, lw=1.8, marker=MARKERS[j], ms=5,
                    mew=0.8, mec="white", zorder=3)
            ends.append((P[-1], Y[-1], wid, color))
    if logy:
        ax.set_yscale("log")
    style(ax)
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.set_xlabel("power cap (W)", fontsize=10, color=INK)
    ax.set_ylabel(ylab, fontsize=10, color=INK)
    end_labels(ax, ends)


def figure(data_dir, out, ycol, suptitle, ylab):
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.8))
    panel(axs[0], data_dir, "prefill", ycol, "PREFILL", ylab, logy=False)
    panel(axs[1], data_dir, "decode", ycol, "DECODE  (log scale)", ylab, logy=True)
    handles = [plt.Line2D([], [], color=c, lw=3, label=m) for m, c, _ in GROUPS]
    axs[0].legend(handles=handles, loc="upper left", fontsize=8.5,
                  frameon=False, title="model", title_fontsize=8.5)
    fig.suptitle(suptitle, fontsize=13, color=INK)
    fig.text(0.5, 0.012,
             "x = power cap (cap_w), grid 200–700 W  ·  tok/J = throughput / power_sample_avg_w (computed, CSV column ignored)  ·  "
             "chat-phi3: smoke run, 4 caps only  ·  "
             "fastchat-qwen15b actually loaded Qwen2.5-3B (config slip → overlaps translate-qwen3b)",
             ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--data", default=os.path.join(root, "data_h200"))
    ap.add_argument("--out", default=os.path.join(root, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    figure(a.data, os.path.join(a.out, "fig_h200_power_vs_throughput.png"),
           "throughput", "H200 · token throughput vs power cap — 10 workloads × 4 models",
           "throughput (tok/s)")
    figure(a.data, os.path.join(a.out, "fig_h200_power_vs_tokperJ.png"),
           "tokperj", "H200 · energy efficiency vs power cap — 10 workloads × 4 models",
           "energy efficiency (tok/J)")


if __name__ == "__main__":
    main()
