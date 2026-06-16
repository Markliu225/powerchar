"""The two requested curves, from the steady-state (jitter-free) data:

  Fig 1  figures/pt_efficiency_vs_power.png   x = GPU power (W),  y = efficiency (tok/J = (tok/s)/W)
  Fig 2  figures/pt_power_vs_throughput.png   x = throughput (tok/s),  y = GPU power (W)

Both phases on each axis. Log scales are used where the prefill/decode dynamic
range demands it, so both phases stay readable. Each point is annotated with its
batch; the SM-clock spread per point is small (steady state), confirming low jitter.

  python code/plot_pt.py
"""
from __future__ import annotations
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C


def load(name):
    with open(os.path.join(C.RESULTS_DIR, name)) as f:
        rows = [{k: (float(v) if v not in ("", None) else float("nan")) if k not in ("phase",) else v
                 for k, v in r.items()} for r in csv.DictReader(f)]
    return sorted(rows, key=lambda r: r["batch"])


def annotate(ax, xs, ys, bs):
    for x, y, b in zip(xs, ys, bs):
        ax.annotate(f"b{int(b)}", (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")


def main():
    pre = load("prefill.csv")
    dec = load("decode.csv")
    cap = 250.0
    series = [("prefill (compute-bound)", pre, "C1", "s"),
              ("decode (memory-bound)", dec, "C0", "o")]

    # ---- Fig 1: efficiency (tok/J) vs power (W) ----------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    for lbl, rows, c, m in series:
        P = [r["power_avg_w"] for r in rows]
        E = [r["tok_per_joule"] for r in rows]
        B = [r["batch"] for r in rows]
        Pe = [r.get("power_std_w", 0.0) for r in rows]           # x error bar = power jitter
        ax.errorbar(P, E, xerr=Pe, fmt=m + "-", color=c, ms=8, capsize=3,
                    elinewidth=1, label=lbl)
        annotate(ax, P, E, B)
    ax.axvline(cap, color="r", ls="--", alpha=.5, label=f"power cap {cap:.0f} W")
    ax.set_yscale("log")
    ax.set_xlabel("GPU power (W)   — error bar = ±1σ power jitter over the window")
    ax.set_ylabel("energy efficiency  (tok/J = tokens/s ÷ watts)")
    ax.set_title("Efficiency vs power  (steady-state, clock un-throttled)")
    ax.legend(); ax.grid(alpha=.3, which="both"); fig.tight_layout()
    out1 = os.path.join(C.FIGURES_DIR, "pt_efficiency_vs_power.png")
    fig.savefig(out1, dpi=130); print("wrote", out1)

    # ---- Fig 2: power (W) vs throughput (tok/s) ----------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    for lbl, rows, c, m in series:
        T = [r["throughput_tok_s"] for r in rows]
        P = [r["power_avg_w"] for r in rows]
        B = [r["batch"] for r in rows]
        Pe = [r.get("power_std_w", 0.0) for r in rows]           # y error bar = power jitter
        ax.errorbar(T, P, yerr=Pe, fmt=m + "-", color=c, ms=8, capsize=3,
                    elinewidth=1, label=lbl)
        annotate(ax, T, P, B)
    ax.axhline(cap, color="r", ls="--", alpha=.5, label=f"power cap {cap:.0f} W")
    ax.set_xscale("log")
    ax.set_xlabel("token throughput (tok/s)")
    ax.set_ylabel("GPU power (W)   — error bar = ±1σ power jitter over the window")
    ax.set_title("Power vs throughput  (steady-state batch sweep, both phases)")
    ax.legend(); ax.grid(alpha=.3, which="both"); fig.tight_layout()
    out2 = os.path.join(C.FIGURES_DIR, "pt_power_vs_throughput.png")
    fig.savefig(out2, dpi=130); print("wrote", out2)


if __name__ == "__main__":
    main()
