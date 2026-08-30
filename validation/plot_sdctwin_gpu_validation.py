"""Figure: SDCTwin's GPU power-throughput model vs. measurement (H200, summarize-qwen7b).

A single ACM-column figure for the demo paper: two panels, (a) prefill and (b) decode, each
showing the MEASURED points of one workload on one card against the analytical curve of
MODEL_AND_RESULTS.zh.md, with the panel's MAPE.

Nothing is fitted or re-defined here. The script imports validation/validate_model.py and calls
its load(), so the arrays, the fit and the power-axis rule are BYTE-FOR-BYTE the ones behind
validation/val_mape.csv:
  * prefill is a LOCKED-CLOCK sweep at the 700 W cap -> x = measured draw, 346-690 W, 6 points,
    all kept (the low-clock points are set clocks, not caps that failed to engage);
  * decode is a CAP sweep -> x = the enforced cap, and fitlib.cap_sweep_mask has already dropped
    the clock-floor points (200/226/266 W), leaving 313-700 W, 7 points.
The two MAPEs are asserted against the published 2.43% / 4.18% so the figure can never silently
drift away from the table.

python3 validation/plot_sdctwin_gpu_validation.py -> fig_gpu_validation.pdf + .png in this folder
"""
from __future__ import annotations
import os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validate_model as vm                                          # noqa: E402  puts fitlib on the path
import fitlib                                                        # noqa: E402
import palette                                                       # noqa: E402  the paper palette

HW, WID = "H200", "summarize-qwen7b"
PHASES = (("prefill", "(a) Prefill"), ("decode", "(b) Decode"))
EXPECT = {"prefill": 2.43, "decode": 4.18}       # validation/val_mape.csv, rounded to 2 dp
NDENSE = 200                                     # model curve resolution, inside the measured domain

# Navy points / orange curve: the repo palette's contrasting pair, and the one pairing that stays
# separable under deuteranopia. Both panels use it, so hue means measured-vs-model, never phase.
C_MEAS, C_MODEL = palette.PAL["navy"], palette.PAL["orange"]
INK, MUTE = palette.INK, palette.MUTE

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,                           # TrueType, not Type-3 (ACM)
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "lines.linewidth": 1.0, "lines.markersize": 4,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    # the one piece of setup main() would have done for us
    vm.FIG_HW[HW]["f_max"] = fitlib.resolve_f_max(vm.FIG_HW[HW]["data"])
    d = vm.load(HW, WID)

    fig, axes = plt.subplots(1, 2, figsize=(3.33, 1.45), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.03, hspace=0.0)
    report = []
    for ax, (phase, title) in zip(axes, PHASES):
        s = d[phase]
        P, T, fn = s["P"], s["T"], s["fn"]
        m = vm.mape(T, fn(P))
        report.append((phase, len(P), float(P.min()), float(P.max()), m))
        if abs(round(m, 2) - EXPECT[phase]) > 5e-3:
            raise SystemExit(f"{phase}: MAPE {m:.2f}% != published {EXPECT[phase]}% — "
                             "the pipeline moved, check it before touching the figure")

        Pd = np.linspace(P.min(), P.max(), NDENSE)
        ax.plot(Pd, fn(Pd), "-", color=C_MODEL, label="OrbiWiz model", zorder=2)
        ax.plot(P, T, "o", mfc="none", mec=C_MEAS, mew=0.9, ls="none",
                label=f"Measured ({HW})", zorder=3)

        ax.set_xticks([x for x in (300, 400, 500, 600, 700) if P.min() - 40 <= x <= P.max() + 40])
        ax.set_title(title, pad=2.5)
        ax.set_xlabel("GPU power (W)", labelpad=1.5)
        ax.set_ylabel("Throughput (tok/s)", labelpad=1.5)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.text(0.04, 0.96, f"MAPE {m:.1f}%", transform=ax.transAxes,
                ha="left", va="top", fontsize=7, color=INK)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTE)
        ax.tick_params(colors=MUTE, labelcolor=INK)

    # One legend for the whole figure, on its own row under the panels: at this size any in-axes
    # box would sit on top of a curve — both phases rise across the full panel.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="outside lower center", ncol=2, frameon=False,
               handlelength=1.5, handletextpad=0.5, columnspacing=1.6, borderaxespad=0.0,
               borderpad=0.0)

    for ext, kw in (("pdf", {}), ("png", dict(dpi=300))):
        p = os.path.join(HERE, f"fig_gpu_validation.{ext}")
        fig.savefig(p, **kw)
        print("wrote", p)
    plt.close(fig)

    print(f"\n{HW} / {WID}")
    for phase, n, lo, hi, m in report:
        print(f"  {phase:<8} n_points={n}  power {lo:.0f}-{hi:.0f} W  MAPE {m:.2f}%")


if __name__ == "__main__":
    main()
