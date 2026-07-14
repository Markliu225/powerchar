"""Per-use-case-class power curves on H200: power vs throughput & power vs tok/J, PER PHASE.

Same pipeline as ../plot_power_curves.py (V100), retargeted at the H200 dataset. For each class
plot the PREFILL and DECODE curves of the mapped measured workload separately (MAP imported,
not copied), on a log throughput axis.

H200 data revision (2026-07-13): PREFILL is now a CLOCK sweep at a fixed 700 W cap, so its power
axis is the MEASURED draw (power_avg_w, ~300-710 W) — this cleanly fixes the earlier low-cap
cap-under-enforcement artifact. DECODE is still a cap sweep (200-700 W). The shared loader
(../plot_power_curves.py) auto-detects the axis per phase. `classify-qwen7b` (the Extract mapping)
was dropped from this data revision, so the Extract class is omitted here (9 of 10 classes).

python3 plot_power_curves.py -> fig_workload_power_throughput.png
                                fig_workload_power_tokj.png
                                workload_power_curves.csv     (all in this h200/ folder)
"""
from __future__ import annotations
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)                  # workload_analysis/
ROOT = os.path.dirname(PARENT)
sys.path.insert(0, PARENT)
import plot_power_curves as V                   # noqa: E402  the V100 pipeline (shared parts)

DATA = os.path.join(ROOT, "data_h200")
CAP_LO, CAP_HI = 200.0, 700.0                   # H200 power range (decode cap span; prefill ~300-710)
V.DATA, V.CAP_LO, V.CAP_HI = DATA, CAP_LO, CAP_HI
V.F_MAX = V.fitlib.resolve_f_max(DATA)

NAME, MAP, BANDS, CAVEAT = V.NAME, V.MAP, V.BANDS, V.CAVEAT
band_of, ratio_str, fmt = V.band_of, V.ratio_str, V.fmt
PRE_C, DEC_C = V.PRE_C, V.DEC_C
INK, INK2, MUTE, GRID = V.INK, V.INK2, V.MUTE, V.GRID

FOOT = ("prefill: CLOCK-swept at a fixed 700 W cap → x = MEASURED draw power_avg_w (this revision fixes the "
        "earlier low-cap under-enforcement)  ·  decode: cap-swept → x = enforced cap  ·  dots = raw measured grid  ·  "
        "P:D = the class's aggregate ratio from workload_ratios.csv (labels the class; enters no curve)\n"
        "* = the class-to-workload mapping carries an accounting/scale caveat (Chat: trace accounting — serving "
        "shape with KV reuse ~1:2;  Summarization / Closed QA / Code: Dolly- or 2023-trace ratios on production-scale "
        "curves), see rack_power_capping/v100/WORKLOADS.zh.md §2  ·  Extract omitted: classify-qwen7b not in this data revision")


def _avail(wid):
    return os.path.exists(os.path.join(DATA, f"{wid}_prefill.csv"))


def panels(fig_axes, classes, curves, draw):
    """Common panel chrome (H200 power axis, log y); `draw(ax, cl, cv)` adds the metric content."""
    for ax, cl in zip(fig_axes, classes):
        cv = curves[MAP[cl["klass"]]]
        bname, _, _, bcol = band_of(cl["r"])
        ax.set_yscale("log")
        draw(ax, cl, cv)
        star = "*" if cl["klass"] in CAVEAT else ""
        ax.set_title(f"{NAME[cl['klass']]}   P:D {ratio_str(cl['r'])}{star}",
                     fontsize=10.5, color=INK, pad=22)
        ax.text(0.5, 1.10, f"via {MAP[cl['klass']]} · {V.MODEL_SHORT[MAP[cl['klass']]]} · "
                f"dec {cv['ctx']}×{cv['b_dec']}", transform=ax.transAxes,
                ha="center", fontsize=7.3, color=INK2)
        ax.text(0.03, 0.96, bname, transform=ax.transAxes, ha="left", va="top",
                fontsize=7.8, color=bcol, weight="bold")
        ax.set_xlim(180, 720)
        ax.set_xticks([200, 300, 400, 500, 600, 700])
        ax.grid(alpha=.45, color=GRID, lw=0.7, which="both")
        ax.tick_params(labelsize=8, colors=MUTE)
        [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]
        [ax.spines[s].set_color(GRID) for s in ("left", "bottom")]
    for ax in fig_axes[len(classes):]:          # hide any leftover panels (9 classes in a 2x5 grid)
        ax.set_visible(False)


def phase_legend(fig):
    fig.legend(handles=[Line2D([], [], color=PRE_C, lw=2, label="prefill (measured draw)"),
                        Line2D([], [], color=DEC_C, lw=2, label="decode (enforced cap)")],
               loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=9, frameon=False)


def main():
    rows = list(csv.DictReader(open(os.path.join(PARENT, "workload_ratios.csv"))))
    classes = sorted([dict(klass=r["klass"], r=float(r["ratio_agg"])) for r in rows
                      if _avail(MAP[r["klass"]])], key=lambda c: c["r"])   # decode-heavy -> prefill-heavy
    curves = {wid: V.load_curves(wid) for wid in sorted({MAP[c["klass"]] for c in classes})}
    g = np.linspace(CAP_LO, CAP_HI, 1501)

    # ---------------- figure 1: power vs throughput (per phase, log y) ----------------
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.4))
    axes = axes.ravel()

    def draw_T(ax, cl, cv):
        for rng, fn, x, y, c in ((cv["pre_rng"], cv["Tpre"], cv["pre_x"], cv["pre_y"], PRE_C),
                                 (cv["dec_rng"], cv["Tdec"], cv["dec_x"], cv["dec_y"], DEC_C)):
            gg = g[(g >= rng[0]) & (g <= rng[1])]
            T = fn(gg)
            ax.plot(gg, T, color=c, lw=2, zorder=3)
            ax.plot(x, y, "o", ms=4.5, color=c, mec="white", mew=0.9, zorder=4)
            ax.annotate(fmt(T[-1]), (gg[-1], T[-1]), textcoords="offset points",
                        xytext=(-2, 6), ha="right", fontsize=7.8, color=c)
        lo = min(cv["pre_y"].min(), cv["dec_y"].min())
        hi = max(cv["pre_y"].max(), cv["dec_y"].max())
        ax.set_ylim(lo * 0.35, hi * 3.2)

    panels(axes, classes, curves, draw_T)
    for ax in axes[5:]:
        ax.set_xlabel("GPU power (W)", fontsize=8.5, color=INK2)
    for ax in (axes[0], axes[5]):
        ax.set_ylabel("throughput (tok/s, log)", fontsize=8.5, color=INK2)
    fig.suptitle("Power vs throughput on H200 — prefill vs decode, per use-case class\n"
                 "prefill clock-swept (x = measured draw), decode cap-swept (x = enforced cap); "
                 "log axis — the phases sit 1-2 orders of magnitude apart",
                 fontsize=12.5, color=INK)
    phase_legend(fig)
    fig.text(0.5, 0.012, FOOT, ha="center", fontsize=7.2, color=INK2)
    fig.tight_layout(rect=(0, 0.065, 1, 0.90))
    fig.savefig(os.path.join(HERE, "fig_workload_power_throughput.png"), dpi=130, bbox_inches="tight")
    print("wrote fig_workload_power_throughput.png")

    # ---------------- figure 2: power vs tok/J (per phase, log y) ----------------
    fig2, axes2 = plt.subplots(2, 5, figsize=(16, 7.4))
    axes2 = axes2.ravel()

    def draw_E(ax, cl, cv):
        effs = []
        for rng, fn, x, y, c in ((cv["pre_rng"], cv["Tpre"], cv["pre_x"], cv["pre_y"], PRE_C),
                                 (cv["dec_rng"], cv["Tdec"], cv["dec_x"], cv["dec_y"], DEC_C)):
            gg = g[(g >= rng[0]) & (g <= rng[1])]
            E = fn(gg) / gg
            m = y / x
            ax.plot(gg, E, color=c, lw=2, zorder=3)
            ax.plot(x, m, "o", ms=4.5, color=c, mec="white", mew=0.9, zorder=4)
            i = int(np.argmax(E))
            right = gg[i] > (rng[0] + rng[1]) / 2
            ax.plot(gg[i], E[i], "o", ms=8, mfc="none", mec=INK, mew=1.2, zorder=5)
            ax.annotate(f"{gg[i]:.0f}W · {fmt(E[i])}", (gg[i], E[i]), textcoords="offset points",
                        xytext=(-7, 4) if right else (7, 4), ha="right" if right else "left",
                        fontsize=7.8, color=c)
            effs.append(m)
        lo = min(effs[0].min(), effs[1].min()); hi = max(effs[0].max(), effs[1].max())
        ax.set_ylim(lo * 0.35, hi * 5)

    panels(axes2, classes, curves, draw_E)
    for ax in axes2[5:]:
        ax.set_xlabel("GPU power (W)", fontsize=8.5, color=INK2)
    for ax in (axes2[0], axes2[5]):
        ax.set_ylabel("efficiency (tok/J), log", fontsize=8.5, color=INK2)
    fig2.suptitle("Power vs energy efficiency (tok/J) on H200 — prefill vs decode, per use-case class\n"
                  r"tok/J $= T_{phase}(P)\,/\,P$;  rings = per-phase efficiency sweet spots — "
                  "the caps the rack recipes float between", fontsize=12.5, color=INK)
    phase_legend(fig2)
    fig2.text(0.5, 0.012, FOOT, ha="center", fontsize=7.2, color=INK2)
    fig2.tight_layout(rect=(0, 0.065, 1, 0.90))
    fig2.savefig(os.path.join(HERE, "fig_workload_power_tokj.png"), dpi=130, bbox_inches="tight")
    print("wrote fig_workload_power_tokj.png")

    # ---------------- summary csv (per phase) ----------------
    out = []
    for cl in classes:
        cv = curves[MAP[cl["klass"]]]
        gp = g[(g >= cv["pre_rng"][0]) & (g <= cv["pre_rng"][1])]
        gd = g[(g >= cv["dec_rng"][0]) & (g <= cv["dec_rng"][1])]
        Tp, Td = cv["Tpre"](gp), cv["Tdec"](gd)
        Ep, Ed = Tp / gp, Td / gd
        ip, id_ = int(np.argmax(Ep)), int(np.argmax(Ed))
        d_sat = float(gd[int(np.argmax(Td >= 0.995 * Td[-1]))])
        out.append({"klass": cl["klass"], "band": band_of(cl["r"])[0],
                    "ratio_agg": cl["r"], "via_workload": MAP[cl["klass"]],
                    "mapping_caveat": CAVEAT.get(cl["klass"], ""),
                    "pre_sweet_cap_w": round(float(gp[ip])),
                    "pre_tok_per_j_sweet": round(float(Ep[ip]), 3),
                    "pre_tok_s_700w": round(float(Tp[-1]), 1),
                    "dec_sweet_cap_w": round(float(gd[id_])),
                    "dec_tok_per_j_sweet": round(float(Ed[id_]), 3),
                    "dec_tok_s_700w": round(float(Td[-1]), 1),
                    "dec_sat_cap_w": round(d_sat),
                    "pre_fit_R2": round(cv["pre"]["R2"], 3),
                    "dec_fit_R2": round(cv["dec"]["R2"], 3)})
    path = os.path.join(HERE, "workload_power_curves.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        [w.writerow(r) for r in out]
    print(f"wrote {os.path.basename(path)}")


if __name__ == "__main__":
    main()
