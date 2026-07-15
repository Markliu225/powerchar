"""Per-use-case-class power curves on V100: power vs throughput & power vs tok/J, PER PHASE.

For EACH of the 10 use-case classes in workload_ratios.csv (the InstructGPT/Dolly/trace
taxonomy, see README + REFERENCES.zh.md), plot the PREFILL curve and the DECODE curve of the
V100-measured workload it maps onto (the verified correspondence of
rack_power_capping/v100/WORKLOADS.zh.md §2, same MAP as fig_workload_pd.png) — separately,
matching the repo's disaggregated-serving methodology: the rack solver caps each phase on its
own curve, so the per-phase curves (not a blended one) are the planning primitives.

    throughput:  T_pre(P), T_dec(P)   fitlib unified-model fits, x = enforced cap in [100,250] W
    efficiency:  T_phase / MEASURED draw (power_avg_w)   tok/J computed from throughput & measured
                 power — NOT the set cap, NOT the CSV tok_per_joule column (energy-counter, unreliable);
                 decode's cap under-enforces at low cap so the two differ.  ring = per-phase sweet spot

The two phases differ by 1-2 orders of magnitude -> log throughput/efficiency axis.
Dots = the raw measured grid (portfolio v3 dataset). The class's aggregate P:D ratio labels
the panel (class identity); it does not enter any curve.

python3 plot_power_curves.py -> fig_workload_power_throughput.png
                                fig_workload_power_tokj.png
                                workload_power_curves.csv
"""
from __future__ import annotations
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PORT = os.path.join(ROOT, "pt_cap_gpu1", "portfolio")
sys.path.insert(0, PORT)
import fitlib                                   # noqa: E402

DATA = os.path.join(PORT, "data")
F_MAX = fitlib.resolve_f_max(DATA)
CAP_LO, CAP_HI = 100.0, 250.0                   # measured cap range — never extrapolate

NAME = {"Generation 创作生成": "Generation", "General QA 常识问答": "General QA",
        "Brainstorming 头脑风暴": "Brainstorming", "Open QA 开放问答": "Open QA",
        "Classification 分类": "Classification", "Summarization 摘要": "Summarization",
        "Extract 信息抽取": "Extract", "Chat 多轮对话": "Chat (dialogue)",
        "Closed QA 闭卷问答": "Closed QA", "Code 代码补全": "Code (completion)"}
# class -> measured powerchar workload whose V100 curves it maps onto: the same class-level
# correspondence as fig_workload_pd.png's right-column mapping (WORKLOADS.zh.md §2), resolved
# to each class's anchor member (first-listed in WORKLOAD_CLASSES) where a class has several
MAP = {"Generation 创作生成": "longform-phi3", "General QA 常识问答": "longform-phi3",
       "Brainstorming 头脑风暴": "longform-phi3", "Open QA 开放问答": "longform-phi3",
       "Classification 分类": "translate-qwen3b", "Summarization 摘要": "summarize-qwen7b",
       "Extract 信息抽取": "classify-qwen7b", "Chat 多轮对话": "chat-phi3",
       "Closed QA 闭卷问答": "rag-phi3", "Code 代码补全": "code-phi3"}
MODEL_SHORT = {"longform-phi3": "Phi-3-mini", "chat-phi3": "Phi-3-mini",
               "translate-qwen3b": "Qwen2.5-3B", "rag-phi3": "Phi-3-mini",
               "code-phi3": "Phi-3-mini", "summarize-qwen7b": "Qwen2.5-7B",
               "classify-qwen7b": "Qwen2.5-7B"}
# P:D bands, same thresholds & colors as fig_workload_pd.png (color follows the band entity)
BANDS = [("decode-heavy", 0.0, 0.5, "#d62728"),
         ("balanced", 0.5, 2.0, "#7f7f7f"),
         ("prefill-heavy", 2.0, np.inf, "#1f77b4")]
PRE_C, DEC_C = "#1f77b4", "#ff7f0e"             # phase colors, same as fig_workloads.png fleets
INK, INK2, MUTE, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

# classes whose mapping carries an accounting/scale caveat in WORKLOADS.zh.md §2 (starred)
CAVEAT = {"Chat 多轮对话": "trace accounting (full history re-prefilled per turn); serving shape with KV reuse ~1:2",
          "Summarization 摘要": "Dolly short-doc 2.3:1 on 32k production-scale curves",
          "Extract 信息抽取": "Dolly 3.1:1 on production-scale curves (class shape 100:1); near-flat decode, fit R2<0 - read the dots",
          "Closed QA 闭卷问答": "Dolly 6.2:1 on ~4k RAG-scale curves (class shape 10:1)",
          "Code 代码补全": "2023 completion-trace 73.5:1; the rack doc plans with a conservative 10:1"}

FOOT = ("curves: fitlib unified-model fits of pt_cap_gpu1/portfolio/data (V100-DGXS-32GB, v3; caps 100-250 W)  ·  "
        "dots: raw measured grid  ·  P:D = the class's aggregate ratio from workload_ratios.csv "
        "(labels the class; it enters no curve)\n"
        "* = the class-to-workload mapping carries an accounting/scale caveat (Chat: trace accounting — serving "
        "shape with KV reuse is ~1:2;  Summarization / Extract / Closed QA / Code: Dolly- or 2023-trace ratios "
        "on production-scale curves)\n"
        "full list: rack_power_capping/v100/WORKLOADS.zh.md §2 + the mapping_caveat column of "
        "workload_power_curves.csv  ·  annotations follow the fitted curve — where dots visibly deviate "
        "(Extract: near-flat decode, fit R²<0) read the dots")

ratio_str = lambda x: f"{x:.1f}:1" if x >= 1 else f"1:{1/x:.0f}"
band_of = lambda r: next(b for b in BANDS if b[1] <= r < b[2])
fmt = lambda v: (f"{v/1e3:.2f}k" if v >= 1e3 else f"{v:.0f}" if v >= 100
                 else f"{v:.1f}" if v >= 10 else f"{v:.2f}")


def _read(path):
    """Return (power_axis, throughput, sm_clk, rows). Power axis = the ENFORCED cap when the cap is
    swept (V100, decode); when cap is held fixed and the SM CLOCK is swept instead (H200 prefill,
    v-next methodology), fall back to the MEASURED draw power_avg_w as the power axis."""
    rows = [r for r in csv.DictReader(open(path)) if float(r["throughput_tok_s"]) > 0]
    cap = np.array([float(r["cap_w"]) for r in rows])
    thr = np.array([float(r["throughput_tok_s"]) for r in rows])
    clk = np.array([float(r["sm_clk_avg"]) for r in rows])
    pwr = np.array([float(r["power_avg_w"]) for r in rows])   # MEASURED draw (energy-window avg)
    P = cap if np.ptp(cap) > 1e-6 else pwr
    return P, thr, clk, pwr, rows


def load_curves(wid):
    """fitlib fits + raw per-phase points for one measured workload (same recipe as the rack solver).
    Prefill and decode may sweep different power axes (see _read), so their raw points and valid
    power ranges are kept SEPARATE — never assume a shared cap grid."""
    Pp, Tp, Fp, Wp, prow = _read(os.path.join(DATA, f"{wid}_prefill.csv"))
    Pd, Td, Fd, Wd, drow = _read(os.path.join(DATA, f"{wid}_decode.csv"))
    B = float(drow[0]["batch"])
    preT, pre = fitlib.fit_prefill_unified(Pp, Tp, Fp, F_MAX)
    decT, dec = fitlib.fit_decode_additive(Pd, Td, Fd, B, F_MAX)

    def _pwr_of(x, w):                                          # MEASURED draw as a fn of the fit's power axis
        o = np.argsort(x)                                      # (prefill: axis IS measured draw -> identity;
        return lambda q: np.interp(np.asarray(q, float), x[o], w[o])   # decode: cap axis -> map cap to draw)
    return dict(Tpre=lambda q: np.maximum(preT(np.clip(np.asarray(q, float), CAP_LO, CAP_HI)), 1e-9),
                Tdec=lambda q: np.maximum(decT(np.clip(np.asarray(q, float), CAP_LO, CAP_HI)), 1e-9),
                pre_x=Pp, pre_y=Tp, dec_x=Pd, dec_y=Td,           # raw points, each on its own axis
                pre_pwr=Wp, dec_pwr=Wd,                           # MEASURED draw (power_avg_w) per raw point
                pre_pwr_of=_pwr_of(Pp, Wp), dec_pwr_of=_pwr_of(Pd, Wd),   # tok/J denominator = measured draw
                pre_rng=(float(Pp.min()), float(Pp.max())),       # measured power span per phase
                dec_rng=(float(Pd.min()), float(Pd.max())),
                pre=pre, dec=dec, ctx=drow[0]["ctx"], b_dec=drow[0]["batch"])


def panels(fig_axes, classes, curves, draw):
    """Common panel chrome; `draw(ax, cl, cv)` adds the metric-specific content (log y)."""
    for ax, cl in zip(fig_axes, classes):
        cv = curves[MAP[cl["klass"]]]
        bname, _, _, bcol = band_of(cl["r"])
        ax.set_yscale("log")
        draw(ax, cl, cv)
        star = "*" if cl["klass"] in CAVEAT else ""
        ax.set_title(f"{NAME[cl['klass']]}   P:D {ratio_str(cl['r'])}{star}",
                     fontsize=10.5, color=INK, pad=22)
        ax.text(0.5, 1.10, f"via {MAP[cl['klass']]} · {MODEL_SHORT[MAP[cl['klass']]]} · "
                f"dec {cv['ctx']}×{cv['b_dec']}", transform=ax.transAxes,
                ha="center", fontsize=7.3, color=INK2)
        ax.text(0.03, 0.96, bname, transform=ax.transAxes, ha="left", va="top",
                fontsize=7.8, color=bcol, weight="bold")
        ax.set_xlim(92, 258)
        ax.set_xticks([100, 150, 200, 250])
        ax.grid(alpha=.45, color=GRID, lw=0.7, which="both")
        ax.tick_params(labelsize=8, colors=MUTE)
        [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]
        [ax.spines[s].set_color(GRID) for s in ("left", "bottom")]


def phase_legend(fig):
    fig.legend(handles=[Line2D([], [], color=PRE_C, lw=2, label="prefill"),
                        Line2D([], [], color=DEC_C, lw=2, label="decode")],
               loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=9, frameon=False)


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "workload_ratios.csv"))))
    classes = sorted([dict(klass=r["klass"], r=float(r["ratio_agg"])) for r in rows],
                     key=lambda c: c["r"])      # decode-heavy -> prefill-heavy
    curves = {wid: load_curves(wid) for wid in sorted(set(MAP.values()))}
    g = np.linspace(CAP_LO, CAP_HI, 1501)

    # ---------------- figure 1: power vs throughput (per phase, log y) ----------------
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.4))
    axes = axes.ravel()

    def draw_T(ax, cl, cv):
        for rng, fn, x, y, c in ((cv["pre_rng"], cv["Tpre"], cv["pre_x"], cv["pre_y"], PRE_C),
                                 (cv["dec_rng"], cv["Tdec"], cv["dec_x"], cv["dec_y"], DEC_C)):
            gg = g[(g >= rng[0]) & (g <= rng[1])]          # draw the fit only over measured power
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
        ax.set_xlabel("power cap (W)", fontsize=8.5, color=INK2)
    for ax in (axes[0], axes[5]):
        ax.set_ylabel("throughput (tok/s, log)", fontsize=8.5, color=INK2)
    fig.suptitle("Power vs throughput on V100 — prefill vs decode, per use-case class\n"
                 "each class's mapped workload measured PER PHASE (disaggregated serving, "
                 "x = enforced cap); log axis — the phases sit 1-2 orders of magnitude apart",
                 fontsize=12.5, color=INK)
    phase_legend(fig)
    fig.text(0.5, 0.012, FOOT, ha="center", fontsize=7.2, color=INK2)
    fig.tight_layout(rect=(0, 0.075, 1, 0.90))
    fig.savefig(os.path.join(HERE, "fig_workload_power_throughput.png"), dpi=130,
                bbox_inches="tight")
    print("wrote fig_workload_power_throughput.png")

    # ---------------- figure 2: power vs tok/J (per phase, log y) ----------------
    fig2, axes2 = plt.subplots(2, 5, figsize=(16, 7.4))
    axes2 = axes2.ravel()

    def draw_E(ax, cl, cv):
        effs = []
        for rng, fn, x, y, w, wof, c in (
                (cv["pre_rng"], cv["Tpre"], cv["pre_x"], cv["pre_y"], cv["pre_pwr"], cv["pre_pwr_of"], PRE_C),
                (cv["dec_rng"], cv["Tdec"], cv["dec_x"], cv["dec_y"], cv["dec_pwr"], cv["dec_pwr_of"], DEC_C)):
            gg = g[(g >= rng[0]) & (g <= rng[1])]
            E = fn(gg) / wof(gg)                    # tok/J = fitted T / MEASURED draw (not the set cap, not CSV)
            m = y / w                               # measured tok/J = raw throughput / raw measured power_avg_w
            ax.plot(gg, E, color=c, lw=2, zorder=3)
            ax.plot(x, m, "o", ms=4.5, color=c, mec="white", mew=0.9, zorder=4)
            i = int(np.argmax(E))
            right = gg[i] > (rng[0] + rng[1]) / 2      # ring in right half -> label to its left
            ax.plot(gg[i], E[i], "o", ms=8, mfc="none", mec=INK, mew=1.2, zorder=5)
            ax.annotate(f"{gg[i]:.0f}W · {fmt(E[i])}", (gg[i], E[i]), textcoords="offset points",
                        xytext=(-7, 4) if right else (7, 4), ha="right" if right else "left",
                        fontsize=7.8, color=c)
            effs.append(m)
        lo = min(effs[0].min(), effs[1].min()); hi = max(effs[0].max(), effs[1].max())
        ax.set_ylim(lo * 0.35, hi * 5)          # extra headroom: ring labels clear the band tag

    panels(axes2, classes, curves, draw_E)
    for ax in axes2[5:]:
        ax.set_xlabel("power cap (W)", fontsize=8.5, color=INK2)
    for ax in (axes2[0], axes2[5]):
        ax.set_ylabel("efficiency (tok/J on measured draw, log)", fontsize=8.5, color=INK2)
    fig2.suptitle("Power vs energy efficiency (tok/J) on V100 — prefill vs decode, per use-case class\n"
                  r"tok/J $= T_{phase}\,/\,$MEASURED draw (power_avg_w, not the set cap, not the CSV column);"
                  "  rings = per-phase sweet spots",
                  fontsize=12.5, color=INK)
    phase_legend(fig2)
    fig2.text(0.5, 0.012, FOOT, ha="center", fontsize=7.2, color=INK2)
    fig2.tight_layout(rect=(0, 0.075, 1, 0.90))
    fig2.savefig(os.path.join(HERE, "fig_workload_power_tokj.png"), dpi=130,
                 bbox_inches="tight")
    print("wrote fig_workload_power_tokj.png")

    # ---------------- summary csv (per phase) ----------------
    out = []
    for cl in classes:
        cv = curves[MAP[cl["klass"]]]
        gp = g[(g >= cv["pre_rng"][0]) & (g <= cv["pre_rng"][1])]
        gd = g[(g >= cv["dec_rng"][0]) & (g <= cv["dec_rng"][1])]
        Tp, Td = cv["Tpre"](gp), cv["Tdec"](gd)
        Ep, Ed = Tp / cv["pre_pwr_of"](gp), Td / cv["dec_pwr_of"](gd)   # tok/J on MEASURED draw, not set cap
        ip, id_ = int(np.argmax(Ep)), int(np.argmax(Ed))
        d_sat = float(gd[int(np.argmax(Td >= 0.995 * Td[-1]))])  # decode saturation cap
        out.append({"klass": cl["klass"], "band": band_of(cl["r"])[0],
                    "ratio_agg": cl["r"], "via_workload": MAP[cl["klass"]],
                    "mapping_caveat": CAVEAT.get(cl["klass"], ""),
                    "pre_sweet_cap_w": round(float(gp[ip])),
                    "pre_tok_per_j_sweet": round(float(Ep[ip]), 3),
                    "pre_tok_s_250w": round(float(Tp[-1]), 1),
                    "dec_sweet_cap_w": round(float(gd[id_])),
                    "dec_tok_per_j_sweet": round(float(Ed[id_]), 3),
                    "dec_tok_s_250w": round(float(Td[-1]), 1),
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
