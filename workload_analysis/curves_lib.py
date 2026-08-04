"""Per-production-class power curves: power vs throughput & power vs tok/J, PER PHASE (log y).

For the 7 production workload classes of workload_classes.csv (the paper's II-C trace-based
taxonomy: Reasoning / Assistant API / Multimodal / Chat / Long-context chat / Agentic / Code,
P:D 0.83 ... 110.7 from ServeGen, DynamoLLM/Azure'24 and Mooncake traces), each maps onto a
measured workload by nearest decode-context scale (MAP below). Per class, the PREFILL and DECODE
curves of that anchor are co-plotted on ONE panel; they sit 1-2 orders of magnitude apart, so the
y-axis is LOG. The rack solver caps each phase on its own curve, so these are the primitives.

    model (MODEL_AND_RESULTS.zh.md): prefill is compute-bound throughout → X_pre ∝ φ(P) (linear in
    f_sm); decode per-token time is a SUM of two rooflines + overhead, τ=max(a/x,b)+max(c/x,d)+e →
    X_dec=1/τ, giving three segments (compute-bound rise → attention saturates → bandwidth plateau).
    throughput:  T_pre(P), T_dec(P)   fitlib.fit_{prefill,decode}_theory over the measured power range
    efficiency:  T_phase / MEASURED draw (power_avg_w)   tok/J from throughput & measured power —
                 NOT the set cap, NOT the CSV tok_per_joule column (energy-counter, unreliable);
                 decode's cap under-enforces at low cap so the two differ.  ring = per-phase sweet spot

Dots = raw measured grid. The class's aggregate P:D ratio labels identity; it enters no curve.

SHARED LIBRARY: this module holds the taxonomy (NAME/MAP/CAVEAT/BANDS), the fit loader
(load_curves), and the figure builder (build_power_figs). It is imported — not run — by the
per-hardware wrappers v100/plot_power_curves.py and h200/plot_power_curves.py (which set DATA /
CAP_LO / CAP_HI / F_MAX and call build_power_figs), and by solve_rack_capping.py (taxonomy).
One implementation, so the two hardwares cannot drift.
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
CLK_FLOOR = 135.0                               # lowest hardware SM clock (H200 wrapper overrides)

# The taxonomy = the paper's II-C production workload classes (trace-measured P:D, eq. (1)
# rho-bar = sum Lp / sum Ld): Reasoning 0.83 (ServeGen) · Assistant API 7.7 (ServeGen) ·
# Multimodal 9.4 (ServeGen) · Chat 15.5 (DynamoLLM/Azure'24) · Long-context chat 35.1 (Mooncake) ·
# Agentic tool-use 47.2 (Mooncake) · Code completion 110.7 (DynamoLLM/Azure'24).
# Data lives in workload_classes.csv (klass, Lp_mean, Ld_mean, ratio_agg, source, via_workload).
NAME = {"推理": "Reasoning", "助手API": "Assistant API", "多模态图文": "Multimodal",
        "对话": "Chat (dialogue)", "长上下文对话": "Long-context chat",
        "Agentic工具调用": "Agentic tool-use", "代码补全": "Code completion"}
# class -> measured workload whose curves it maps onto. ONE anchor per class, none shared, so every
# class rests on its own sweep. Anchors are matched on request SHAPE (prompt-heavy vs decode-heavy,
# and the context scale each phase actually sees), not on decode context alone. Mirrored in
# workload_classes.csv (via_workload column) — keep the two in step.
# All anchors exist on BOTH V100 and H200. fastchat-qwen15b anchors no class; it stays in the
# validation set only.
# A tuple value is a BLEND: the class has no sweep of its own and takes the GEOMETRIC mean of the
# named anchors' curves. Geometric, not arithmetic: the two decode ceilings sit two orders of
# magnitude apart (256 vs 32768 ctx), so an arithmetic mean would land within 0.4% of the faster
# anchor and would not be a blend at all; on the log axis these curves live on, the geometric mean
# is the midpoint.
MAP = {"推理": "longform-phi3",            # ~2.8k eff. ctx, decode-dominant -> dec 4096x8 anchor
       "助手API": "translate-qwen3b",      # ~0.7k total -> 512x64
       "多模态图文": "rag-phi3",           # ~1.3k eff. ctx -> dec 1024x32 (vision tokens priced as text)
       "对话": "chat-phi3",                # short accumulated history, high concurrency -> dec 256x64
       "长上下文对话": ("chat-phi3", "summarize-qwen7b"),   # blend: rho-bar 35.1 sits between the
                                           # Chat (15.5) and Agentic (47.2) anchors
       "Agentic工具调用": "summarize-qwen7b",  # ~8.7k per-step re-prefill -> 32k (conservative)
       "代码补全": "code-phi3"}            # its own Azure-trace basis, dec 2048x16
# P:D bands, same thresholds & colors as fig_workload_pd.png (color follows the band entity)
BANDS = [("decode-heavy", 0.0, 0.5, "#d62728"),
         ("balanced", 0.5, 2.0, "#7f7f7f"),
         ("prefill-heavy", 2.0, np.inf, "#1f77b4")]
PRE_C, DEC_C = "#1f77b4", "#ff7f0e"             # phase colors (repo fleet palette)
INK, INK2, MUTE, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

# classes whose anchor mapping carries a scale/accounting caveat (starred in the figures)
CAVEAT = {"推理": "anchor longform-phi3 dec 4096x8; qwen3think-4b (8192x8) is the closer shape but has no H200 data",
          "助手API": "translate curves (512x64) stand in for its ~0.7k programmatic requests",
          "多模态图文": "text-only curves - vision tokens priced as text prefill (no multimodal measurement); "
                        "dec 1024x32 vs ~1.3k effective context, the closest anchor in the set",
          "对话": "anchor chat-phi3 dec 256x64 vs ~1.7k accumulated history - anchor context SHORTER than "
                  "the class, so the decode ceiling is OPTIMISTIC (T proportional to 1/C)",
          "长上下文对话": "SYNTHETIC: geometric mean of the chat-phi3 and summarize-qwen7b curves, no sweep of "
                          "its own. Effective context ~sqrt(256*32768)=2.9k against a real ~12.2k, so the decode "
                          "ceiling is OPTIMISTIC by ~4x; summarize-qwen7b alone would be ~2.7x CONSERVATIVE",
          "Agentic工具调用": "32k summarize curves vs ~8.7k per-step context - conservative; "
                             "full-history re-prefill accounting"}

FOOT = ("lines = model fits, dots = measured  ·  title = class P:D, colored by band "
        "(red decode-heavy · gray balanced · blue prefill-heavy)  ·  "
        "P:D from production traces (ServeGen NSDI'26 · DynamoLLM-Azure'24 · Mooncake FAST'25)  ·  tok/J on measured draw")

ratio_str = lambda x: f"{x:.1f}:1" if x >= 1 else f"{x:.2f}:1"
band_of = lambda r: next(b for b in BANDS if b[1] <= r < b[2])
fmt = lambda v: (f"{v/1e3:.1f}k" if v >= 1e3 else f"{v:.0f}" if v >= 100
                 else f"{v:.1f}" if v >= 10 else f"{v:.2f}")


def _read(path):
    """(power_axis, throughput, sm_clk, measured_draw, rows). Power axis = ENFORCED cap when swept
    (decode); when cap is fixed and the SM CLOCK is swept (H200 prefill), the MEASURED draw
    power_avg_w. Measured draw is always returned separately (the tok/J denominator).

    A CAP sweep keeps only the points where the cap is actually the operating constraint —
    fitlib.cap_sweep_mask drops clock-floor points (cap unreachably low; H200 decode) and
    governor-stall points (draw well below the cap at a mid-range clock; V100's lowest cap on light
    prefill). A LOCKED-CLOCK sweep keeps every point: its lowest clock is a deliberately set
    operating point, not a failure to enforce."""
    rows = [r for r in csv.DictReader(open(path)) if float(r["throughput_tok_s"]) > 0]
    cap = np.array([float(r["cap_w"]) for r in rows])
    thr = np.array([float(r["throughput_tok_s"]) for r in rows])
    clk = np.array([float(r["sm_clk_avg"]) for r in rows])
    pwr = np.array([float(r["power_avg_w"]) for r in rows])   # MEASURED draw (energy-window avg)
    cap_swept = np.ptp(cap) > 1e-6
    if cap_swept:
        k = fitlib.cap_sweep_mask(cap, clk, pwr, CLK_FLOOR)
        cap, thr, clk, pwr = cap[k], thr[k], clk[k], pwr[k]
        rows = [r for r, kk in zip(rows, k) if kk]
    return (cap if cap_swept else pwr), thr, clk, pwr, rows


def load_curves(wid):
    """fitlib fits + raw per-phase points for one measured workload (same recipe as the rack solver)."""
    Pp, Tp, Fp, Wp, prow = _read(os.path.join(DATA, f"{wid}_prefill.csv"))
    Pd, Td, Fd, Wd, drow = _read(os.path.join(DATA, f"{wid}_decode.csv"))
    B = float(drow[0]["batch"])
    # UPDATED first-principles theory (MODEL_AND_RESULTS.zh.md): prefill LINEAR in phi;
    # decode = sum of two rooflines + overhead (three segments). See fitlib.fit_*_theory.
    # The DVFS side (P_stat, gamma) is the per-GPU, per-PHASE calibration of eq. (3), shared by
    # every workload (per phase because the effective exponent differs between them — see fitlib).
    preT, pre = fitlib.fit_prefill_theory(Pp, Tp, Fp, F_MAX,
                                          fitlib.calibrate_power_side(DATA, F_MAX, CLK_FLOOR, "prefill"))
    decT, dec = fitlib.fit_decode_theory(Pd, Td, Fd, B, F_MAX,
                                         fitlib.calibrate_power_side(DATA, F_MAX, CLK_FLOOR, "decode"))

    def _pwr_of(x, w):                                          # MEASURED draw as a fn of the fit's power axis
        o = np.argsort(x)                                      # (prefill: axis IS measured draw -> identity;
        return lambda q: np.interp(np.asarray(q, float), x[o], w[o])   # decode: cap axis -> map cap to draw)
    return dict(Tpre=lambda q: np.maximum(preT(np.clip(np.asarray(q, float), CAP_LO, CAP_HI)), 1e-9),
                Tdec=lambda q: np.maximum(decT(np.clip(np.asarray(q, float), CAP_LO, CAP_HI)), 1e-9),
                pre_x=Pp, pre_y=Tp, dec_x=Pd, dec_y=Td, pre_F=Fp, dec_F=Fd,
                pre_pwr=Wp, dec_pwr=Wd, pre_pwr_of=_pwr_of(Pp, Wp), dec_pwr_of=_pwr_of(Pd, Wd),
                pre_rng=(float(Pp.min()), float(Pp.max())), dec_rng=(float(Pd.min()), float(Pd.max())),
                pre=pre, dec=dec, ctx=drow[0]["ctx"], b_dec=drow[0]["batch"])


def synth_curves(parts):
    """A SYNTHETIC (mock) anchor for a class with no sweep of its own: the sources' MEASURED points
    are rank-paired along the power axis and averaged into one synthetic sweep (throughput
    geometrically — fitlib.synth_points), which is then fitted with the SAME theory model as any
    measured workload. The panel is indistinguishable from a measured one (dots + fit); the figure
    title carries the mock note."""
    Pp, Tp, Fp, Wp = fitlib.synth_points([(p["pre_x"], p["pre_y"], p["pre_F"], p["pre_pwr"])
                                          for p in parts])
    Pd, Td, Fd, Wd = fitlib.synth_points([(p["dec_x"], p["dec_y"], p["dec_F"], p["dec_pwr"])
                                          for p in parts])
    preT, pre = fitlib.fit_prefill_theory(Pp, Tp, Fp, F_MAX,
                                          fitlib.calibrate_power_side(DATA, F_MAX, CLK_FLOOR, "prefill"))
    decT, dec = fitlib.fit_decode_theory(Pd, Td, Fd, float(parts[0]["b_dec"]), F_MAX,
                                         fitlib.calibrate_power_side(DATA, F_MAX, CLK_FLOOR, "decode"))

    def _pwr_of(x, w):
        o = np.argsort(x)
        return lambda q: np.interp(np.asarray(q, float), x[o], w[o])
    return dict(Tpre=lambda q: np.maximum(preT(np.clip(np.asarray(q, float), CAP_LO, CAP_HI)), 1e-9),
                Tdec=lambda q: np.maximum(decT(np.clip(np.asarray(q, float), CAP_LO, CAP_HI)), 1e-9),
                pre_x=Pp, pre_y=Tp, dec_x=Pd, dec_y=Td, pre_F=Fp, dec_F=Fd,
                pre_pwr=Wp, dec_pwr=Wd, pre_pwr_of=_pwr_of(Pp, Wp), dec_pwr_of=_pwr_of(Pd, Wd),
                pre_rng=(float(Pp.min()), float(Pp.max())), dec_rng=(float(Pd.min()), float(Pd.max())),
                pre=pre, dec=dec, ctx="synthetic", b_dec=parts[0]["b_dec"])


def resolve(entry, cache):
    """MAP entry -> curve bundle. A str is a measured anchor, a tuple is a synthetic one built from
    the named measured sweeps (rank-paired point average, then fitted like a measurement)."""
    if isinstance(entry, tuple):
        return synth_curves([resolve(e, cache) for e in entry])
    if entry not in cache:
        cache[entry] = load_curves(entry)
    return cache[entry]


def _phase_series(cv, phase, metric, g):
    """(gg, curve_y, dot_x, dot_y, color) for one phase & metric ('T' throughput | 'E' tok/J)."""
    if phase == "pre":
        rng, fn, x, y, w, wof, c = cv["pre_rng"], cv["Tpre"], cv["pre_x"], cv["pre_y"], cv["pre_pwr"], cv["pre_pwr_of"], PRE_C
    else:
        rng, fn, x, y, w, wof, c = cv["dec_rng"], cv["Tdec"], cv["dec_x"], cv["dec_y"], cv["dec_pwr"], cv["dec_pwr_of"], DEC_C
    gg = g[(g >= rng[0]) & (g <= rng[1])]
    if metric == "T":
        return gg, fn(gg), x, y, c
    return gg, fn(gg) / wof(gg), x, y / w, c        # tok/J = fitted T / MEASURED draw; dots = raw T / raw draw


def _panel(ax, cl, cv, metric, g):
    """One class panel: BOTH phases co-plotted on a LOG y-axis (they sit 1-2 orders apart)."""
    ax.set_yscale("log")
    mid = (CAP_LO + CAP_HI) / 2
    dys = []
    for phase in ("pre", "dec"):
        gg, curve, dx, dy, c = _phase_series(cv, phase, metric, g)
        ax.plot(gg, curve, color=c, lw=2, zorder=3)
        ax.plot(dx, dy, "o", ms=4.5, color=c, mec="white", mew=0.9, zorder=4)
        if metric == "E":
            i = int(np.argmax(curve)); right = gg[i] > mid
            ax.plot(gg[i], curve[i], "o", ms=8, mfc="none", mec=INK, mew=1.2, zorder=5)
            ax.annotate(f"{gg[i]:.0f}W · {fmt(curve[i])}", (gg[i], curve[i]), textcoords="offset points",
                        xytext=(-7, 4) if right else (7, 4), ha="right" if right else "left",
                        fontsize=7.6, color=c)
        else:
            ax.annotate(fmt(curve[-1]), (gg[-1], curve[-1]), textcoords="offset points",
                        xytext=(-2, 6), ha="right", fontsize=7.8, color=c)
        dys.append(dy)
    lo = min(float(dys[0].min()), float(dys[1].min())); hi = max(float(dys[0].max()), float(dys[1].max()))
    ax.set_ylim(lo * 0.35, hi * (5 if metric == "E" else 3.2))
    bcol = band_of(cl["r"])[3]                       # band conveyed by title color, no extra text
    ax.set_title(f"{NAME[cl['klass']]}   {ratio_str(cl['r'])}", fontsize=11, color=bcol,
                 weight="bold", pad=8)
    ax.set_xlim(CAP_LO * 0.92, CAP_HI * 1.03)
    ax.grid(alpha=.45, color=GRID, lw=0.7, which="both")
    ax.tick_params(labelsize=8, colors=MUTE)
    [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]
    [ax.spines[s].set_color(GRID) for s in ("left", "bottom")]


def build_power_figs(ratios_path, out_dir, hw, xlab):
    """Per-use-case-class panels (both phases co-plotted, LOG y) for throughput and tok/J, + the
    per-class CSV. Reads DATA/F_MAX/CAP_* module globals (H200 overrides them), so one implementation
    serves both hardwares."""
    have = lambda e: all(os.path.exists(os.path.join(DATA, f"{w}_prefill.csv"))
                         for w in (e if isinstance(e, tuple) else (e,)))
    rows = list(csv.DictReader(open(ratios_path, encoding="utf-8")))
    classes = sorted([dict(klass=r["klass"], r=float(r["ratio_agg"])) for r in rows
                      if have(MAP[r["klass"]])],
                     key=lambda c: c["r"])       # decode-heavy -> prefill-heavy
    cache = {}
    curves = {MAP[c["klass"]]: resolve(MAP[c["klass"]], cache) for c in classes}
    g = np.linspace(CAP_LO, CAP_HI, 1501)
    n = len(classes); ncol = 4; nrow = int(np.ceil(n / ncol))   # 7 classes -> balanced 4+3
    # classes on a synthetic anchor are flagged in the FIGURE TITLE only — the panel itself is
    # drawn exactly like a measured one (that is the point of the synthetic sweep)
    synth = [c["klass"] for c in classes if isinstance(MAP[c["klass"]], tuple)]
    note = ("\n⚠ MOCK anchor: " + ", ".join(NAME[k] for k in synth) + " has no sweep of its own — "
            "its points are the per-power mean of the "
            + " & ".join(sorted({w for k in synth for w in MAP[k]}))
            + " measurements, fitted like any workload") if synth else ""

    for metric, fname, ylab, ttl in [
        ("T", "fig_workload_power_throughput.png", "throughput (tok/s, log)",
         f"Power vs throughput on {hw} — prefill & decode per production workload class" + note),
        ("E", "fig_workload_power_tokj.png", "efficiency (tok/J, log)",
         f"Power vs energy efficiency (tok/J) on {hw} — rings mark each phase's sweet spot" + note)]:
        fig, axes = plt.subplots(nrow, ncol, figsize=(14.5, 3.9 * nrow), squeeze=False)
        axes = axes.ravel()
        for j, cl in enumerate(classes):
            _panel(axes[j], cl, curves[MAP[cl["klass"]]], metric, g)
        for ax in axes[n:]:
            ax.set_visible(False)
        for j in range(n):                           # x-label on each column's LOWEST visible panel
            if j + ncol >= n:
                axes[j].set_xlabel(xlab, fontsize=9, color=INK2)
        for k in range(0, n, ncol):
            axes[k].set_ylabel(ylab, fontsize=9, color=INK2)
        fig.legend(handles=[Line2D([], [], color=PRE_C, lw=2.2, label="prefill"),
                            Line2D([], [], color=DEC_C, lw=2.2, label="decode")],
                   loc="lower right", bbox_to_anchor=(0.97, 0.10), fontsize=10.5, frameon=False)
        fig.suptitle(ttl, fontsize=13, color=INK)
        fig.text(0.5, 0.008, FOOT, ha="center", fontsize=7.2, color=INK2)
        fig.tight_layout(rect=(0, 0.035, 1, 0.925 if note else 0.945))
        fig.savefig(os.path.join(out_dir, fname), dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("wrote", os.path.join(out_dir, fname))

    _write_csv(classes, curves, out_dir, g)


def _write_csv(classes, curves, out_dir, g):
    """Per-class summary (sweet caps, tok/J on measured draw, plateau throughput, fit R²)."""
    out = []
    for cl in classes:
        cv = curves[MAP[cl["klass"]]]
        gp = g[(g >= cv["pre_rng"][0]) & (g <= cv["pre_rng"][1])]
        gd = g[(g >= cv["dec_rng"][0]) & (g <= cv["dec_rng"][1])]
        Tp, Td = cv["Tpre"](gp), cv["Tdec"](gd)
        Ep, Ed = Tp / cv["pre_pwr_of"](gp), Td / cv["dec_pwr_of"](gd)   # tok/J on MEASURED draw
        ip, id_ = int(np.argmax(Ep)), int(np.argmax(Ed))
        d_sat = float(gd[int(np.argmax(Td >= 0.995 * Td[-1]))])
        e = MAP[cl["klass"]]
        out.append({"klass": cl["klass"], "band": band_of(cl["r"])[0], "ratio_agg": cl["r"],
                    "via_workload": "+".join(e) if isinstance(e, tuple) else e,
                    "mapping_caveat": CAVEAT.get(cl["klass"], ""),
                    "pre_sweet_cap_w": round(float(gp[ip])), "pre_tok_per_j_sweet": round(float(Ep[ip]), 3),
                    "pre_tok_s_caphi": round(float(Tp[-1]), 1),
                    "dec_sweet_cap_w": round(float(gd[id_])), "dec_tok_per_j_sweet": round(float(Ed[id_]), 3),
                    "dec_tok_s_caphi": round(float(Td[-1]), 1), "dec_sat_cap_w": round(d_sat),
                    "pre_fit_R2": round(cv["pre"]["R2"], 3), "dec_fit_R2": round(cv["dec"]["R2"], 3)})
    path = os.path.join(out_dir, "workload_power_curves.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); [w.writerow(r) for r in out]
    print(f"wrote {os.path.basename(path)}")


if __name__ == "__main__":
    raise SystemExit("curves_lib.py is a shared library — run v100/plot_power_curves.py or "
                     "h200/plot_power_curves.py instead.")
