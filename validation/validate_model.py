"""Model validation on V100 & H200 — the three checks of the paper's validation section.

The model under test is the first-principles analytical model of MODEL_AND_RESULTS.zh.md, as
implemented (and calibrated in §III) by pt_cap_gpu1/portfolio/fitlib.py:

    PREFILL  X_pre(P) = a * phi(P)                          compute-bound throughout, linear in f_sm
    DECODE   X_dec(P) = 1 / [max(a/x,b) + max(c/x,d) + e]   two rooflines + overhead -> three segments
    power    phi(P) = x(P) = ((P - P_s)/chi)^(1/theta)      the DVFS law, eq. (1)-(2)

CHECK 1 (accuracy of the throughput curve) -- per hardware x workload x phase, predict throughput at
  every measured power point in the model's DOMAIN and score MAPE against the measurement. The
  domain is the cap-controlled DVFS region: fitlib.cap_sweep_mask drops clock-floor points (cap
  unreachably low; H200 decode) and governor-stall points (draw well below the cap at a mid-range
  clock; V100's lowest cap on light prefill) at read time -- the same rule the whole repo fits with.
  Two figures: per MODEL (fig_val_curves.png; bottom row = the MOCK RTX 5090, projection only) and
  per PRODUCTION CLASS of Table I (fig_val_curves_class.png; Long-context chat = synthetic anchor).
  The mock row appears in the FIGURES ONLY -- scoring the model on data generated from the model is
  circular, so no 5090 number enters any CSV/median.
  -> fig_val_curves.png, fig_val_curves_class.png, val_mape.csv, val_mape_by_model.csv

CHECK 2 (efficiency-curve accuracy) -- the planner works on tok/J, so the efficiency curve
  (throughput / MEASURED draw, same definition as workload_analysis/curves_lib.py) is validated in
  the same terms as check 1: MAPE of the model efficiency curve against the measured efficiency
  points, drawn on the fig_workload_power_tokj panel layout (3 hardware rows x 7 classes). On the
  measured grid the draw cancels, so the number equals check 1's throughput MAPE -- the check adds
  the efficiency VIEW, not a new metric. The earlier optimum-point (ΔP) comparison was retired:
  efficiency peaks are second-order flat, making the argmax ill-conditioned and operationally
  irrelevant (a mislocated cap near the peak costs ~nothing).
  -> fig_val_peff.png, val_peff.csv

CHECK 3 (comparison against simpler analytical forms) -- two baselines fitted to the SAME data and
  the same power-side (P_s, chi, theta), so only the throughput-side structure differs:
    A  single power law over the whole range, X = a*x^p        the classic DVFS form, no segments
    B  power law under a hard bandwidth ceiling, X = min(a*x^p, X_max)   a clip, no mixed segment
    ours  the two-roofline sum above                           compute -> mixed -> plateau
  Scored by MAPE over the whole range and inside the SATURATION region (the upper half of the
  domain's power range), which is where the three structures disagree. Because ours carries more
  free parameters (5 vs 2 and 3) an in-sample win would be cheap, so leave-one-out CV MAPE is
  reported alongside -- that is the number that decides it.
  -> fig_val_baselines.png, val_baselines.csv

python3 validation/validate_model.py -> 4 PNGs + 4 CSVs in this folder
"""
from __future__ import annotations
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pt_cap_gpu1", "portfolio"))
import fitlib                                                        # noqa: E402
import portfolio                                                     # noqa: E402  the workload table

# ---- hardware under test ------------------------------------------------------------------------
# HW drives the CHECKS (MAPE tables, P_eff, baselines): measured hardware only. FIG_HW adds the
# MOCK RTX 5090 row to the two Check-1 curve FIGURES — synthesized data (data_5090/make_mock_5090.py)
# never enters the metric CSVs, because scoring the model against data generated FROM the model is
# circular; the row exists to show the projected curves alongside the measured cards.
HW = {
    "V100": dict(data=os.path.join(ROOT, "pt_cap_gpu1", "portfolio", "data"), clk_floor=135.0,
                 note="cap-swept 100-250 W, both phases"),
    "H200": dict(data=os.path.join(ROOT, "data_h200"), clk_floor=345.0,
                 note="decode cap-swept 200-700 W; prefill clock-swept at the 700 W cap"),
}
FIG_HW = {**HW,
          "RTX 5090": dict(data=os.path.join(ROOT, "data_5090"), clk_floor=210.0,
                                  note="MOCK: synthesized from the H200 fits x 5090 spec ratios")}
# In a CAP sweep, points whose SM clock sits on the hardware floor are DROPPED at read time: there
# the cap is no longer a frequency knob (the driver meets it by stalling), so two different
# throughputs appear at the same clock and no X=f(phi) model can describe them. V100 loses none;
# H200 decode loses 1-3 per workload at the bottom of the sweep.
# This applies ONLY where the cap is the swept variable. H200 prefill is a LOCKED-CLOCK sweep at a
# fixed 700 W cap: its 345 MHz point is a deliberately set clock (power_limited=False, sm_clk_avg
# tracks clk_set_mhz exactly), not a cap that failed to engage, and must be kept.
CLK_FLOOR_TOL = 1.02
# Qwen3-4B was only ever swept on V100, so it cannot appear in a cross-hardware comparison.
EXCLUDE_MODELS = {"Qwen/Qwen3-4B-Instruct-2507"}
# ---- the MODELS under test (check 1 is reported per model x phase, not per task type) -----------
# The measured workloads are (model, shape) pairs; several shapes share one model. SHAPE = the
# per-workload (prefill seq x batch, decode ctx x batch) actually swept.
WL = {w["id"]: w for w in portfolio.PORTFOLIO}
MODEL_OF = {w["id"]: w["model_id"] for w in portfolio.PORTFOLIO}
SHORT = {"Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B", "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B",
         "microsoft/Phi-3-mini-4k-instruct": "Phi-3-mini-4k (3.8B)",
         "Qwen/Qwen3-4B-Instruct-2507": "Qwen3-4B (2025)",
         "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B"}
MODEL_ORDER = [m for m in ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
                           "microsoft/Phi-3-mini-4k-instruct", "Qwen/Qwen2.5-7B-Instruct"]
               if m not in EXCLUDE_MODELS]                          # by parameter count
# Workloads with a documented measurement caveat, excluded from REPRESENTATIVE selection only (they
# stay in every table): classify-qwen7b is the under-saturated B=8 burst whose cap stops engaging
# above ~180 W — see pt_cap_gpu1/portfolio/DATA_QUALITY.zh.md.
FLAGGED = {"classify-qwen7b"}


def representative(model, wids):
    """The representative workload of one model: among that model's measured shapes (minus any with
    a documented measurement caveat), the one with the MEDIAN decode context, upper median on ties.
    Chosen on workload SHAPE alone — never on how well the model fits it — so the panel cannot be
    accused of showing the best case."""
    c = sorted([w for w in wids if MODEL_OF.get(w) == model and w not in FLAGGED],
               key=lambda w: WL[w]["decode_ctx"])
    if not c:
        c = sorted([w for w in wids if MODEL_OF.get(w) == model],
                   key=lambda w: WL[w]["decode_ctx"])
    return c[len(c) // 2] if c else None


def shape_label(wid):
    w = WL[wid]
    return (f"prefill {w['prefill_seq_len']}×{w['prefill_batch']}  ·  "
            f"decode {w['decode_ctx']}×{w['decode_batch']}")

# ---- palette (repo fleet colors; the 3-form trio validated for CVD separation & contrast) --------
PRE_C, DEC_C = "#1f77b4", "#ff7f0e"                 # phase identity, same as curves_lib.py
FORM_C = {"ours": "#1f77b4", "A": "#cc6600", "B": "#9467bd"}
# distinct dash patterns as well as hue: below the ceiling B coincides with A exactly, and with one
# pattern the hidden curve reads as "missing" rather than "identical"
DASH = {"ours": "-", "A": (0, (6, 2)), "B": (0, (1.6, 1.8))}
INK, INK2, MUTE, GRID, OUT = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#b9b7b0"
FORM_LABEL = {"ours": "ours: two-roofline sum (3 segments)",
              "A": "A: single power law (classic DVFS)",
              "B": "B: power law + hard ceiling"}


# ================================================================================ data
def read(path, clk_floor):
    """(power_axis, throughput, sm_clk, measured_draw, rows). Power axis = the ENFORCED cap when it
    is swept (decode, and V100 prefill); when the cap is fixed and the SM CLOCK is swept instead
    (H200 prefill) it is the measured draw. Same rule as curves_lib._read.

    When the cap is the swept variable, clock-floor points are dropped here (see CLK_FLOOR_TOL) so
    everything downstream — fits, figures, metrics — sees only the DVFS-controlled region. A
    locked-clock sweep is left intact: there the clock is set directly and the cap never acts."""
    rows = [r for r in csv.DictReader(open(path)) if float(r["throughput_tok_s"]) > 0]
    cap = np.array([float(r["cap_w"]) for r in rows])
    thr = np.array([float(r["throughput_tok_s"]) for r in rows])
    clk = np.array([float(r["sm_clk_avg"]) for r in rows])
    pwr = np.array([float(r["power_avg_w"]) for r in rows])
    cap_swept = np.ptp(cap) > 1e-6
    if cap_swept:
        k = fitlib.cap_sweep_mask(cap, clk, pwr, clk_floor)
        cap, thr, clk, pwr = cap[k], thr[k], clk[k], pwr[k]
        rows = [r for r, kk in zip(rows, k) if kk]
    P = cap if cap_swept else pwr
    o = np.argsort(P)
    return P[o], thr[o], clk[o], pwr[o], [rows[i] for i in o]


def workloads(data):
    return sorted({f.rsplit("_", 1)[0] for f in os.listdir(data) if f.endswith("_prefill.csv")
                   and MODEL_OF.get(f.rsplit("_", 1)[0]) not in EXCLUDE_MODELS})


def load(hw, wid):
    """Both phases of one workload: measured arrays, the paper's fit, and the domain mask."""
    cfg, out = FIG_HW[hw], {}
    fmax = cfg["f_max"]
    for phase, suffix in (("prefill", "_prefill.csv"), ("decode", "_decode.csv")):
        cal = fitlib.calibrate_power_side(cfg["data"], fmax, cfg["clk_floor"], phase)
        P, T, F, W, rows = read(os.path.join(cfg["data"], wid + suffix), cfg["clk_floor"])
        B = float(rows[0]["batch"])
        fn, pr = (fitlib.fit_prefill_theory(P, T, F, fmax, cal) if phase == "prefill"
                  else fitlib.fit_decode_theory(P, T, F, B, fmax, cal))
        out[phase] = dict(P=P, T=T, F=F, W=W, B=B, fn=fn, pr=pr,
                          dom=np.ones(len(P), bool))     # everything read is in-domain by construction
    return out


mape = lambda y, yh: float(np.mean(np.abs(np.asarray(yh) - np.asarray(y)) / np.asarray(y)) * 100)


# ================================================================================ CHECK 1
def check1(hw, store):
    rows = []
    for wid in HW[hw]["wids"]:
        d = store[hw][wid]
        for phase in ("prefill", "decode"):
            s = d[phase]
            rows.append(dict(
                hw=hw, model=SHORT.get(MODEL_OF.get(wid), "?"), workload=wid,
                is_representative=(wid == representative(MODEL_OF.get(wid), HW[hw]["wids"])),
                phase=phase, n_points=len(s["P"]),
                P_lo_w=round(float(s["P"].min())), P_hi_w=round(float(s["P"].max())),
                MAPE_pct=round(mape(s["T"], s["fn"](s["P"])), 2),
                R2=round(s["pr"]["R2"], 4)))
    return rows


def by_model(rows):
    """TABLE 3 — MAPE by MODEL and phase. Several swept shapes share one model, so each cell is the
    median over that model's shapes with the min-max range and the shape count beside it."""
    out = []
    for hw in HW:
        for model in MODEL_ORDER:
            for phase in ("prefill", "decode"):
                d = [r for r in rows if r["hw"] == hw and r["phase"] == phase
                     and MODEL_OF.get(r["workload"]) == model]
                if not d:
                    continue
                g = lambda k: [r[k] for r in d]
                out.append(dict(
                    hw=hw, model=SHORT[model], phase=phase, n_shapes=len(d),
                    representative=next((r["workload"] for r in d if r["is_representative"]), ""),
                    MAPE_med=round(float(np.median(g("MAPE_pct"))), 2),
                    MAPE_min=min(g("MAPE_pct")), MAPE_max=max(g("MAPE_pct")),
                    R2_med=round(float(np.median(g("R2"))), 4)))
    return out


def _curve_panel(ax, d, title, subtitle=None, ylab=None, xlab=True):
    """One measured-vs-predicted panel: both phases on a LOG y-axis (they sit 1-2 orders apart),
    fitted line + measured dots, each phase labelled with its MAPE. Shared by the per-model and the
    per-workload figures so the two can never drift apart. `subtitle` is accepted but not drawn
    (paper style: no per-panel fine print; the anchor/shape detail lives in the CSVs)."""
    ax.set_yscale("log")
    lo_x = min(d[p]["P"].min() for p in ("prefill", "decode"))
    hi_x = max(d[p]["P"].max() for p in ("prefill", "decode"))
    for phase, c in (("prefill", PRE_C), ("decode", DEC_C)):
        s = d[phase]
        g = np.linspace(s["P"].min(), s["P"].max(), 400)
        ax.plot(g, s["fn"](g), color=c, lw=2, zorder=3)
        ax.plot(s["P"], s["T"], "o", ms=5, color=c, mec="white", mew=0.9, zorder=4)
        ax.annotate(f"{phase}  MAPE {mape(s['T'], s['fn'](s['P'])):.1f}%",
                    (s["P"][-1], s["T"][-1]), textcoords="offset points", xytext=(-3, 9),
                    ha="right", fontsize=14, color=c, weight="bold")
    ax.set_xlim(lo_x - (hi_x - lo_x) * .07, hi_x + (hi_x - lo_x) * .10)
    ys = np.concatenate([d[p]["T"] for p in ("prefill", "decode")])
    ax.set_ylim(ys.min() * 0.32, ys.max() * 4.5)      # headroom so the MAPE labels clear the title
    ax.set_title(title, fontsize=16, color="black", weight="bold", pad=8)
    ax.grid(alpha=.4, color=GRID, lw=0.7, which="both")
    ax.tick_params(labelsize=14, colors="black")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))   # fewer, bigger x labels — they used to collide
    [s_.set_visible(False) for s_ in (ax.spines["top"], ax.spines["right"])]
    [ax.spines[s_].set_color("black") for s_ in ("left", "bottom")]
    if ylab:
        ax.set_ylabel(ylab, fontsize=16, weight="bold", color="black")
    if xlab:
        ax.set_xlabel("GPU power (W)", fontsize=15, color="black")


def _phase_legend(fig, y=-0.004, ncol=2):
    fig.legend(handles=[Line2D([], [], color=PRE_C, lw=2.2, marker="o", ms=5, mec="white",
                               label="prefill — model (line) & measured (dots)"),
                        Line2D([], [], color=DEC_C, lw=2.2, marker="o", ms=5, mec="white",
                               label="decode — model (line) & measured (dots)")],
               loc="lower center", ncol=ncol, fontsize=16, frameon=False, bbox_to_anchor=(0.5, y))


def fig_curves(store):
    """Rows = hardware, cols = the MODELS under test, each at its representative swept shape."""
    fig, axes = plt.subplots(len(FIG_HW), len(MODEL_ORDER), figsize=(19.5, 13.4), squeeze=False)
    for i, hw in enumerate(FIG_HW):
        for j, model in enumerate(MODEL_ORDER):
            ax = axes[i][j]
            wid = representative(model, FIG_HW[hw]["wids"])
            if wid is None:                       # this model was not measured on this hardware
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"{SHORT[model]}\nnot measured on {hw}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=15, color=MUTE)
                continue
            _curve_panel(ax, store[hw][wid], SHORT[model], shape_label(wid),
                         ylab=f"{hw}\nthroughput (tok/s, log)" if j == 0 else None,
                         xlab=(i == len(FIG_HW) - 1))
    _phase_legend(fig, y=-0.012)
    fig.suptitle("Check 1 — throughput-curve accuracy per MODEL: analytical model vs measurement\n"
                 "each column is one model at its representative swept shape\n"
                 "(median decode context among that model's shapes)\n"
                 "⚠ bottom row RTX 5090 = MOCK data (synthesized, no measurement — "
                 "shown for projection only, excluded from every metric)",
                 fontsize=17.5, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.985))   # tight_layout already reserves the suptitle itself
    _save(fig, "fig_val_curves.png")


def classes():
    """The paper's seven PRODUCTION workload classes (Table I) and the measured curve each is anchored
    to, read from workload_classes.csv — the single source of truth shared with curves_lib.py.
    A '+' in via_workload marks a SYNTHETIC anchor: the geometric mean of the named sweeps, used where
    a class has no sweep of its own."""
    path = os.path.join(ROOT, "workload_analysis", "workload_classes.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return sorted([dict(klass=r["klass"], name=r["name_en"], rho=float(r["ratio_agg"]),
                        via=r["via_workload"].split("+")) for r in rows], key=lambda c: c["rho"])


def synth_entry(hw, parts):
    """A synthetic anchor as a store entry: the sources' measured points rank-paired and averaged
    (throughput geometrically — fitlib.synth_points), then fitted with the same theory model. The
    entry is shaped exactly like a measured one, so _curve_panel draws it identically; the figure
    suptitle carries the mock note."""
    out = {}
    for phase in ("prefill", "decode"):
        cal = fitlib.calibrate_power_side(FIG_HW[hw]["data"], FIG_HW[hw]["f_max"],
                                          FIG_HW[hw]["clk_floor"], phase)
        P, T, F, W = fitlib.synth_points([(d[phase]["P"], d[phase]["T"], d[phase]["F"],
                                           d[phase]["W"]) for d in parts])
        B = float(np.mean([d[phase]["B"] for d in parts]))
        fn, pr = (fitlib.fit_prefill_theory(P, T, F, FIG_HW[hw]["f_max"], cal) if phase == "prefill"
                  else fitlib.fit_decode_theory(P, T, F, B, FIG_HW[hw]["f_max"], cal))
        out[phase] = dict(P=P, T=T, F=F, W=W, B=B, fn=fn, pr=pr, dom=np.ones(len(P), bool))
    return out


def fig_curves_class(store):
    """The seven production classes of Table I, ordered by prefill-to-decode ratio. Rows = hardware,
    cols = class, so a column compares the two GPUs on the same class."""
    cl = classes()
    fig, axes = plt.subplots(len(FIG_HW), len(cl), figsize=(4.45 * len(cl), 14.0), squeeze=False)
    for i, hw in enumerate(FIG_HW):
        for j, c in enumerate(cl):
            ax = axes[i][j]
            rho = f"{c['rho']:.1f}:1" if c["rho"] >= 1 else f"{c['rho']:.2f}:1"
            title, ylab = f"{c['name']}   {rho}", (f"{hw}\nthroughput (tok/s, log)" if j == 0 else None)
            if any(v not in store[hw] for v in c["via"]):
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"{c['name']}\nanchor not measured on {hw}",
                        transform=ax.transAxes, ha="center", va="center", fontsize=15, color=MUTE)
            elif len(c["via"]) > 1:                    # synthetic anchor, drawn like any other
                _curve_panel(ax, synth_entry(hw, [store[hw][v] for v in c["via"]]), title,
                             "anchor mean(" + ", ".join(c["via"]) + ")",
                             ylab=ylab, xlab=(i == len(FIG_HW) - 1))
            else:
                _curve_panel(ax, store[hw][c["via"][0]], title,
                             f"anchor {c['via'][0]}  ·  {shape_label(c['via'][0])}",
                             ylab=ylab, xlab=(i == len(FIG_HW) - 1))
    _phase_legend(fig, y=-0.012)
    synth = [c for c in cl if len(c["via"]) > 1]
    note = ("\n⚠ " + ", ".join(c["name"] for c in synth) + " = MOCK anchor (per-power mean of "
            "the " + " & ".join(sorted({v for c in synth for v in c["via"]}))
            + " measurements, fitted like any workload)") if synth else ""
    note += ("\n⚠ bottom row RTX 5090 = MOCK data (synthesized, no measurement — "
             "shown for projection only, excluded from every metric)")
    fig.suptitle("Check 1 — throughput-curve accuracy per PRODUCTION WORKLOAD CLASS: "
                 "analytical model vs measurement\n"
                 "the seven classes of Table I, ordered by prefill-to-decode ratio; each takes the "
                 "measured sweep whose context scale is closest to its own" + note,
                 fontsize=18, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.985))   # tight_layout already reserves the suptitle itself
    _save(fig, "fig_val_curves_class.png")


# ================================================================================ CHECK 2
def check2(hw, store):
    """Efficiency-curve accuracy, scored by MAPE — the same criterion as check 1, on the tok/J axis
    the planner actually optimizes. Efficiency = throughput / MEASURED draw; the model efficiency
    curve is the fitted throughput over the interpolated measured draw. NOTE: at the measured grid
    the draw cancels, so the efficiency MAPE is numerically IDENTICAL to the throughput MAPE of
    check 1 — this check adds the efficiency VIEW (peak location and flatness are visible), not a
    new number. The earlier optimum-point comparison (ΔP between predicted and measured argmax) was
    retired: efficiency peaks are second-order flat, so the argmax location is ill-conditioned
    (a <2% curve error can move it ~80 W) while costing almost nothing — the wrong quantity to
    validate."""
    rows = []
    for wid in HW[hw]["wids"]:
        d = store[hw][wid]
        for phase in ("prefill", "decode"):
            s = d[phase]
            P, T, W = s["P"], s["T"], s["W"]
            if len(P) < 3:
                continue
            rows.append(dict(hw=hw, workload=wid, phase=phase, n_points=len(P),
                             eff_MAPE_pct=round(mape(T / W, s["fn"](P) / W), 2)))
    return rows


ratio_str = lambda x: f"{x:.1f}:1" if x >= 1 else f"{x:.2f}:1"
fmt_eff = lambda v: (f"{v/1e3:.1f}k" if v >= 1e3 else f"{v:.0f}" if v >= 100
                     else f"{v:.1f}" if v >= 10 else f"{v:.2f}")


def fig_peff(rows, store):
    """fig_workload_power_tokj.png panels, scored like check 1: per-CLASS efficiency curves (both
    phases co-plotted, log y, class title = name + P:D in band color), one row per hardware
    (V100 / H200 / RTX 5090-mock), each phase annotated with its efficiency-curve MAPE. No optimum
    markers — the flat peaks make the argmax the wrong quantity to compare (see check2)."""
    cl = classes()
    fig, axes = plt.subplots(len(FIG_HW), len(cl), figsize=(4.45 * len(cl), 14.0), squeeze=False)
    for i, hw in enumerate(FIG_HW):
        for j, c in enumerate(cl):
            ax = axes[i][j]
            ax.set_yscale("log")
            if any(v not in store[hw] for v in c["via"]):
                ax.set_axis_off()
                continue
            d = (synth_entry(hw, [store[hw][v] for v in c["via"]]) if len(c["via"]) > 1
                 else store[hw][c["via"][0]])
            lo_x = min(d[p]["P"].min() for p in ("prefill", "decode"))
            hi_x = max(d[p]["P"].max() for p in ("prefill", "decode"))
            mid = (lo_x + hi_x) / 2
            ymin, ymax = np.inf, -np.inf
            for phase, pc in (("prefill", PRE_C), ("decode", DEC_C)):
                s_ = d[phase]
                P, T, W = s_["P"], s_["T"], s_["W"]
                if len(P) < 3:
                    continue
                eff = T / W                                    # measured efficiency (tok/J on draw)
                g = np.linspace(P.min(), P.max(), 1501)
                o = np.argsort(P)
                w_of = np.interp(g, P[o], W[o])
                eff_m = s_["fn"](g) / w_of                     # model efficiency
                ax.plot(g, eff_m, color=pc, lw=2, zorder=3)
                ax.plot(P, eff, "o", ms=4.5, color=pc, mec="white", mew=0.9, zorder=4)
                # sweet spot marked as in the tokj figure (display only — not a scored quantity)
                k = int(np.argmax(eff_m))
                p_opt, e_opt = float(g[k]), float(eff_m[k])
                right = p_opt > mid
                ax.plot(p_opt, e_opt, "o", ms=8, mfc="none", mec="black", mew=1.2, zorder=5)
                ax.annotate(f"{p_opt:.0f}W", (p_opt, e_opt),
                            textcoords="offset points", xytext=(-8, 6) if right else (8, 6),
                            ha="right" if right else "left", fontsize=13, color=pc)
                e = mape(eff, s_["fn"](P) / W)
                ax.annotate(f"{phase}  MAPE {e:.1f}%", (P[-1], float(eff[-1])),
                            textcoords="offset points", xytext=(-3, -19), ha="right",
                            fontsize=14, color=pc, weight="bold")
                ymin, ymax = min(ymin, float(eff.min())), max(ymax, float(eff.max()))
            ax.set_ylim(ymin * 0.35, ymax * 5)
            ax.set_xlim(lo_x - (hi_x - lo_x) * .08, hi_x + (hi_x - lo_x) * .03)
            ax.set_title(f"{c['name']}   {ratio_str(c['rho'])}", fontsize=16,
                         color="black", weight="bold", pad=8)   # same style as _curve_panel
            ax.grid(alpha=.45, color=GRID, lw=0.7, which="both")
            ax.tick_params(labelsize=14, colors="black")
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))   # fewer, bigger x labels
            [sp.set_visible(False) for sp in (ax.spines["top"], ax.spines["right"])]
            [ax.spines[sp].set_color("black") for sp in ("left", "bottom")]
            if j == 0:
                ax.set_ylabel(f"{hw}\nefficiency (tok/J, log)", fontsize=16, weight="bold",
                              color="black")
            if i == len(FIG_HW) - 1:
                ax.set_xlabel("GPU power (W)", fontsize=15, color="black")
    fig.legend(handles=[Line2D([], [], color=PRE_C, lw=2.2, marker="o", ms=5, mec="white",
                               label="prefill — model (line) & measured (dots)"),
                        Line2D([], [], color=DEC_C, lw=2.2, marker="o", ms=5, mec="white",
                               label="decode — model (line) & measured (dots)"),
                        Line2D([], [], color="black", lw=0, marker="o", ms=8, mfc="none", mew=1.2,
                               label="ring = efficiency sweet spot (display only, not scored)")],
               loc="lower center", ncol=3, fontsize=16, frameon=False, bbox_to_anchor=(0.5, -0.006))
    med = {hw: np.median([r["eff_MAPE_pct"] for r in rows if r["hw"] == hw]) for hw in HW}
    fig.suptitle("Check 2 — efficiency-curve accuracy per production workload class "
                 "(efficiency = tok/s per measured watt)   ·   "
                 + "   ·   ".join(f"{hw} median MAPE = {med[hw]:.1f}%" for hw in HW)
                 + "\n⚠ bottom row RTX 5090 = MOCK data (synthesized, no measurement — projection "
                 "only, excluded from every metric)"
                 "\n⚠ Long-context chat = MOCK anchor "
                 "(per-power mean of the chat-phi3 & summarize-qwen7b measurements)",
                 fontsize=18, color="black")
    fig.tight_layout(rect=(0, 0.05, 1, 0.985))   # tight_layout already reserves the suptitle itself
    _save(fig, "fig_val_peff.png")


# ================================================================================ CHECK 3
def fit_forms(P, T, F, B, fmax, phase):
    """The three throughput-side structures on a SHARED power side (P_s, chi, theta), so the only
    thing that differs is how throughput depends on the relative clock x.

    All three are fitted under the SAME loss — least squares on the RELATIVE residual
    (pred/meas - 1) — because they are scored by MAPE. This matters: fitlib fits the deployed decode
    model on the absolute per-token time tau, which weights the slow low-power points far more
    heavily than MAPE does, so scoring that fit with MAPE would hand the baselines a free win on
    loss mismatch rather than on structure. Check 1 keeps the deployed fit (it validates what the
    paper actually uses); check 3 equalizes the loss so only structure is being compared."""
    x = np.clip(F / fmax, 1e-3, 1.0)
    Ps, chi, th, _ = fitlib.fit_power_side(P, x)
    xf = lambda Q: np.clip(np.maximum((np.asarray(Q, float) - Ps) / chi, 1e-9) ** (1.0 / th), 1e-3, 1.0)

    # A — a single power law over the whole range (the classic DVFS form): X = a * x^p
    coef, *_ = np.linalg.lstsq(np.vstack([np.ones_like(x), np.log(x)]).T, np.log(T), rcond=None)
    a0, p0_ = float(np.exp(coef[0])), float(np.clip(coef[1], 1e-3, 12.0))
    rA = least_squares(lambda q: (q[0] * x ** q[1]) / T - 1.0, [a0, p0_],
                       bounds=([1e-9, 1e-3], [np.inf, 12.0]), method="trf", max_nfev=20000)
    aA, pA = (float(v) for v in rA.x)
    fA = lambda Q: aA * xf(Q) ** pA

    # B — the same power law clipped by a hard bandwidth ceiling: X = min(a * x^p, X_max)
    rB = least_squares(lambda q: np.minimum(q[0] * x ** q[1], q[2]) / T - 1.0,
                       [aA, pA, float(T.max()) * 1.05],
                       bounds=([1e-9, 1e-3, float(T.max()) * 0.5], [np.inf, 12.0, np.inf]),
                       method="trf", max_nfev=20000)
    aB, pB, xmB = (float(v) for v in rB.x)
    fB = lambda Q: np.minimum(aB * xf(Q) ** pB, xmB)

    # ours — the paper's structure, same loss as A and B
    if phase == "prefill":                                  # X = a * phi, a single scale
        aO = float(np.sum(x / T) / np.sum((x / T) ** 2))
        fO = lambda Q: aO * xf(Q)
    else:                                                   # tau = max(a/x,b) + max(c/x,d) + e
        tau = 1.0 / T
        tau_plat = float(np.min(tau))
        o = np.argsort(x)
        comp = float(tau[o[0]] * x[o[0]])
        seed = [.5 * comp, .35 * tau_plat, .5 * comp, .35 * tau_plat, .15 * tau_plat]
        rt = lambda q, xx: np.maximum(q[0] / xx, q[1]) + np.maximum(q[2] / xx, q[3]) + q[4]
        rO = least_squares(lambda q: 1.0 / (rt(q, x) * T) - 1.0, seed,
                           bounds=(1e-12, np.inf), method="trf", max_nfev=20000)
        fO = lambda Q: 1.0 / rt(rO.x, xf(Q))
    return {"A": fA, "B": fB, "ours": fO}


def loo_mape(P, T, F, B, fmax, phase):
    """Leave-one-out CV MAPE — the fairness control, since 'ours' carries the most free parameters
    (5 vs 2 and 3) and would otherwise win in-sample on flexibility alone."""
    acc = {k: [] for k in ("A", "B", "ours")}
    for i in range(len(P)):
        m = np.ones(len(P), bool); m[i] = False
        try:
            fits = fit_forms(P[m], T[m], F[m], B, fmax, phase)
        except Exception:
            continue
        for k, fn in fits.items():
            acc[k].append(abs(float(fn(P[i:i + 1])[0]) - T[i]) / T[i] * 100)
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}


def check3(hw, store):
    """Score the three forms over the whole domain and inside the saturation region (upper half of
    the domain's power range) — that upper half is where the three structures disagree."""
    rows, curves = [], {}
    fmax = HW[hw]["f_max"]
    for wid in HW[hw]["wids"]:
        d = store[hw][wid]
        for phase in ("prefill", "decode"):
            s = d[phase]
            k = s["dom"]
            P, T, F = s["P"][k], s["T"][k], s["F"][k]
            if len(P) < 5:
                continue
            fits = fit_forms(P, T, F, s["B"], fmax, phase)
            mid = P.min() + 0.5 * (P.max() - P.min())
            sat = P >= mid
            loo = loo_mape(P, T, F, s["B"], fmax, phase)
            rec = dict(hw=hw, workload=wid, phase=phase, n_points=len(P),
                       sat_from_w=round(float(mid)), n_sat=int(sat.sum()))
            for key in ("A", "B", "ours"):
                rec[f"MAPE_{key}_pct"] = round(mape(T, fits[key](P)), 2)
                rec[f"MAPEsat_{key}_pct"] = round(mape(T[sat], fits[key](P[sat])), 2)
                rec[f"LOO_{key}_pct"] = round(loo[key], 2)
            rows.append(rec)
            curves[(wid, phase)] = (P, T, fits)
    return rows, curves


def fig_baselines(rows, curves):
    """Left: the three forms on the decode workload where they disagree most. Middle/right: MAPE per
    workload over the whole domain and inside the saturation region. Rows = hardware."""
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.8), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    for i, hw in enumerate(HW):
        dec = [r for r in rows if r["hw"] == hw and r["phase"] == "decode"]
        pick = max(dec, key=lambda r: max(r["MAPEsat_A_pct"], r["MAPEsat_B_pct"]) - r["MAPEsat_ours_pct"])
        P, T, fits = curves[(hw, pick["workload"], "decode")]

        ax = axes[i][0]
        g = np.linspace(P.min(), P.max(), 600)
        for key in ("A", "B", "ours"):
            ax.plot(g, fits[key](g), color=FORM_C[key], lw=2.6 if key == "ours" else 1.8,
                    ls=DASH[key], zorder=4 if key == "ours" else 3)
        ax.plot(P, T, "o", ms=6, color=INK, mec="white", mew=1.0, zorder=5)
        ax.axvspan(P.min() + 0.5 * (P.max() - P.min()), P.max(), color=GRID, alpha=.55, zorder=0, lw=0)
        ax.annotate("saturation region\n(scored separately)",
                    (P.min() + 0.5 * (P.max() - P.min()), T.min()), textcoords="offset points",
                    xytext=(6, 2), fontsize=7.8, color=MUTE)
        ax.set_title(f"{hw} · {pick['workload']} decode — where the forms diverge",
                     fontsize=10.2, color=INK, weight="bold")
        ax.set_ylabel(f"{hw}\nthroughput (tok/s)", fontsize=10, weight="bold", color=INK)
        ax.set_xlabel("GPU power (W)", fontsize=9.4, color=INK2)

        for col, (mkey, ttl) in enumerate([("MAPE", "whole domain"), ("MAPEsat", "saturation region")], 1):
            ax = axes[i][col]
            dd = sorted(dec, key=lambda r: r["workload"])
            y = np.arange(len(dd)); h = 0.26
            for off, key in zip((-h, 0, h), ("A", "B", "ours")):
                ax.barh(y + off, [r[f"{mkey}_{key}_pct"] for r in dd], height=h * 0.92,
                        color=FORM_C[key], zorder=3)
            ax.set_yticks(y); ax.set_yticklabels([r["workload"] for r in dd], fontsize=7.6)
            ax.invert_yaxis()
            xlab = f"decode MAPE — {ttl} (%)"
            if col == 1:            # the fairness control belongs next to the in-sample numbers
                lo_ = {k: np.median([r[f"LOO_{k}_pct"] for r in dd]) for k in ("A", "B", "ours")}
                xlab += (f"\nleave-one-out CV median:  A {lo_['A']:.1f} · B {lo_['B']:.1f} · "
                         f"ours {lo_['ours']:.1f} %   (ours fits 5 free parameters vs 2 and 3)")
            ax.set_xlabel(xlab, fontsize=9.4, color=INK2)
            med = {k: np.median([r[f"{mkey}_{k}_pct"] for r in dd]) for k in ("A", "B", "ours")}
            ax.set_title(f"{hw} — {ttl}   median  A {med['A']:.1f} · B {med['B']:.1f} · "
                         f"ours {med['ours']:.1f} %", fontsize=9.6, color=INK, weight="bold")
        for a in axes[i]:
            a.grid(alpha=.4, color=GRID, lw=0.7)
            a.set_axisbelow(True)
            a.tick_params(labelsize=8, colors=MUTE)
            [s_.set_visible(False) for s_ in (a.spines["top"], a.spines["right"])]
            [a.spines[s_].set_color(GRID) for s_ in ("left", "bottom")]
    fig.legend(handles=[Line2D([], [], color=FORM_C[k], lw=2.4, ls=DASH[k], label=FORM_LABEL[k])
                        for k in ("ours", "A", "B")]
                       + [Line2D([], [], color=INK, lw=0, marker="o", ms=6, mec="white",
                                 label="measured")],
               loc="lower center", ncol=4, fontsize=9.2, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Check 3 — the analytical model against two simpler forms fitted to the same data\n"
                 "all three share the calibrated power side P(x); only the throughput-side structure "
                 "differs", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    _save(fig, "fig_val_baselines.png")


# ================================================================================ driver
def _save(fig, name):
    p = os.path.join(HERE, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def _csv(rows, name):
    p = os.path.join(HERE, name)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        [w.writerow(r) for r in rows]
    print("wrote", name)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    for hw, cfg in FIG_HW.items():
        cfg["f_max"] = fitlib.resolve_f_max(cfg["data"])
        cfg["wids"] = workloads(cfg["data"])

    store = {hw: {wid: load(hw, wid) for wid in FIG_HW[hw]["wids"]} for hw in FIG_HW}

    r1 = [r for hw in HW for r in check1(hw, store)]
    _csv(r1, "val_mape.csv")
    _csv(by_model(r1), "val_mape_by_model.csv")     # TABLE 3: by model x phase
    fig_curves(store)
    fig_curves_class(store)

    r2 = [r for hw in HW for r in check2(hw, store)]
    _csv(r2, "val_peff.csv")
    fig_peff(r2, store)

    r3, curves = [], {}
    for hw in HW:
        rr, cc = check3(hw, store)
        r3 += rr
        for (wid, phase), v in cc.items():
            curves[(hw, wid, phase)] = v
    _csv(r3, "val_baselines.csv")
    fig_baselines(r3, curves)

    print("\n--- summary ------------------------------------------------------------")
    for hw in HW:
        for phase in ("prefill", "decode"):
            e = [r["MAPE_pct"] for r in r1 if r["hw"] == hw and r["phase"] == phase]
            print(f"{hw} {phase:8s} MAPE median {np.median(e):5.2f}%  "
                  f"[{min(e):5.2f}, {max(e):5.2f}]  n={len(e)}")
        d = [r["eff_MAPE_pct"] for r in r2 if r["hw"] == hw]
        print(f"{hw} eff-curve MAPE median {np.median(d):5.2f}%  [{min(d):5.2f}, {max(d):5.2f}]  "
              f"(identical to throughput MAPE by construction — the draw cancels on the grid)")
        d = [r for r in r3 if r["hw"] == hw and r["phase"] == "decode"]
        for k in ("A", "B", "ours"):
            print(f"{hw} decode {k:4s} MAPE {np.median([r[f'MAPE_{k}_pct'] for r in d]):5.1f}% · "
                  f"sat {np.median([r[f'MAPEsat_{k}_pct'] for r in d]):5.1f}% · "
                  f"LOO {np.median([r[f'LOO_{k}_pct'] for r in d]):5.1f}%")


if __name__ == "__main__":
    main()
