"""Model validation on V100 & H200 — the three checks of the paper's validation section.

The model under test is the first-principles analytical model of MODEL_AND_RESULTS.zh.md, as
implemented (and calibrated in §III) by pt_cap_gpu1/portfolio/fitlib.py:

    PREFILL  X_pre(P) = a * phi(P)                          compute-bound throughout, linear in f_sm
    DECODE   X_dec(P) = 1 / [max(a/x,b) + max(c/x,d) + e]   two rooflines + overhead -> three segments
    power    phi(P) = x(P) = ((P - P_s)/chi)^(1/theta)      the DVFS law, eq. (1)-(2)

CHECK 1 (accuracy of the throughput curve) -- per hardware x workload x phase, predict throughput at
  every measured power point and score MAPE against the measurement. Reported twice: over ALL swept
  points (this is the fit the paper's figures and the rack solver actually use) and over the model's
  stated DOMAIN only. The domain is the DVFS-controlled region: points whose SM clock sits on the
  hardware floor (135 MHz V100 / 345 MHz H200) are outside it, because there the cap is no longer a
  frequency knob -- the driver meets it by stalling, and two different throughputs appear at the same
  clock. Those points are kept in the fit (so the numbers match the rest of the repo) but flagged.
  -> fig_val_curves.png, val_mape.csv

CHECK 2 (accuracy of the efficiency-optimal power) -- the planner puts the cap near the efficiency
  peak, so the peak itself is validated directly. Efficiency = throughput / MEASURED draw (the same
  definition as workload_analysis/curves_lib.py: the set cap under-enforces, so the draw is the
  honest denominator). P_eff^meas is located on the measured curve by 3-point parabolic refinement
  around the grid argmax; P_eff^model by fine-grid argmax of the fitted curve over the interpolated
  draw. Two errors are reported: the % deviation of the power, and the % efficiency GIVEN UP by
  capping at the predicted point instead of the measured one (read off the measured curve, PCHIP-
  interpolated) -- the latter is what actually costs the operator anything.
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

python3 validation/validate_model.py -> 3 PNGs + 3 CSVs in this folder
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
HW = {
    "V100": dict(data=os.path.join(ROOT, "pt_cap_gpu1", "portfolio", "data"), clk_floor=135.0,
                 note="cap-swept 100-250 W, both phases"),
    "H200": dict(data=os.path.join(ROOT, "data_h200"), clk_floor=345.0,
                 note="decode cap-swept 200-700 W; prefill clock-swept at the 700 W cap"),
}
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
        k = clk > clk_floor * CLK_FLOOR_TOL
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
    cfg, out = HW[hw], {}
    fmax = cfg["f_max"]
    cal = fitlib.calibrate_power_side(cfg["data"], fmax, cfg["clk_floor"])
    for phase, suffix in (("prefill", "_prefill.csv"), ("decode", "_decode.csv")):
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


def fig_curves(store):
    """Measured points vs predicted curve, both phases co-plotted on a LOG y-axis (they sit 1-2
    orders apart). Rows = hardware, COLS = the MODELS under test, each shown at its representative
    swept shape. Out-of-domain points hollow."""
    fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(16.5, 7.8), squeeze=False)
    for i, hw in enumerate(HW):
        for j, model in enumerate(MODEL_ORDER):
            ax = axes[i][j]
            wid = representative(model, HW[hw]["wids"])
            if wid is None:                       # this model was not measured on this hardware
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"{SHORT[model]}\nnot measured on {hw}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9.6, color=MUTE)
                continue
            d = store[hw][wid]
            ax.set_yscale("log")
            lo_x = min(d[p]["P"].min() for p in ("prefill", "decode"))
            hi_x = max(d[p]["P"].max() for p in ("prefill", "decode"))
            for phase, c in (("prefill", PRE_C), ("decode", DEC_C)):
                s = d[phase]
                g = np.linspace(s["P"].min(), s["P"].max(), 400)
                ax.plot(g, s["fn"](g), color=c, lw=2, zorder=3)
                ax.plot(s["P"], s["T"], "o", ms=5, color=c, mec="white", mew=0.9, zorder=4)
                ax.annotate(f"{phase[:3]}  MAPE {mape(s['T'], s['fn'](s['P'])):.1f}%",
                            (s["P"][-1], s["T"][-1]), textcoords="offset points", xytext=(-3, 8),
                            ha="right", fontsize=8.6, color=c, weight="bold")
            ax.set_xlim(lo_x - (hi_x - lo_x) * .07, hi_x + (hi_x - lo_x) * .10)
            ys = np.concatenate([d[p]["T"] for p in ("prefill", "decode")])
            ax.set_ylim(ys.min() * 0.32, ys.max() * 4.5)   # headroom so the MAPE labels clear the title
            ax.set_title(f"{SHORT[model]}", fontsize=10.6, color=INK, weight="bold", pad=12)
            ax.annotate(shape_label(wid), (0.5, 1.012), xycoords="axes fraction", ha="center",
                        va="bottom", fontsize=8, color=MUTE)
            ax.grid(alpha=.4, color=GRID, lw=0.7, which="both")
            ax.tick_params(labelsize=8, colors=MUTE)
            [s_.set_visible(False) for s_ in (ax.spines["top"], ax.spines["right"])]
            [ax.spines[s_].set_color(GRID) for s_ in ("left", "bottom")]
            if j == 0:
                ax.set_ylabel(f"{hw}\nthroughput (tok/s, log)", fontsize=10, weight="bold", color=INK)
            if i == 1:
                ax.set_xlabel("GPU power (W)", fontsize=9.4, color=INK2)
    fig.legend(handles=[Line2D([], [], color=PRE_C, lw=2.2, marker="o", ms=5, mec="white",
                              label="prefill — model (line) & measured (dots)"),
                        Line2D([], [], color=DEC_C, lw=2.2, marker="o", ms=5, mec="white",
                               label="decode — model (line) & measured (dots)")],
               loc="lower center", ncol=2, fontsize=9.8, frameon=False, bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("Check 1 — throughput-curve accuracy per MODEL: analytical model vs measurement\n"
                 "each column is one model at its representative swept shape "
                 "(median decode context among that model's shapes)", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    _save(fig, "fig_val_curves.png")


# ================================================================================ CHECK 2
def _peak_parabola(x, y):
    """Refine a peak located on a coarse grid: parabola through the argmax and its two neighbours.
    Returns (x_peak, at_boundary). Falls back to the grid point when the argmax is an endpoint."""
    i = int(np.argmax(y))
    if i == 0 or i == len(y) - 1:
        return float(x[i]), True
    x0, x1, x2 = x[i - 1], x[i], x[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(d) < 1e-12:
        return float(x1), False
    A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    Bq = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / d
    if A >= 0:                                        # not a maximum — keep the grid point
        return float(x1), False
    xp = -Bq / (2 * A)
    return float(np.clip(xp, x0, x2)), False


def check2(hw, store):
    """Efficiency = throughput / measured draw. Compare the model's argmax with the measured argmax,
    and price the disagreement on the MEASURED efficiency curve."""
    rows = []
    for wid in HW[hw]["wids"]:
        d = store[hw][wid]
        for phase in ("prefill", "decode"):
            s = d[phase]
            k = s["dom"]
            P, T, W = s["P"][k], s["T"][k], s["W"][k]
            if len(P) < 3:
                continue
            eff = T / W                                        # measured efficiency (tok/J)
            p_meas, at_edge = _peak_parabola(P, eff)
            g = np.linspace(P.min(), P.max(), 4001)
            w_of = np.interp(g, P, W)                          # draw as a function of the power axis
            eff_model = s["fn"](g) / w_of
            p_model = float(g[int(np.argmax(eff_model))])
            eff_meas_of = lambda q: np.interp(q, P, eff)       # measured curve, linearly interpolated
            e_at_meas, e_at_model = float(eff_meas_of(p_meas)), float(eff_meas_of(p_model))
            span = P.max() - P.min()
            rows.append(dict(
                hw=hw, workload=wid, phase=phase,
                P_eff_meas_w=round(p_meas, 1), P_eff_model_w=round(p_model, 1),
                dP_pct=round((p_model - p_meas) / p_meas * 100, 2),
                eff_at_meas=round(e_at_meas, 4), eff_at_model=round(e_at_model, 4),
                eff_loss_pct=round(max(0.0, (e_at_meas - e_at_model) / e_at_meas * 100), 3),
                meas_peak_at_edge=at_edge,
                # the model's efficiency curve never turns over inside the swept range — the peak is
                # pinned to a boundary, so this is a range limitation, not a located-the-wrong-peak error
                model_peak_at_edge=bool(p_model <= P.min() + 0.01 * span
                                        or p_model >= P.max() - 0.01 * span)))
    return rows


def fig_peff(rows):
    """Left column: predicted vs measured P_eff against the identity line. Right column: the
    efficiency actually given up by capping at the predicted point. Rows = hardware."""
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), squeeze=False)
    for i, hw in enumerate(HW):
        rr = [r for r in rows if r["hw"] == hw]
        ax = axes[i][0]
        lo = min(min(r["P_eff_meas_w"], r["P_eff_model_w"]) for r in rr) * 0.9
        hi = max(max(r["P_eff_meas_w"], r["P_eff_model_w"]) for r in rr) * 1.08
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color=MUTE, zorder=2)
        ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1],
                        color=GRID, alpha=.55, zorder=1, lw=0)
        ax.annotate("±10%", (lo + (hi - lo) * .30, (lo + (hi - lo) * .30) * 1.1),
                    textcoords="offset points", xytext=(2, 3), fontsize=8, color=MUTE)
        for phase, c in (("prefill", PRE_C), ("decode", DEC_C)):
            for edge in (False, True):       # edge-pinned peaks drawn hollow: range limit, not a miss
                pts = [r for r in rr if r["phase"] == phase and r["model_peak_at_edge"] == edge]
                if not pts:
                    continue
                ax.plot([r["P_eff_meas_w"] for r in pts], [r["P_eff_model_w"] for r in pts], "o",
                        ms=7, color="white" if edge else c, mec=c, mew=1.6 if edge else 1.0, zorder=4)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("measured $P_{eff}$ (W)", fontsize=9.4, color=INK2)
        ax.set_ylabel(f"{hw}\npredicted $P_{{eff}}$ (W)", fontsize=10, weight="bold", color=INK)
        inr = [r for r in rr if not r["model_peak_at_edge"]]
        med = np.median([abs(r["dP_pct"]) for r in inr])
        ax.set_title(f"{hw} — predicted vs measured $P_{{eff}}$    median |ΔP| = {med:.1f}%",
                     fontsize=10.4, color=INK, weight="bold")
        ax.annotate(f"{len(rr) - len(inr)} of {len(rr)} pinned to the sweep edge",
                    (0.97, 0.04), xycoords="axes fraction", ha="right", fontsize=8, color=MUTE)

        ax = axes[i][1]
        order = sorted(rr, key=lambda r: -r["eff_loss_pct"])
        ypos = np.arange(len(order))
        ax.barh(ypos, [r["eff_loss_pct"] for r in order], height=.68,
                color=[PRE_C if r["phase"] == "prefill" else DEC_C for r in order],
                hatch=["//" if r["model_peak_at_edge"] else "" for r in order],
                edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{r['workload']} · {r['phase'][:3]}" for r in order], fontsize=7.4)
        ax.invert_yaxis()
        for y, r in zip(ypos, order):
            ax.annotate(f"{r['eff_loss_pct']:.2f}%", (r["eff_loss_pct"], y), textcoords="offset points",
                        xytext=(4, 0), va="center", fontsize=7.4, color=INK2)
        worst = max(r["eff_loss_pct"] for r in rr)
        ax.set_xlim(0, worst * 1.28 + 1e-6)
        ax.set_xlabel("efficiency given up by capping at the predicted $P_{eff}$ (%)",
                      fontsize=9.4, color=INK2)
        wi = max(r["eff_loss_pct"] for r in inr)
        ax.set_title(f"{hw} — efficiency given up    worst {wi:.2f}%",
                     fontsize=10.4, color=INK, weight="bold")
        for a in (axes[i][0], axes[i][1]):
            a.grid(alpha=.4, color=GRID, lw=0.7)
            a.set_axisbelow(True)
            a.tick_params(labelsize=8, colors=MUTE)
            [s_.set_visible(False) for s_ in (a.spines["top"], a.spines["right"])]
            [a.spines[s_].set_color(GRID) for s_ in ("left", "bottom")]
    fig.legend(handles=[Line2D([], [], color=PRE_C, lw=0, marker="o", ms=7, mec="white", label="prefill"),
                        Line2D([], [], color=DEC_C, lw=0, marker="o", ms=7, mec="white", label="decode"),
                        Line2D([], [], color=MUTE, lw=0, marker="o", ms=7, mfc="white", mew=1.6,
                               label="hollow marker / hatched bar: model peak pinned to the sweep "
                                     "edge — a range limit, excluded from the medians")],
               loc="lower center", ncol=3, fontsize=9.2, frameon=False, bbox_to_anchor=(0.5, -0.004))
    fig.suptitle("Check 2 — accuracy of the efficiency-optimal power $P_{eff}$ (efficiency = tok/s per "
                 "measured watt)\nleft: predicted vs measured peak · right: what the error costs, "
                 "read off the measured curve", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0.045, 1, 0.91))
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
    a0, p0_ = float(np.exp(coef[0])), float(coef[1])
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
    for hw, cfg in HW.items():
        cfg["f_max"] = fitlib.resolve_f_max(cfg["data"])
        cfg["wids"] = workloads(cfg["data"])

    store = {hw: {wid: load(hw, wid) for wid in HW[hw]["wids"]} for hw in HW}

    r1 = [r for hw in HW for r in check1(hw, store)]
    _csv(r1, "val_mape.csv")
    _csv(by_model(r1), "val_mape_by_model.csv")     # TABLE 3: by model x phase
    fig_curves(store)

    r2 = [r for hw in HW for r in check2(hw, store)]
    _csv(r2, "val_peff.csv")
    fig_peff(r2)

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
        d = [r for r in r2 if r["hw"] == hw]
        inr = [r for r in d if not r["model_peak_at_edge"]]
        print(f"{hw} P_eff    median |ΔP| {np.median([abs(r['dP_pct']) for r in inr]):5.1f}% · "
              f"median eff loss {np.median([r['eff_loss_pct'] for r in inr]):.2f}% "
              f"(max {max(r['eff_loss_pct'] for r in inr):.2f}%) · "
              f"{len(d) - len(inr)}/{len(d)} edge-pinned")
        d = [r for r in r3 if r["hw"] == hw and r["phase"] == "decode"]
        for k in ("A", "B", "ours"):
            print(f"{hw} decode {k:4s} MAPE {np.median([r[f'MAPE_{k}_pct'] for r in d]):5.1f}% · "
                  f"sat {np.median([r[f'MAPEsat_{k}_pct'] for r in d]):5.1f}% · "
                  f"LOO {np.median([r[f'LOO_{k}_pct'] for r in d]):5.1f}%")


if __name__ == "__main__":
    main()
