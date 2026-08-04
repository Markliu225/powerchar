"""Shared fitting library -- THE canonical implementation of the unified explicit P<->T model.

One law for both phases (see ../prefill_model_theory.md and ../decode_model_theory.md):

  per-token time  t(x) = T_mem + C*x^-p        x = f_sm / f_max
  power           P(x) = P_s + chi*x^theta
  compose         x(P) = ((P-P_s)/chi)^(1/theta), clamped <= 1

  PREFILL (T_mem -> 0):  Throughput(P) = T_fmax * x(P)^p          single concave power law
  DECODE  (T_mem floor): Throughput(P) = B / (T_mem + Cc*(x(P)^-p - 1))   three stages + plateau

UPDATED first-principles theory (MODEL_AND_RESULTS.zh.md), used by the power-vs-throughput /
power-vs-tok/J figures — fit_prefill_theory / fit_decode_theory at the bottom of this file:
  PREFILL:  X_pre(P) = a * phi(P)                     linear in f_sm (compute-bound throughout)
  DECODE :  X_dec(P) = 1 / [max(a/x,b) + max(c/x,d) + e]   sum of two rooflines + overhead
The legacy fits above are kept for the rack solver and the old-vs-new comparison plots.
All plot scripts import from here so the figures can never drift apart.
"""
from __future__ import annotations
import csv
import glob
import json
import os
import numpy as np
from scipy.optimize import least_squares


# ------------------------------------------------------------------ metadata
def resolve_f_max(data_dir: str) -> float:
    """Max SM clock for x=f/f_max from the run's meta.json; loud V100 fallback."""
    try:
        v = float(json.load(open(os.path.join(data_dir, "meta.json")))["f_max_mhz"])
        print(f"F_MAX = {v:.0f} MHz  (from {os.path.join(data_dir, 'meta.json')})")
        return v
    except Exception as e:
        print(f"[WARN] no usable meta.json in {data_dir} ({type(e).__name__}); "
              f"falling back to V100 F_MAX=1530 MHz.\n"
              f"       If this data is NOT from a V100, the clock-space fits will be skewed!")
        return 1530.0


# ------------------------------------------------------------------ metrics
def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - yhat) ** 2) / ss if ss > 0 else float("nan")


def rel_rmse(y, yhat):
    """RMSE as % of the mean level (fairer than R^2 on near-flat curves)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)) / np.mean(y) * 100)


# ------------------------------------------------------------------ power side (shared)
# The DVFS law of the paper, eq. (3)-(4):
#     P(f) = P_stat + chi * (f/f_max)^gamma          phi(P) = ((P - P_stat)/chi)^(1/gamma)
# clipped to [phi_min, 1], phi_min = the lowest hardware clock over f_max.
#
# Which parameters are per GPU and which are per workload follows the paper's own taxonomy
# (Sec. III-A): P_stat is a HARDWARE parameter ("comes from the GPU specification") and gamma is an
# EFFICIENCY parameter "calibrated once for each GPU architecture", while the dynamic term
# P_dyn = alpha*C_sw*V^2*f carries the activity factor alpha, which is what actually changes between
# workloads. So P_stat and gamma are calibrated ONCE per GPU over every workload and phase, and only
# chi (the alpha-dependent dynamic span) is fitted per series.
#
# chi is deliberately NOT fixed to (P_max - P_stat) as eq. (3) writes it: that form asserts the GPU
# draws its full TDP at f_max, which contradicts the premise of the paper itself (actual draw rarely
# reaches TDP) and, measured here, costs an order of magnitude in accuracy.
#
# The fit is done in the direction the function is USED — minimising the error of phi(P) against the
# measured clock, not of P(f) against the measured power. Fitting P(f) forward looks excellent
# (residuals of a few watts) yet inverts badly near the low end, where it used to drive P_s onto its
# search bound and collapse phi at the lowest measured point.
_CAL_CACHE: dict = {}


def calibrate_power_side(data_dir: str, f_max: float, clk_floor: float = 0.0) -> dict:
    """Per-GPU calibration of (P_stat, gamma), pooled over every workload and phase in data_dir.

    Cap-swept series drop points whose SM clock sits on the hardware floor: there the cap can no
    longer set the frequency and the driver meets it by stalling, so the point is not on the DVFS
    curve at all. A locked-clock sweep (cap fixed, clock stepped) keeps every point."""
    key = (os.path.abspath(data_dir), float(f_max), float(clk_floor))
    if key in _CAL_CACHE:
        return _CAL_CACHE[key]

    series = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*_prefill.csv"))
                       + glob.glob(os.path.join(data_dir, "*_decode.csv"))):
        rows = [r for r in csv.DictReader(open(path)) if float(r["throughput_tok_s"]) > 0]
        if not rows:
            continue
        cap = np.array([float(r["cap_w"]) for r in rows])
        clk = np.array([float(r["sm_clk_avg"]) for r in rows])
        pwr = np.array([float(r["power_avg_w"]) for r in rows])
        cap_swept = np.ptp(cap) > 1e-6
        if cap_swept:
            k = clk > clk_floor * 1.02
            cap, clk, pwr = cap[k], clk[k], pwr[k]
        if len(clk) < 3:
            continue
        series.append((cap if cap_swept else pwr, np.clip(clk / f_max, 1e-3, 1.0)))
    if not series:
        raise RuntimeError(f"no usable power sweeps in {data_dir}")

    lo = min(float(P.min()) for P, _ in series)
    chi_of = lambda P, x, Ps, g: float(np.sum(np.maximum(P - Ps, 1e-9) * x ** g) / np.sum(x ** (2 * g)))

    def resid(q):
        Ps, g = q
        return np.concatenate([_phi(P, Ps, max(chi_of(P, x, Ps, g), 1e-6), g, 1e-3) - x
                               for P, x in series])

    r = least_squares(resid, [0.5 * lo, 2.5], bounds=([0.0, 1.0], [0.95 * lo, 6.0]), max_nfev=20000)
    phi_min = 1e-3
    try:                                        # phi_min = lowest hardware clock / f_max, eq. (4)
        phi_min = float(json.load(open(os.path.join(data_dir, "meta.json")))["sm_clk_min_mhz"]) / f_max
    except Exception:
        pass
    cal = dict(P_stat=float(r.x[0]), gamma=float(r.x[1]), phi_min=phi_min,
               n_series=len(series), n_points=int(sum(len(x) for _, x in series)))
    print(f"POWER-SIDE CALIBRATION {os.path.basename(os.path.normpath(data_dir))}: "
          f"P_stat={cal['P_stat']:.1f} W  gamma={cal['gamma']:.3f}  phi_min={phi_min:.3f}  "
          f"({cal['n_series']} series, {cal['n_points']} points)")
    _CAL_CACHE[key] = cal
    return cal


# ------------------------------------------------------------------ synthetic anchors
def synth_points(series):
    """Average several measured sweeps into ONE synthetic sweep for a class with no sweep of its
    own. Points are paired by RANK along the power axis (the sweeps share a grid layout, not exact
    values); power, clock and draw take the arithmetic mean, throughput the GEOMETRIC mean — the
    sources sit orders of magnitude apart, and an arithmetic mean of throughputs would collapse
    onto the faster source. The result is fitted downstream exactly like a measured workload.

    series: list of (P, T, F, W) tuples. Returns (P, T, F, W) of the synthetic sweep."""
    n = min(len(s[0]) for s in series)
    srt = []
    for P, T, F, W in series:
        o = np.argsort(np.asarray(P, float))
        idx = o[np.round(np.linspace(0, len(o) - 1, n)).astype(int)]
        srt.append([np.asarray(a, float)[idx] for a in (P, T, F, W)])
    Pm = np.mean([s[0] for s in srt], axis=0)
    Tg = np.exp(np.mean([np.log(np.maximum(s[1], 1e-12)) for s in srt], axis=0))
    Fm = np.mean([s[2] for s in srt], axis=0)
    Wm = np.mean([s[3] for s in srt], axis=0)
    return Pm, Tg, Fm, Wm


def _phi(Q, Ps, chi, th, phi_min):
    Q = np.asarray(Q, float)
    return np.clip(np.maximum((Q - Ps) / chi, 1e-12) ** (1.0 / th), phi_min, 1.0)


def fit_power_side(P, x, cal=None):
    """Return (P_s, chi, theta, railed). With a calibration, P_s and theta are the per-GPU constants
    and only chi is fitted here (closed-form LS). Without one, fall back to the legacy per-series
    grid search over all three."""
    if cal is not None:
        Ps, th = cal["P_stat"], cal["gamma"]
        chi = float(np.sum(np.maximum(P - Ps, 1e-9) * x ** th) / np.sum(x ** (2 * th)))
        return Ps, max(chi, 1e-6), th, False
    best = None
    TH = np.linspace(1.0, 6.0, 160)
    for Ps in np.linspace(10, P.min() - 2, 60):
        for th in TH:
            g = x ** th
            chi = np.sum(g * (P - Ps)) / np.sum(g * g)
            if chi <= 0:
                continue
            e = np.sum((Ps + chi * g - P) ** 2)
            if best is None or e < best[0]:
                best = (e, Ps, th, chi)
    _, Ps, th, chi = best
    th_railed = th >= TH[-1] - 1e-9
    return Ps, chi, th, th_railed


def _x_of_P(Q, Ps, chi, th, phi_min=1e-3):
    return _phi(Q, Ps, chi, th, phi_min)


# ------------------------------------------------------------------ prefill (unified)
def fit_prefill_unified(P, T, F, f_max, cal=None):
    """T(P) = T_fmax * x(P)^p   (the T_mem->0 case)."""
    x = np.clip(F / f_max, 1e-3, 1.0)
    Ps, chi, th, th_railed = fit_power_side(P, x, cal)
    pmin = cal["phi_min"] if cal else 1e-3
    A = np.vstack([np.ones_like(x), np.log(x)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(T), rcond=None)
    T_fmax, p = float(np.exp(coef[0])), float(coef[1])

    def T_of_P(Q):
        return T_fmax * _x_of_P(Q, Ps, chi, th, pmin) ** p

    pr = dict(P_s=Ps, chi=chi, theta=th, p=p, T_fmax=T_fmax, exp_PT=p / th,
              th_railed=th_railed, R2_clk=r2(T, T_fmax * x ** p),
              R2=r2(T, T_of_P(P)), relRMSE=rel_rmse(T, T_of_P(P)))
    return T_of_P, pr


# ------------------------------------------------------------------ decode (additive 3-stage)
def fit_decode_additive(P, T, F, B, f_max, cal=None):
    """T(P) = B / (T_mem + Cc*(x(P)^-p - 1)); T_mem anchored to the plateau (top-3 by throughput)."""
    x = np.clip(F / f_max, 1e-3, 1.0)
    Ps, chi, th, th_railed = fit_power_side(P, x, cal)
    pmin = cal["phi_min"] if cal else 1e-3

    T_plateau = float(np.mean(np.sort(T)[-3:]))
    T_mem = B / T_plateau
    y = B / T - T_mem
    best = None
    PG = np.linspace(0.2, 10.0, 300)
    for p in PG:
        g = x ** (-p) - 1.0
        d = np.sum(g * g)
        Cc = max(np.sum(g * y) / d, 0.0) if d > 0 else 0.0
        e = np.sum((Cc * g - y) ** 2)
        if best is None or e < best[0]:
            best = (e, p, Cc)
    _, p, Cc = best
    edge = (p >= PG[-1] - 1e-9) or (p <= PG[0] + 1e-9) or th_railed

    def T_of_P(Q):
        xx = _x_of_P(Q, Ps, chi, th, pmin)
        return B / (T_mem + Cc * (xx ** (-p) - 1.0))

    if Cc > 0:
        x1 = (Cc / (T_mem + Cc)) ** (1.0 / p)
        x2 = (Cc / (0.05 * T_mem + Cc)) ** (1.0 / p)
        P1, P2 = Ps + chi * x1 ** th, Ps + chi * x2 ** th
    else:
        P1 = P2 = float("nan")

    pr = dict(P_s=Ps, chi=chi, theta=th, T_mem_ms=T_mem * 1e3, Cc_ms=Cc * 1e3, p=p,
              T_plateau=T_plateau, T_max=T_plateau, P1=P1, P2=P2, edge=edge,
              R2_clk=r2(T, B / (T_mem + Cc * (x ** (-p) - 1.0))),
              R2=r2(T, T_of_P(P)), relRMSE=rel_rmse(T, T_of_P(P)))
    return T_of_P, pr


# ================================================================= UPDATED first-principles theory
# The analytical model of MODEL_AND_RESULTS.zh.md. Relative frequency phi=f_sm/f_max is the shared
# knob; both phases plug F(P)=eta_C*F_peak*phi into a roofline per-token time. These are the forms
# the power-vs-throughput / power-vs-tok/J figures draw (plot_power_curves.py); the legacy fits above
# are kept for the rack solver and the old-vs-new comparison plots.

def fit_prefill_theory(P, T, F, f_max, cal=None):
    """§3: prefill is compute-bound over the whole range, so throughput is LINEAR in phi:
        X_pre(P) = a * phi(P),   phi(P)=x(P)=((P-P_s)/chi)^(1/theta)   (exponent 1, i.e. X ∝ f_sm).
    a = eta_C^pre * F_peak / (2N + 2Lds) is fit as one scale (LS through the origin in x)."""
    x = np.clip(F / f_max, 1e-3, 1.0)
    Ps, chi, th, th_railed = fit_power_side(P, x, cal)
    pmin = cal["phi_min"] if cal else 1e-3
    a = float(np.sum(x * T) / np.sum(x * x))            # X = a*x, least squares through origin

    def T_of_P(Q):
        return a * _x_of_P(Q, Ps, chi, th, pmin)

    pr = dict(P_s=Ps, chi=chi, theta=th, a=a, T_fmax=a, p=1.0, th_railed=th_railed,
              R2_clk=r2(T, a * x), R2=r2(T, T_of_P(P)), relRMSE=rel_rmse(T, T_of_P(P)))
    return T_of_P, pr


def fit_decode_theory(P, T, F, B, f_max, cal=None):
    """§4: decode per-token time is the SUM of two rooflines (linear part + attention part) plus a
    fixed per-token overhead, x=f_sm/f_max:
        tau(x) = max(a/x, b) + max(c/x, d) + e,   X_dec = 1/tau.
        a,c = compute times (∝1/phi) of the linear / attention parts at f_max;
        b,d = their bandwidth-bound (phi-independent) times; e = T0/B.
    Three segments emerge as phi rises: both compute-bound (X∝phi) -> attention saturates (mixed) ->
    both bandwidth-bound (plateau X=1/(b+d+e)). Fit (a,b,c,d,e)>=0 to measured (x, 1/T)."""
    from scipy.optimize import least_squares
    x = np.clip(F / f_max, 1e-3, 1.0)
    Ps, chi, th, th_railed = fit_power_side(P, x, cal)
    pmin = cal["phi_min"] if cal else 1e-3
    tau = 1.0 / np.asarray(T, float)                    # measured per-token time (s)

    def rt(par, xx):
        a, b, c, d, e = par
        return np.maximum(a / xx, b) + np.maximum(c / xx, d) + e

    tau_plat = float(np.min(tau))                       # plateau ~ b+d+e (fastest = highest throughput)
    o = np.argsort(x)
    comp = float(tau[o[0]] * x[o[0]])                   # tau*x at lowest clock ~ (a+c)
    p0 = [0.5 * comp, 0.35 * tau_plat, 0.5 * comp, 0.35 * tau_plat, 0.15 * tau_plat]
    res = least_squares(lambda par: rt(par, x) - tau, p0, bounds=(1e-12, np.inf),
                        method="trf", max_nfev=20000)
    a, b, c, d, e = (float(v) for v in res.x)

    def T_of_P(Q):
        xx = _x_of_P(Q, Ps, chi, th, pmin)
        return 1.0 / (np.maximum(a / xx, b) + np.maximum(c / xx, d) + e)

    T_plateau = 1.0 / (b + d + e)
    xk = sorted([min(a / b, 1.0), min(c / d, 1.0)])     # the two compute->bandwidth crossover clocks
    P1, P2 = (Ps + chi * xk[0] ** th, Ps + chi * xk[1] ** th)
    pr = dict(P_s=Ps, chi=chi, theta=th, a=a, b=b, c=c, d=d, e=e, th_railed=th_railed,
              T_plateau=T_plateau, T_max=T_plateau, P1=P1, P2=P2,
              R2_clk=r2(T, 1.0 / rt(res.x, x)), R2=r2(T, T_of_P(P)), relRMSE=rel_rmse(T, T_of_P(P)))
    return T_of_P, pr
