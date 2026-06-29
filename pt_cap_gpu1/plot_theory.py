"""Read CSV -> fit models -> plot.  Prefill & decode power<->throughput on V100 GPU1.

Pipeline (one entry point: `python3 plot_theory.py`):
  1. LOAD   pt_cap.csv (prefill)  +  decode_frontier.csv (the pre-extracted decode frontier).
  2. FIT    a model to each phase.
  3. PLOT   throughput-vs-power and efficiency-vs-power, measured + model -> fig_theory_vs_measured.png.

The decode FRONTIER (max throughput per power budget) is produced separately by make_frontier.py
(decode_pt.csv -> decode_frontier.csv); this script just reads it.

Axes: POWER (x) -> THROUGHPUT (y).  Models:
  PREFILL  compute-bound V²f:   P(T) = P0 + kappa*T*(1+rho*T)^2     (plotted as its inverse T(P))
  DECODE   PIECEWISE roofline:  T(P) = min( T_{V²f}(P), T_max )
           power-limited V²f rise -> bandwidth-limited cap T_max.
"""
from __future__ import annotations
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PREFILL_CSV = "prefill.csv"            # phase=prefill rows
FRONTIER_CSV = "decode.csv"  # produced by make_frontier.py
OUT = "fig_theory_vs_measured.png"

# ---- DECODE curve hand-tuning -------------------------------------------------------------------
# T_dec(P) = min( T_{V²f}(P), T_max ), with the V²f core P = P0 + kappa*T*(1+rho*T)^2.
# Set any of these to a NUMBER to lock that parameter by hand; leave None to auto-fit it from the CSV.
DEC_P0 = 93        # W       core static floor          (auto ~35)
DEC_KAPPA = 2.4e-2     #         core dynamic coeff         (auto ~2.4e-2)
DEC_RHO = 1.0e-3       #         core V-f coupling          (auto ~3.0e-3)
DEC_TMAX = None      # tok/s   bandwidth ceiling          (auto ~721)

# ---- PREFILL curve hand-tuning ------------------------------------------------------------------
# V²f core P = P0 + kappa*T*(1+rho*T)^2.  Set to a NUMBER to lock by hand; None = auto-fit from CSV.
PRE_P0 = 73        # W       static floor               (auto ~73)
PRE_KAPPA = 3.13e-3     #         dynamic coeff              (auto ~1.5e-3)
PRE_RHO = 1.8e-4       #         V-f coupling               (auto ~3.3e-4)


# ======================================================================= LOAD
def read(name, p_key, t_key, where=lambda r: True):
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, name))) if where(r)]
    return (np.array([float(r[p_key]) for r in rows]),
            np.array([float(r[t_key]) for r in rows]))


# ======================================================================== FIT
def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def v2f_inverse(P0, kappa, rho, t_hi):
    """Invert P=P0+kappa*T*(1+rho*T)^2 -> concave T(P) over T in (0, t_hi] by interpolation."""
    Tg = np.linspace(1, t_hi, 8000)
    Pg = P0 + kappa * Tg * (1 + rho * Tg) ** 2        # P(T): monotone -> invertible
    return lambda P: np.interp(P, Pg, Tg)


def apply_overrides(params, overrides):
    """overrides = list of (key, value|None): lock the non-None ones by hand; record them in 'manual'."""
    for k, v in overrides:
        if v is not None:
            params[k] = v
    params["manual"] = [k for k, v in overrides if v is not None]
    return params


def fit_v2f(P, T):
    """Fit V²f  P(T)=P0+kappa*T*(1+rho*T)^2  (grid P0,rho; kappa LS).  Return the param dict."""
    best = None
    for P0 in np.linspace(0, P.min(), 60):
        for rho in np.linspace(0, 3e-3, 700):
            x = T * (1 + rho * T) ** 2
            kap = np.sum(x * (P - P0)) / np.sum(x * x)
            if kap > 0:
                e = np.sum((P0 + kap * x - P) ** 2)
                if best is None or e < best[0]:
                    best = (e, P0, rho, kap)
    _, P0, rho, kap = best
    return dict(P0=P0, kappa=kap, rho=rho)


def fit_prefill(P, T):
    """Prefill V²f throughput-vs-power T(P).  Auto-fit, then apply any hand-set PRE_* overrides."""
    pr = apply_overrides(fit_v2f(P, T), [("P0", PRE_P0), ("kappa", PRE_KAPPA), ("rho", PRE_RHO)])
    T_of_P = v2f_inverse(pr["P0"], pr["kappa"], pr["rho"], T.max() * 2.0)
    pr["R2"] = r2(T, T_of_P(P))
    return T_of_P, pr


def decode_model(P0, kappa, rho, T_max):
    """Build T(P) = min( T_{V²f}(P), T_max ) from explicit parameters."""
    core = v2f_inverse(P0, kappa, rho, max(T_max * 1.5, 1200))
    return lambda P: np.minimum(core(P), T_max)


def fit_decode(P, T):
    """Decode PIECEWISE roofline on the frontier:  T(P) = min(T_{V²f}(P), T_max).
    Auto-fit, then apply any hand-set DEC_* overrides. The 1-2 degenerate tiny-batch points
    (b1/b2 draw <100W, T~20-40) are a low-concurrency corner -> excluded from the FIT (still plotted)."""
    keep = T >= 0.25 * T.max()
    pr = fit_v2f(P[keep], T[keep])
    Tcore = v2f_inverse(pr["P0"], pr["kappa"], pr["rho"], T[keep].max() * 2.0)
    pr["T_max"] = min((np.sum((np.minimum(Tcore(P[keep]), tm) - T[keep]) ** 2), tm)
                      for tm in np.linspace(T[keep].max() * 0.98, T[keep].max() * 1.8, 800))[1]
    apply_overrides(pr, [("P0", DEC_P0), ("kappa", DEC_KAPPA), ("rho", DEC_RHO), ("T_max", DEC_TMAX)])
    T_of_P = decode_model(pr["P0"], pr["kappa"], pr["rho"], pr["T_max"])
    pr["R2"] = r2(T[keep], T_of_P(P[keep]))
    return T_of_P, pr


# ======================================================================= PLOT
BOX = dict(boxstyle="round", fc="#fffce0", ec="#888", alpha=.95)


# def textbox(a, txt, xy=(0.04, 0.06)):
#     a.text(*xy, txt, transform=a.transAxes, va="top", ha="left", fontsize=8.5, bbox=BOX)


def main():
    # ---- 1. LOAD ----
    Pp, Tp = read(PREFILL_CSV, "power_avg_w", "throughput_tok_s", lambda r: r["phase"] == "prefill")
    Pf, Tf = read(FRONTIER_CSV, "power_w", "throughput_tok_s")        # decode frontier

    # ---- 2. FIT ----
    preT_of_P, pp = fit_prefill(Pp, Tp); r2_pre = pp["R2"]
    decT_of_P, dp = fit_decode(Pf, Tf)
    print(f"prefill V²f: P0={pp['P0']:.0f} kappa={pp['kappa']:.2e} rho={pp['rho']:.1e}  R²={r2_pre:.3f}"
          + (f"  [hand-set: {', '.join(pp['manual'])}]" if pp['manual'] else "  [all auto-fit]"))
    print(f"decode  min(V²f,T_max): P0={dp['P0']:.0f} kappa={dp['kappa']:.2e} rho={dp['rho']:.1e} "
          f"T_max={dp['T_max']:.0f}  R²={dp['R2']:.3f}"
          + (f"  [hand-set: {', '.join(dp['manual'])}]" if dp['manual'] else "  [all auto-fit]"))

    txt_pre = (r"$T(P):\ P=P_0+\kappa T(1+\rho T)^2$  (V²f)" + "\n"
               rf"$P_0={pp['P0']:.0f}$, $\kappa={pp['kappa']:.2e}$, $\rho={pp['rho']:.2e}$,  $R^2={r2_pre:.3f}$")
    tag = (f"  [hand-set: {', '.join(dp['manual'])}]" if dp['manual'] else "")
    txt_dec = (r"piecewise:  $T_{dec}(P)=\min(\,T_{V^2f}(P),\ T_{max}\,)$" + "\n"
               rf"power-limited V²f rise $\to$ bandwidth cap $T_{{max}}$" + "\n"
               rf"core: $P_0={dp['P0']:.0f}$, $\kappa={dp['kappa']:.2e}$, $\rho={dp['rho']:.1e}$;  "
               rf"$T_{{max}}={dp['T_max']:.0f}$,  $R^2={dp['R2']:.3f}$" + tag)

    # ---- 3. PLOT ----
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    gP = np.linspace(Pp.min() * .97, Pp.max() * 1.02, 200)
    gF = np.linspace(Pf.min() * .97, Pf.max() * 1.02, 200)

    a = ax[0, 0]
    a.scatter(Pp, Tp, c="C1", s=55, ec="k", lw=.5, zorder=5, label="measured")
    a.plot(gP, preT_of_P(gP), "k-", lw=2, label=f"theory (R²={r2_pre:.3f})")
    a.set(xlabel="power (W)", ylabel="throughput (tok/s)", title="PREFILL · throughput vs power")
    a.legend(loc="lower right"); a.grid(alpha=.3); # textbox(a, txt_pre)

    a = ax[0, 1]
    a.scatter(Pf, Tf, c="C0", s=55, ec="k", lw=.5, zorder=5, label="frontier (max T per power)")
    a.plot(gF, decT_of_P(gF), "k-", lw=2, label=f"theory (R²={dp['R2']:.3f})")
    a.axhline(dp["T_max"], color="gray", ls=":", lw=1.3, label=f"bandwidth ceiling T_max={dp['T_max']:.0f}")
    a.set(xlabel="power (W)", ylabel="throughput (tok/s)", title="DECODE · throughput vs power (frontier)")
    a.legend(loc="lower right", fontsize=8); a.grid(alpha=.3); # textbox(a, txt_dec)

    a = ax[1, 0]
    a.scatter(Pp, Tp / Pp, c="C1", s=55, ec="k", lw=.5, zorder=5, label="measured")
    a.plot(gP, preT_of_P(gP) / gP, "k-", lw=2, label="theory")
    a.set(xlabel="power (W)", ylabel="efficiency (tok/J)", title="PREFILL · efficiency vs power")
    a.legend(loc="lower center"); a.grid(alpha=.3); # textbox(a, txt_pre, xy=(0.04, 0.34))

    a = ax[1, 1]
    a.scatter(Pf, Tf / Pf, c="C0", s=55, ec="k", lw=.5, zorder=5, label="frontier")
    a.plot(gF, decT_of_P(gF) / gF, "k-", lw=2, label="theory")
    a.set(xlabel="power (W)", ylabel="efficiency (tok/J)", title="DECODE · efficiency vs power (frontier)")
    a.legend(loc="upper right", fontsize=8); a.grid(alpha=.3); # textbox(a, txt_dec, xy=(0.36, 0.96))

    fig.suptitle("Phi-3-mini on V100 GPU1 — throughput↔power: prefill V²f, decode frontier vs piecewise min(V²f, T_max)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, OUT), dpi=130, bbox_inches="tight")
    print("wrote", os.path.join(HERE, OUT))


if __name__ == "__main__":
    main()
