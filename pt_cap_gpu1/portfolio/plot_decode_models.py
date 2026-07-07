"""DECODE model refinement: three-stage ADDITIVE T_mem+T_comp model vs the old piecewise min().

The old model  T(P) = min(T_V2f(P), T_max)  is the PERFECT-OVERLAP roofline limit: per-token time
= max(T_mem, T_comp). It systematically underfits the mid-power region where compute and memory
times are comparable. The refined model (same physics as ../decode_model_theory.md, applied
portfolio-wide) is the ADDITIVE (serial / no-overlap) limit with three stages:

  per-token time   t(x)   = T_mem + T_comp(x),      T_comp(x) = Cc*(x^-p - 1),   x = f_sm/f_max
  power            P(x)   = P_s + chi * x^theta
  explicit         T(P)   = B / ( T_mem + Cc*[ ((P-P_s)/chi)^(-p/theta) - 1 ] ),  x clamped <= 1

  stage I   compute-dominated   (T_comp >> T_mem):  power-law rise  ~ (P-P_s)^(p/theta)
  stage II  compute-memory MIX  (T_comp ~  T_mem):  additive transition, diminishing returns
  stage III memory plateau      (T_comp -> 0)    :  T = B/T_mem, decoupled from P
  boundaries: I/II at T_comp = T_mem  ->  x1 = (Cc/(T_mem+Cc))^(1/p)
              II/III at T_comp = 5% T_mem -> x2 = (Cc/(0.05*T_mem+Cc))^(1/p)

FIT (two-step, uses the sm_clk telemetry recorded at every cap point -- no extra measurement):
  1. power side  (sm_clk, power):      grid P_s x theta, LS chi
  2. thpt side   (sm_clk, throughput): T_mem anchored to the plateau (top-3 mean, B/T_plateau);
                                       grid p, LS Cc on  B/T - T_mem = Cc*(x^-p - 1)
  3. compose T(P); score BOTH models with R^2 on ALL points in power space (fair comparison).

Extreme memory-bound workloads (summarize/classify: nearly flat curves) are the small-Cc limit of
the SAME law -- no special-casing, unlike the old model where they had to be flagged 'saturated'.

Outputs: fig_decode_models.png (8 panels, both models + stage boundaries)
         decode_model_compare.csv (params + R^2 old vs new)
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from portfolio import PORTFOLIO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PORTFOLIO_DATA", os.path.join(HERE, "data"))   # e.g. data_v1 for the old set
F_MAX = 1530.0                     # V100 max SM clock (MHz)


def read_decode(path):
    if not os.path.exists(path):
        return None
    rows = [r for r in csv.DictReader(open(path)) if float(r["throughput_tok_s"]) > 0]
    return dict(P=np.array([float(r["power_avg_w"]) for r in rows]),
                T=np.array([float(r["throughput_tok_s"]) for r in rows]),
                F=np.array([float(r["sm_clk_avg"]) for r in rows]))


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - yhat) ** 2) / ss if ss > 0 else float("nan")


def rel_rmse(y, yhat):
    """RMSE as % of the mean level. Fairer than R^2 for near-flat curves, where R^2's
    denominator (total variance) goes to zero and even a tiny misfit looks catastrophic."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)) / np.mean(y) * 100)


# ---------------------------------------------------------------- additive three-stage model
def fit_additive(P, T, F, B):
    """Two-step fit in clock space, composed into T(P). Returns (T_of_P, params)."""
    x = np.clip(F / F_MAX, 1e-3, 1.0)

    # -- 1. power side: P = P_s + chi * x^theta ------------------------------------
    # (P_s and theta are partially degenerate -- POWER_THROUGHPUT_MODEL.md section 8 -- so the
    #  individual values are indicative; only the composed x(P) matters for T(P).)
    best = None
    TH_GRID = np.linspace(1.0, 6.0, 160)
    for Ps in np.linspace(10, P.min() - 2, 60):
        for th in TH_GRID:
            g = x ** th
            chi = np.sum(g * (P - Ps)) / np.sum(g * g)
            if chi <= 0:
                continue
            e = np.sum((Ps + chi * g - P) ** 2)
            if best is None or e < best[0]:
                best = (e, Ps, th, chi)
    _, Ps, th, chi = best
    if th >= TH_GRID[-1] - 1e-9:
        print(f"    [warn] theta railed at grid max {TH_GRID[-1]:.1f}")
    r2_pow = r2(P, Ps + chi * x ** th)

    # -- 2. throughput side: B/T = T_mem + Cc*(x^-p - 1), T_mem anchored to plateau --
    T_plateau = float(np.mean(np.sort(T)[-3:]))
    T_mem = B / T_plateau                      # seconds per token at the ceiling
    y = B / T - T_mem                          # residual compute time per token
    best = None
    P_GRID = np.linspace(0.2, 10.0, 300)
    for p in P_GRID:
        g = x ** (-p) - 1.0
        denom = np.sum(g * g)
        Cc = max(np.sum(g * y) / denom, 0.0) if denom > 0 else 0.0
        e = np.sum((Cc * g - y) ** 2)
        if best is None or e < best[0]:
            best = (e, p, Cc)
    _, p, Cc = best
    # p is an EFFECTIVE exponent (OPS ~ f^p including occupancy/latency-hiding collapse at low
    # clock), so p>1 is expected; for near-flat workloads the residual y is tiny and p is weakly
    # identified -- flag any grid-edge hit so those rows are not over-interpreted.
    edge = (p >= P_GRID[-1] - 1e-9) or (p <= P_GRID[0] + 1e-9) or (th >= TH_GRID[-1] - 1e-9)
    if edge:
        print(f"    [warn] exponent railed (p={p:.2f}, theta={th:.2f}) -- weakly identified")
    r2_clk = r2(T, B / (T_mem + Cc * (x ** (-p) - 1.0)))

    # -- 3. compose T(P): invert x(P), clamp to [x_lo, 1] ---------------------------
    def T_of_P(Q):
        Q = np.asarray(Q, float)
        base = np.maximum((Q - Ps) / chi, 1e-9)        # P<P_s (below static floor) -> x floor
        xx = np.clip(base ** (1.0 / th), 1e-3, 1.0)
        return B / (T_mem + Cc * (xx ** (-p) - 1.0))

    # stage boundaries (in x, then power)
    if Cc > 0:
        x1 = (Cc / (T_mem + Cc)) ** (1.0 / p)              # T_comp = T_mem
        x2 = (Cc / (0.05 * T_mem + Cc)) ** (1.0 / p)       # T_comp = 5% T_mem
    else:
        x1 = x2 = 0.0                                       # flat: stage III everywhere
    P1 = Ps + chi * x1 ** th if x1 > 0 else float("nan")
    P2 = Ps + chi * x2 ** th if x2 > 0 else float("nan")

    pr = dict(P_s=Ps, chi=chi, theta=th, T_mem_ms=T_mem * 1e3, Cc_ms=Cc * 1e3, p=p,
              T_plateau=T_plateau, P1=P1, P2=P2, edge=edge,
              R2_pow=r2_pow, R2_clk=r2_clk, R2=r2(T, T_of_P(P)),
              relRMSE=rel_rmse(T, T_of_P(P)))
    return T_of_P, pr


# ---------------------------------------------------------------- leave-one-out CV
def loo_rmse(P, T, F, B, fit_fn):
    """Overfitting guard (8 points, additive fits ~6 params): refit on 7 points, predict the
    held-out one; report the pooled out-of-sample RMSE as % of the mean level."""
    errs = []
    for i in range(len(P)):
        m = np.ones(len(P), bool); m[i] = False
        try:
            f, _ = fit_fn(P[m], T[m], F[m], B) if F is not None else fit_fn(P[m], T[m])
            errs.append(float(f(np.array([P[i]]))[0]) - T[i])
        except Exception:
            errs.append(np.nan)
    errs = np.array(errs, float)
    ok = np.isfinite(errs)
    return float(np.sqrt(np.mean(errs[ok] ** 2)) / np.mean(T[ok]) * 100)


# ---------------------------------------------------------------- old piecewise min() model
def v2f_inverse(P0, kappa, rho, t_hi):
    Tg = np.linspace(1, t_hi, 8000)
    Pg = P0 + kappa * Tg * (1 + rho * Tg) ** 2
    return lambda P: np.interp(P, Pg, Tg)


def fit_old_min(P, T):
    """v1 model, reproduced FAITHFULLY (fairness for the comparison):
      - saturated special case exactly as plot_portfolio.py: if the lowest-power point already
        exceeds 72% of the plateau, the v1 model is a flat ceiling (no spurious rise);
      - rho grid widened to 0.05 (the 3e-3 ceiling railed on 8/8 workloads and handicapped
        the old model -- verified: longform R2 0.66->0.93 with the proper grid).
    Plateau = top-3-by-throughput mean. Metrics on ALL points."""
    o = np.argsort(P); Ps_, Ts_ = P[o], T[o]
    Tmax = float(np.mean(Ts_[-3:]))                           # v1 anchor: 3 highest-POWER points
    if Ts_[0] / Tmax > 0.72:                                  # v1 saturated branch
        f = lambda Q: np.full_like(np.asarray(Q, float), Tmax)
        return f, dict(T_max=Tmax, R2=r2(T, f(P)), relRMSE=rel_rmse(T, f(P)), saturated=True)
    keep = T >= 0.25 * Tmax
    best = None
    for P0 in np.linspace(0, P[keep].min(), 60):
        for rho in np.linspace(0, 5e-2, 500):
            g = T[keep] * (1 + rho * T[keep]) ** 2
            kap = np.sum(g * (P[keep] - P0)) / np.sum(g * g)
            if kap > 0:
                e = np.sum((P0 + kap * g - P[keep]) ** 2)
                if best is None or e < best[0]:
                    best = (e, P0, rho, kap)
    _, P0, rho, kap = best
    core = v2f_inverse(P0, kappa=kap, rho=rho, t_hi=max(T.max() * 2.0, Tmax * 1.5))
    f = lambda Q: np.minimum(core(Q), Tmax)
    return f, dict(T_max=Tmax, R2=r2(T, f(P)), relRMSE=rel_rmse(T, f(P)), saturated=False)


# ---------------------------------------------------------------- main
def main():
    recs = []
    for w in PORTFOLIO:
        d = read_decode(os.path.join(DATA, f"{w['id']}_decode.csv"))
        if d is None or len(d["P"]) < 4:
            continue
        addF, add = fit_additive(d["P"], d["T"], d["F"], w["decode_batch"])
        oldF, old = fit_old_min(d["P"], d["T"])
        add["LOO"] = loo_rmse(d["P"], d["T"], d["F"], w["decode_batch"], fit_additive)
        old["LOO"] = loo_rmse(d["P"], d["T"], None, None, fit_old_min)
        recs.append(dict(w=w, d=d, addF=addF, add=add, oldF=oldF, old=old))
        print(f"{w['id']:<18} old R2={old['R2']:+.3f} rmse={old['relRMSE']:.1f}% loo={old['LOO']:.1f}%  ->  "
              f"additive R2={add['R2']:+.3f} rmse={add['relRMSE']:.1f}% loo={add['LOO']:.1f}%   "
              f"[T_mem={add['T_mem_ms']:.1f}ms Cc={add['Cc_ms']:.2f}ms p={add['p']:.2f}"
              f"{' EDGE' if add['edge'] else ''} | stages P1={add['P1']:.0f} P2={add['P2']:.0f}W]")

    # ---- figure: 8 panels, data + both models + stage boundaries ----
    n = len(recs)
    ncol = 2
    nrow = (n + 1) // 2
    fig, ax = plt.subplots(nrow, ncol, figsize=(12, 3.0 * nrow), squeeze=False)
    for i, r in enumerate(recs):
        a = ax[i // ncol, i % ncol]
        d, w = r["d"], r["w"]
        g = np.linspace(d["P"].min() * 0.97, d["P"].max() * 1.03, 300)
        a.scatter(d["P"], d["T"], c="C0", s=45, ec="k", lw=.5, zorder=6, label="measured")
        a.plot(g, r["oldF"](g), color="gray", ls="--", lw=1.6,
               label=f"old min(V²f,T_max)  R²={r['old']['R2']:.2f} · {r['old']['relRMSE']:.1f}%")
        a.plot(g, r["addF"](g), "k-", lw=2.0,
               label=f"additive 3-stage  R²={r['add']['R2']:.2f} · {r['add']['relRMSE']:.1f}%")
        for Pb, lab in ((r["add"]["P1"], "I|II"), (r["add"]["P2"], "II|III")):
            if r["add"]["edge"]:
                break                                  # railed exponents -> boundaries are artifacts
            if np.isfinite(Pb) and d["P"].min() * 0.95 < Pb < d["P"].max() * 1.05:
                a.axvline(Pb, color="C3", ls=":", lw=1.2)
                a.text(Pb, a.get_ylim()[0], f" {lab}", color="C3", fontsize=7,
                       va="bottom", ha="left")
        a.set_title(f"{w['id']}   (C={w['decode_ctx']}, B={w['decode_batch']})", fontsize=10)
        a.grid(alpha=.3); a.legend(loc="lower right", fontsize=7.5)
        if i % ncol == 0:
            a.set_ylabel("decode tok/s")
        if i // ncol == nrow - 1:
            a.set_xlabel("power (W)")
    for j in range(n, nrow * ncol):
        ax[j // ncol, j % ncol].axis("off")
    fig.suptitle("DECODE refined: additive $t=T_{mem}+T_{comp}$ three-stage model vs old piecewise min()\n"
                 "stage I compute-dominated · stage II compute–memory mix · stage III bandwidth plateau",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = os.path.join(HERE, "fig_decode_models.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); print("wrote", out)

    # ---- csv ----
    out = os.path.join(HERE, "decode_model_compare.csv")
    keys = ["id", "decode_ctx", "decode_batch", "R2_old", "R2_additive",
            "relRMSE_old_pct", "relRMSE_additive_pct", "LOO_old_pct", "LOO_additive_pct",
            "R2_power_fit", "R2_clock_fit", "P_s_W", "chi_W", "theta", "T_mem_ms", "Cc_ms", "p",
            "exponents_railed", "stageI_II_W", "stageII_III_W", "T_plateau"]
    with open(out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys); wtr.writeheader()
        for r in recs:
            w, a, o = r["w"], r["add"], r["old"]
            # suppress the stage boundaries when the exponents railed: they are then artifacts
            show_stage = np.isfinite(a["P1"]) and not a["edge"]
            wtr.writerow({"id": w["id"], "decode_ctx": w["decode_ctx"],
                          "decode_batch": w["decode_batch"],
                          "R2_old": round(o["R2"], 3), "R2_additive": round(a["R2"], 3),
                          "relRMSE_old_pct": round(o["relRMSE"], 1),
                          "relRMSE_additive_pct": round(a["relRMSE"], 1),
                          "LOO_old_pct": round(o["LOO"], 1), "LOO_additive_pct": round(a["LOO"], 1),
                          "R2_power_fit": round(a["R2_pow"], 3), "R2_clock_fit": round(a["R2_clk"], 3),
                          "P_s_W": round(a["P_s"], 1), "chi_W": round(a["chi"], 1),
                          "theta": round(a["theta"], 2), "T_mem_ms": round(a["T_mem_ms"], 2),
                          "Cc_ms": round(a["Cc_ms"], 3), "p": round(a["p"], 2),
                          "exponents_railed": a["edge"],
                          "stageI_II_W": round(a["P1"], 1) if show_stage else "",
                          "stageII_III_W": round(a["P2"], 1) if show_stage else "",
                          "T_plateau": round(a["T_plateau"], 1)})
    print("wrote", out)


if __name__ == "__main__":
    main()
