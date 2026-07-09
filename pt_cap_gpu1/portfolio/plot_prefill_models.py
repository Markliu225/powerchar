"""PREFILL under the SAME explicit construction as the decode model (decode_model_theory.md):

  per-token time   t(x) = T_mem + C*x^-p          x = f_sm/f_max
  power            P(x) = P_s + chi*x^theta
  ------------------------------------------------
  PREFILL is the T_mem -> 0 degenerate case (compute-bound, I >> I*): t = C*x^-p, so

      Throughput(P) = T_fmax * ( (P - P_s)/chi )^{p/theta} ,   x clamped <= 1

  ONE segment, no stages, no plateau inside the power range (the frequency ceiling x=1 is
  the only saturation, and the cap range typically ends below it). p is the effective
  compute-rate exponent (T ~ f^p; DVFS measured p~0.9), theta the DVFS power exponent.

FIT: identical two-step clock-space procedure as the decode fit:
  1. power side  (sm_clk, power):      grid P_s x theta, LS chi
  2. thpt side   (sm_clk, throughput): log-space LS  ->  p, T_fmax
  3. compose Throughput(P); score in power space (R^2, rel-RMSE, LOO)
BASELINE: the legacy V2f form  P(T) = P0 + kappa*T*(1+rho*T)^2  (fitted as before, scored
identically) -- so the doc can claim the unified model on evidence, not aesthetics.

Outputs: fig_prefill_models.png, prefill_model_compare.csv  (into PORTFOLIO_DATA dir if set)
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from portfolio import PORTFOLIO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PORTFOLIO_DATA", os.path.join(HERE, "data"))
if not os.path.isabs(DATA):
    DATA = os.path.join(HERE, DATA)
FIGDIR = DATA if os.environ.get("PORTFOLIO_DATA") else HERE


def _f_max():
    import json
    try:
        v = float(json.load(open(os.path.join(DATA, "meta.json")))["f_max_mhz"])
        print(f"F_MAX = {v:.0f} MHz  (from {os.path.join(DATA, 'meta.json')})")
        return v
    except Exception as e:
        print(f"[WARN] no usable meta.json in {DATA} ({type(e).__name__}); fallback F_MAX=1530 (V100)")
        return 1530.0


F_MAX = _f_max()


def read_prefill(path):
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
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)) / np.mean(y) * 100)


# ------------------------------------------------ unified explicit model (same as decode's core)
def fit_power_side(P, x):
    """P = P_s + chi * x^theta  (grid P_s x theta, LS chi) -- shared with the decode fit."""
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
    return Ps, chi, th


def fit_unified(P, T, F):
    """Prefill as the T_mem->0 case: T(x)=T_fmax*x^p; compose with x(P)."""
    x = np.clip(F / F_MAX, 1e-3, 1.0)
    Ps, chi, th = fit_power_side(P, x)
    # throughput side: log T = log T_fmax + p log x  (exact LS)
    A = np.vstack([np.ones_like(x), np.log(x)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(T), rcond=None)
    T_fmax, p = float(np.exp(coef[0])), float(coef[1])
    r2_clk = r2(T, T_fmax * x ** p)

    def T_of_P(Q):
        Q = np.asarray(Q, float)
        base = np.maximum((Q - Ps) / chi, 1e-9)
        xx = np.clip(base ** (1.0 / th), 1e-3, 1.0)
        return T_fmax * xx ** p

    pr = dict(P_s=Ps, chi=chi, theta=th, p=p, T_fmax=T_fmax, exp_PT=p / th,
              R2_clk=r2_clk, R2=r2(T, T_of_P(P)), relRMSE=rel_rmse(T, T_of_P(P)))
    return T_of_P, pr


# ------------------------------------------------ legacy V2f baseline (P(T) form, inverted)
def v2f_inverse(P0, kappa, rho, t_hi):
    Tg = np.linspace(1, t_hi, 8000)
    Pg = P0 + kappa * Tg * (1 + rho * Tg) ** 2
    return lambda P: np.interp(P, Pg, Tg)


def fit_v2f_baseline(P, T):
    best = None
    for P0 in np.linspace(0, P.min(), 60):
        for rho in np.linspace(0, 3e-3, 500):
            g = T * (1 + rho * T) ** 2
            kap = np.sum(g * (P - P0)) / np.sum(g * g)
            if kap > 0:
                e = np.sum((P0 + kap * g - P) ** 2)
                if best is None or e < best[0]:
                    best = (e, P0, rho, kap)
    _, P0, rho, kap = best
    f = v2f_inverse(P0, kap, rho, T.max() * 2.0)
    return f, dict(P0=P0, kappa=kap, rho=rho, R2=r2(T, f(P)), relRMSE=rel_rmse(T, f(P)))


def loo(P, T, F, fit_fn):
    errs = []
    for i in range(len(P)):
        m = np.ones(len(P), bool); m[i] = False
        try:
            f, _ = fit_fn(P[m], T[m], F[m]) if F is not None else fit_fn(P[m], T[m])
            errs.append(float(f(np.array([P[i]]))[0]) - T[i])
        except Exception:
            errs.append(np.nan)
    errs = np.array(errs, float); ok = np.isfinite(errs)
    return float(np.sqrt(np.mean(errs[ok] ** 2)) / np.mean(T[ok]) * 100)


def main():
    recs = []
    for w in PORTFOLIO:
        d = read_prefill(os.path.join(DATA, f"{w['id']}_prefill.csv"))
        if d is None or len(d["P"]) < 4:
            continue
        uF, uni = fit_unified(d["P"], d["T"], d["F"])
        vF, v2f = fit_v2f_baseline(d["P"], d["T"])
        uni["LOO"] = loo(d["P"], d["T"], d["F"], fit_unified)
        v2f["LOO"] = loo(d["P"], d["T"], None, fit_v2f_baseline)
        recs.append(dict(w=w, d=d, uF=uF, uni=uni, vF=vF, v2f=v2f))
        print(f"{w['id']:<18} V2f R2={v2f['R2']:+.3f} rmse={v2f['relRMSE']:.1f}% loo={v2f['LOO']:.1f}%  ->  "
              f"unified R2={uni['R2']:+.3f} rmse={uni['relRMSE']:.1f}% loo={uni['LOO']:.1f}%   "
              f"[p={uni['p']:.2f} th={uni['theta']:.2f} p/th={uni['exp_PT']:.2f} "
              f"P_s={uni['P_s']:.0f} T_fmax={uni['T_fmax']:.0f}]")

    n = len(recs); ncol = 2; nrow = (n + 1) // 2
    fig, ax = plt.subplots(nrow, ncol, figsize=(12, 3.0 * nrow), squeeze=False)
    for i, r in enumerate(recs):
        a = ax[i // ncol, i % ncol]
        d, w = r["d"], r["w"]
        g = np.linspace(d["P"].min() * 0.97, d["P"].max() * 1.03, 300)
        a.scatter(d["P"], d["T"], c="C1", s=45, ec="k", lw=.5, zorder=6, label="measured")
        a.plot(g, r["vF"](g), color="gray", ls="--", lw=1.6,
               label=f"legacy V²f P(T)  R²={r['v2f']['R2']:.2f} · {r['v2f']['relRMSE']:.1f}%")
        a.plot(g, r["uF"](g), "k-", lw=2.0,
               label=f"unified $T_{{fmax}}\\,x(P)^p$  R²={r['uni']['R2']:.2f} · {r['uni']['relRMSE']:.1f}%")
        a.set_title(f"{w['id']}   (S={w['prefill_seq_len']}, B={w['prefill_batch']})", fontsize=10)
        a.grid(alpha=.3); a.legend(loc="lower right", fontsize=7.5)
        if i % ncol == 0:
            a.set_ylabel("prefill tok/s")
        if i // ncol == nrow - 1:
            a.set_xlabel("power (W)")
    for j in range(n, nrow * ncol):
        ax[j // ncol, j % ncol].axis("off")
    fig.suptitle("PREFILL under the unified explicit model: $T(P)=T_{fmax}\\,((P-P_s)/\\chi)^{p/\\theta}$"
                 " (the $T_{mem}\\!\\to\\!0$ case of the decode law) vs the legacy V²f $P(T)$ fit",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = os.path.join(FIGDIR, "fig_prefill_models.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); print("wrote", out)

    out = os.path.join(FIGDIR, "prefill_model_compare.csv")
    keys = ["id", "prefill_seq_len", "prefill_batch", "R2_v2f", "R2_unified",
            "relRMSE_v2f_pct", "relRMSE_unified_pct", "LOO_v2f_pct", "LOO_unified_pct",
            "p", "theta", "p_over_theta", "P_s_W", "chi_W", "T_fmax", "R2_clock_fit"]
    with open(out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys); wtr.writeheader()
        for r in recs:
            w, u, v = r["w"], r["uni"], r["v2f"]
            wtr.writerow({"id": w["id"], "prefill_seq_len": w["prefill_seq_len"],
                          "prefill_batch": w["prefill_batch"],
                          "R2_v2f": round(v["R2"], 3), "R2_unified": round(u["R2"], 3),
                          "relRMSE_v2f_pct": round(v["relRMSE"], 1),
                          "relRMSE_unified_pct": round(u["relRMSE"], 1),
                          "LOO_v2f_pct": round(v["LOO"], 1), "LOO_unified_pct": round(u["LOO"], 1),
                          "p": round(u["p"], 2), "theta": round(u["theta"], 2),
                          "p_over_theta": round(u["exp_PT"], 3),
                          "P_s_W": round(u["P_s"], 1), "chi_W": round(u["chi"], 1),
                          "T_fmax": round(u["T_fmax"], 0), "R2_clock_fit": round(u["R2_clk"], 3)})
    print("wrote", out)


if __name__ == "__main__":
    main()
