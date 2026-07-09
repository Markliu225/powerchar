"""Fit the P<->T model to every workload in the portfolio and validate its generality.

For each workload's cap-sweep CSVs (data/<id>_{prefill,decode}.csv) we fit the SAME two laws
used in ../plot_theory.py:
  PREFILL  compute-bound V2f:  P(T) = P0 + kappa*T*(1+rho*T)^2      (convex, no ceiling)
  DECODE   piecewise roofline: T(P) = min( T_{V2f}(P), T_max )       (V2f rise -> bandwidth plateau)

Robustness: the decode plateau T_max is taken as the mean of the 3 highest-power points (always the
saturated tail). A workload whose lowest-cap point already exceeds ~72% of that plateau is flagged
`saturated` (so memory-bound it plateaus below the 100 W floor) and drawn as a flat ceiling instead of
forcing a spurious rise -- this is itself a confirmation: the more memory-bound, the lower the power at
which throughput saturates.

Validation of the ceiling: T_max = B * BW_eff / D_mem, D_mem = weight_bytes + B*C*kv_bytes_per_token.
We calibrate BW_eff PER MODEL (effective decode bandwidth is a model x hardware property), and report the
implied BW_eff = T_max*D_mem/B per workload -- using the CORRECT additive D_mem makes it consistent.

Outputs:
  fig_portfolio_grid.png     small-multiples: prefill + decode throughput-vs-power, model overlay, per workload
  fig_tmax_validation.png    fitted vs predicted decode T_max (per-model BW_eff), colored by model
  portfolio_fits.csv         per-workload fitted params + R^2 + fitted/predicted T_max + implied BW_eff
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from portfolio import PORTFOLIO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PORTFOLIO_DATA", os.path.join(HERE, "data"))   # e.g. data_h200 / data_v1
if not os.path.isabs(DATA):
    DATA = os.path.join(HERE, DATA)
# figures/CSVs land next to the data when PORTFOLIO_DATA is set (never overwrite V100 results)
FIGDIR = DATA if os.environ.get("PORTFOLIO_DATA") else HERE

# fp16 arch facts from results/mm_*_info.json
ARCH = {
    "facebook/opt-1.3b":               dict(weight_bytes=2_631_516_160,  kv_bytes_per_token=196_608),
    "microsoft/Phi-3-mini-4k-instruct":dict(weight_bytes=7_642_159_104,  kv_bytes_per_token=393_216),
    "Qwen/Qwen2.5-1.5B-Instruct":      dict(weight_bytes=3_087_428_608,  kv_bytes_per_token=28_672),
    "Qwen/Qwen2.5-3B-Instruct":        dict(weight_bytes=6_171_877_376,  kv_bytes_per_token=36_864),
    "Qwen/Qwen2.5-7B-Instruct":        dict(weight_bytes=15_231_233_024, kv_bytes_per_token=57_344),
    "Qwen/Qwen3-4B-Instruct-2507":     dict(weight_bytes=8_044_936_192,  kv_bytes_per_token=147_456),
}
MODEL_COLOR = {
    "microsoft/Phi-3-mini-4k-instruct": "C0",
    "Qwen/Qwen2.5-7B-Instruct": "C3",
    "Qwen/Qwen2.5-3B-Instruct": "C2",
    "Qwen/Qwen2.5-1.5B-Instruct": "C4",
    "Qwen/Qwen3-4B-Instruct-2507": "C5",
}
def short(m): return m.split("/")[-1]


# ============================================================== IO
def read_csv(path, p_key="power_avg_w", t_key="throughput_tok_s"):
    if not os.path.exists(path):
        return np.array([]), np.array([]), None
    rows = [r for r in csv.DictReader(open(path)) if float(r[t_key]) > 0]
    # traffic-weighted effective context (v3 CSVs; None for older data without the column)
    ce = [float(r["ctx_eff"]) for r in rows if r.get("ctx_eff")]
    return (np.array([float(r[p_key]) for r in rows]),
            np.array([float(r[t_key]) for r in rows]),
            (sum(ce) / len(ce)) if ce else None)


# ============================================================== FIT
def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - yhat) ** 2) / ss if ss > 0 else float("nan")


def v2f_inverse(P0, kappa, rho, t_hi):
    Tg = np.linspace(1, t_hi, 8000)
    Pg = P0 + kappa * Tg * (1 + rho * Tg) ** 2
    return lambda P: np.interp(P, Pg, Tg)


def fit_v2f(P, T):
    best = None
    for P0 in np.linspace(0, P.min(), 60):
        for rho in np.linspace(0, 3e-3, 500):
            x = T * (1 + rho * T) ** 2
            kap = np.sum(x * (P - P0)) / np.sum(x * x)
            if kap > 0:
                e = np.sum((P0 + kap * x - P) ** 2)
                if best is None or e < best[0]:
                    best = (e, P0, rho, kap)
    _, P0, rho, kap = best
    return dict(P0=P0, kappa=kap, rho=rho)


def fit_prefill(P, T):
    pr = fit_v2f(P, T)
    f = v2f_inverse(pr["P0"], pr["kappa"], pr["rho"], T.max() * 2.0)
    pr["R2"] = r2(T, f(P))
    return f, pr


def fit_decode(P, T):
    """T(P)=min(T_{V2f}(P), T_max). Plateau = mean of the 3 highest-power points (robust).
    Flat/already-saturated curves are drawn as a flat ceiling (no spurious rise)."""
    o = np.argsort(P); P, T = P[o], T[o]
    Tmax = float(np.mean(T[-3:]))
    if T[0] / Tmax > 0.72:                                   # saturated below the min cap
        f = lambda Q: np.full_like(np.asarray(Q, float), Tmax)
        return f, dict(P0=float(P.min()), kappa=0.0, rho=0.0, T_max=Tmax,
                       R2=r2(T, f(P)), saturated=True)
    keep = T >= 0.25 * Tmax
    pr = fit_v2f(P[keep], T[keep])
    core = v2f_inverse(pr["P0"], pr["kappa"], pr["rho"], max(T[keep].max() * 2.0, Tmax * 1.5))
    f = lambda Q: np.minimum(core(Q), Tmax)
    pr["T_max"] = Tmax; pr["R2"] = r2(T, f(P)); pr["saturated"] = False
    return f, pr


# ============================================================== main
def main():
    fits = []
    for w in PORTFOLIO:
        Pp, Tp, _ = read_csv(os.path.join(DATA, f"{w['id']}_prefill.csv"))
        Pd, Td, ctx_eff = read_csv(os.path.join(DATA, f"{w['id']}_decode.csv"))
        rec = dict(w=w, Pp=Pp, Tp=Tp, Pd=Pd, Td=Td, pre=None, dec=None, preF=None, decF=None)
        if len(Pp) >= 3:
            rec["preF"], rec["pre"] = fit_prefill(Pp, Tp)
        if len(Pd) >= 3:
            rec["decF"], rec["dec"] = fit_decode(Pd, Td)
        a = ARCH.get(w["model_id"])
        if a and rec["dec"]:
            C_use = ctx_eff if ctx_eff else w["decode_ctx"]     # drift-corrected context (v3)
            Dmem = a["weight_bytes"] + w["decode_batch"] * C_use * a["kv_bytes_per_token"]
            rec["Dmem"] = Dmem
            rec["B_over_Dmem"] = w["decode_batch"] / Dmem
            rec["BW_impl"] = rec["dec"]["T_max"] * Dmem / w["decode_batch"]   # implied effective BW (B/s)
        fits.append(rec)

    # calibrate BW_eff PER MODEL: T_max ~= BW_eff_model * (B/D_mem)
    per_model = {}
    for f in fits:
        if "B_over_Dmem" in f:
            per_model.setdefault(f["w"]["model_id"], []).append(f)
    bw_model = {}
    for m, fs in per_model.items():
        x = np.array([f["B_over_Dmem"] for f in fs]); y = np.array([f["dec"]["T_max"] for f in fs])
        bw_model[m] = float(np.sum(x * y) / np.sum(x * x))
    for f in fits:
        if "B_over_Dmem" in f:
            f["Tmax_pred"] = bw_model[f["w"]["model_id"]] * f["B_over_Dmem"]
    print("per-model effective BW (GB/s):",
          {short(m): round(bw / 1e9) for m, bw in bw_model.items()})

    _grid(fits)
    _tmax_scatter(fits, bw_model)
    _table(fits, bw_model)


# ---------- small-multiples grid ----------
def _grid(fits):
    n = len(fits)
    fig, ax = plt.subplots(n, 2, figsize=(11, 2.6 * n), squeeze=False)
    for i, f in enumerate(fits):
        w = f["w"]
        a = ax[i, 0]
        if len(f["Pp"]):
            a.scatter(f["Pp"], f["Tp"], c="C1", s=38, ec="k", lw=.4, zorder=5)
        if f["preF"] is not None:
            g = np.linspace(f["Pp"].min() * .97, f["Pp"].max() * 1.02, 200)
            a.plot(g, f["preF"](g), "k-", lw=1.8, label=f"V²f  R²={f['pre']['R2']:.3f}")
            a.legend(loc="lower right", fontsize=8)
        a.set_ylabel(f"{w['id']}\ntok/s", fontsize=9); a.grid(alpha=.3)
        if i == 0:
            a.set_title("PREFILL · throughput vs power", fontsize=11)

        a = ax[i, 1]
        if len(f["Pd"]):
            a.scatter(f["Pd"], f["Td"], c="C0", s=38, ec="k", lw=.4, zorder=5)
        if f["decF"] is not None:
            g = np.linspace(f["Pd"].min() * .97, f["Pd"].max() * 1.02, 200)
            sat = " · saturated≤floor" if f["dec"].get("saturated") else ""
            lbl = (f"min(V²f,T_max){sat}" if f["dec"].get("saturated")
                   else f"min(V²f,T_max)  R²={f['dec']['R2']:.3f}")
            a.plot(g, f["decF"](g), "k-", lw=1.8, label=lbl)
            a.axhline(f["dec"]["T_max"], color="gray", ls=":", lw=1.1, label=f"T_max={f['dec']['T_max']:.0f}")
            a.legend(loc="lower right", fontsize=8)
        a.grid(alpha=.3)
        if i == 0:
            a.set_title("DECODE · throughput vs power", fontsize=11)
    ax[n - 1, 0].set_xlabel("power (W)"); ax[n - 1, 1].set_xlabel("power (W)")
    fig.suptitle("Power-cap P↔T model vs measured across 8 workload types  "
                 "(prefill V²f · decode piecewise bandwidth ceiling)", fontsize=12, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.992))
    out = os.path.join(FIGDIR, "fig_portfolio_grid.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); print("wrote", out)


# ---------- fitted vs predicted decode T_max (per-model BW_eff) ----------
def _tmax_scatter(fits, bw_model):
    pts = [f for f in fits if f.get("dec") and "Tmax_pred" in f]
    if not pts:
        return
    fig, a = plt.subplots(figsize=(7, 6.4))
    lim = max(max(f["Tmax_pred"] for f in pts), max(f["dec"]["T_max"] for f in pts)) * 1.12
    a.plot([0, lim], [0, lim], "k--", lw=1, alpha=.6, label="y = x")
    seen = set()
    for f in pts:
        m = f["w"]["model_id"]
        lab = f"{short(m)}  (BW_eff≈{bw_model[m]/1e9:.0f} GB/s)" if m not in seen else None
        seen.add(m)
        a.scatter(f["Tmax_pred"], f["dec"]["T_max"], s=80, c=MODEL_COLOR.get(m, "C7"),
                  ec="k", zorder=5, label=lab)
        a.annotate(f["w"]["id"], (f["Tmax_pred"], f["dec"]["T_max"]),
                   fontsize=7, xytext=(4, 3), textcoords="offset points")
    a.set(xlabel="predicted T_max = B·BW_eff / (weights + B·C·kv/tok)   [tok/s]",
          ylabel="fitted plateau T_max from cap sweep   [tok/s]",
          title="Decode bandwidth ceiling: theory vs measured\n"
                "(BW_eff calibrated per model; log-log to span the 150× range)")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(3, lim); a.set_ylim(3, lim); a.grid(alpha=.3, which="both"); a.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_tmax_validation.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)


def _table(fits, bw_model):
    out = os.path.join(FIGDIR, "portfolio_fits.csv")
    keys = ["id", "application", "model_id", "decode_ctx", "decode_batch",
            "pre_R2", "dec_R2", "dec_saturated", "dec_Tmax_fit", "dec_Tmax_pred",
            "Dmem_GB", "BW_impl_GBs"]
    with open(out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys); wtr.writeheader()
        for f in fits:
            w = f["w"]
            row = {k: w.get(k) for k in ("id", "application", "model_id", "decode_ctx", "decode_batch")}
            if f["pre"]:
                row["pre_R2"] = round(f["pre"]["R2"], 3)
            if f["dec"]:
                row["dec_R2"] = round(f["dec"]["R2"], 3)
                row["dec_saturated"] = f["dec"].get("saturated", False)
                row["dec_Tmax_fit"] = round(f["dec"]["T_max"], 0)
            if "Tmax_pred" in f:
                row["dec_Tmax_pred"] = round(f["Tmax_pred"], 0)
            if "Dmem" in f:
                row["Dmem_GB"] = round(f["Dmem"] / 1e9, 2)
            if "BW_impl" in f:
                row["BW_impl_GBs"] = round(f["BW_impl"] / 1e9, 0)
            wtr.writerow(row)
    print("wrote", out)


if __name__ == "__main__":
    main()
