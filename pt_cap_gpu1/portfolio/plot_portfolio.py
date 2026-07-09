"""Portfolio overview under the UNIFIED explicit model -- every measured point vs the theory.

One law for both phases (fitlib.py is the canonical implementation; theory in
../prefill_model_theory.md and ../decode_model_theory.md):

  PREFILL  T(P) = T_fmax * x(P)^p                       single concave power law (T_mem -> 0)
  DECODE   T(P) = B / (T_mem + Cc*(x(P)^-p - 1))        3 stages -> bandwidth plateau B/T_mem
  with     x(P) = ((P-P_s)/chi)^(1/theta) clamped <= 1  (same DVFS power model in both)

Ceiling validation: plateau T_max = B * BW_eff / D_mem, D_mem = weights + B*C_eff*kv/token,
BW_eff calibrated PER MODEL (it is a model x hardware x engine property); C_eff is the
drift-corrected effective context recorded by the v3 methodology.

Outputs (into PORTFOLIO_DATA dir when set, e.g. data_h200/):
  fig_portfolio_grid.png     10 rows x (prefill | decode): measured points + unified-model curves
  fig_tmax_validation.png    fitted plateau vs predicted ceiling, per-model BW_eff, log-log
  portfolio_fits.csv         per-workload unified-fit params + R^2 + ceiling numbers
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from portfolio import PORTFOLIO
import fitlib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PORTFOLIO_DATA", os.path.join(HERE, "data"))   # e.g. data_h200 / data_v1
if not os.path.isabs(DATA):
    DATA = os.path.join(HERE, DATA)
# figures/CSVs land next to the data when PORTFOLIO_DATA is set (never overwrite V100 results)
FIGDIR = DATA if os.environ.get("PORTFOLIO_DATA") else HERE
F_MAX = fitlib.resolve_f_max(DATA)

# fp16 arch facts (weights measured from checkpoints; kv = 2*n_kv*h*2B*L)
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
def read_csv(path):
    """-> P, T, F(sm_clk), mean ctx_eff (None when absent: pre-v3 data)."""
    if not os.path.exists(path):
        return (np.array([]),) * 3 + (None,)
    rows = [r for r in csv.DictReader(open(path)) if float(r["throughput_tok_s"]) > 0]
    ce = [float(r["ctx_eff"]) for r in rows if r.get("ctx_eff")]
    return (np.array([float(r["power_avg_w"]) for r in rows]),
            np.array([float(r["throughput_tok_s"]) for r in rows]),
            np.array([float(r["sm_clk_avg"]) for r in rows]),
            (sum(ce) / len(ce)) if ce else None)


# ============================================================== main
def main():
    fits = []
    for w in PORTFOLIO:
        Pp, Tp, Fp, _ = read_csv(os.path.join(DATA, f"{w['id']}_prefill.csv"))
        Pd, Td, Fd, ctx_eff = read_csv(os.path.join(DATA, f"{w['id']}_decode.csv"))
        rec = dict(w=w, Pp=Pp, Tp=Tp, Pd=Pd, Td=Td, pre=None, dec=None, preF=None, decF=None)
        if len(Pp) >= 4:
            rec["preF"], rec["pre"] = fitlib.fit_prefill_unified(Pp, Tp, Fp, F_MAX)
        if len(Pd) >= 4:
            rec["decF"], rec["dec"] = fitlib.fit_decode_additive(Pd, Td, Fd, w["decode_batch"], F_MAX)
        a = ARCH.get(w["model_id"])
        if a and rec["dec"]:
            C_use = ctx_eff if ctx_eff else w["decode_ctx"]     # drift-corrected context (v3)
            Dmem = a["weight_bytes"] + w["decode_batch"] * C_use * a["kv_bytes_per_token"]
            rec["Dmem"] = Dmem
            rec["B_over_Dmem"] = w["decode_batch"] / Dmem
            rec["BW_impl"] = rec["dec"]["T_max"] * Dmem / w["decode_batch"]
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


# ---------- small-multiples grid: unified model vs every measured point ----------
def _grid(fits):
    n = len(fits)
    fig, ax = plt.subplots(n, 2, figsize=(11.5, 2.6 * n), squeeze=False)
    for i, f in enumerate(fits):
        w = f["w"]
        # ---- prefill: single power law ----
        a = ax[i, 0]
        if len(f["Pp"]):
            a.scatter(f["Pp"], f["Tp"], c="C1", s=38, ec="k", lw=.4, zorder=5)
        if f["preF"] is not None:
            g = np.linspace(f["Pp"].min() * .97, f["Pp"].max() * 1.02, 240)
            a.plot(g, f["preF"](g), "k-", lw=1.8,
                   label=f"$T_{{fmax}}x(P)^p$  R²={f['pre']['R2']:.2f} · p={f['pre']['p']:.2f}")
            a.legend(loc="lower right", fontsize=7.5)
        a.set_ylabel(f"{w['id']}\ntok/s", fontsize=9)
        a.grid(alpha=.3)
        if i == 0:
            a.set_title("PREFILL · unified single power law", fontsize=11)

        # ---- decode: additive three-stage ----
        a = ax[i, 1]
        if len(f["Pd"]):
            a.scatter(f["Pd"], f["Td"], c="C0", s=38, ec="k", lw=.4, zorder=5)
        if f["decF"] is not None:
            d = f["dec"]
            g = np.linspace(f["Pd"].min() * .97, f["Pd"].max() * 1.02, 240)
            a.plot(g, f["decF"](g), "k-", lw=1.8,
                   label=f"additive 3-stage  R²={d['R2']:.2f}")
            a.axhline(d["T_max"], color="gray", ls=":", lw=1.1,
                      label=f"plateau $B/T_{{mem}}$={d['T_max']:.0f}")
            if not d["edge"] and np.isfinite(d["P2"]) and g[0] < d["P2"] < g[-1]:
                a.axvline(d["P2"], color="C3", ls=":", lw=1.0)
                a.text(d["P2"], a.get_ylim()[0], " II|III", color="C3", fontsize=7, va="bottom")
            a.legend(loc="lower right", fontsize=7.5)
        a.grid(alpha=.3)
        if i == 0:
            a.set_title("DECODE · unified additive three-stage", fontsize=11)
    ax[n - 1, 0].set_xlabel("power (W)"); ax[n - 1, 1].set_xlabel("power (W)")
    fig.suptitle("Unified explicit P↔T law vs ALL measured points — 10 workload types × 5 models\n"
                 r"one law $t=T_{mem}+C\,x^{-p}$, $P=P_s+\chi x^{\theta}$:  prefill = $T_{mem}\!\to\!0$ "
                 r"(single power law) · decode = $T_{mem}$ floor (3 stages → plateau)",
                 fontsize=12, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = os.path.join(FIGDIR, "fig_portfolio_grid.png")
    fig.savefig(out, dpi=120, bbox_inches="tight"); print("wrote", out)


# ---------- fitted vs predicted decode ceiling (per-model BW_eff) ----------
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
    a.set(xlabel="predicted plateau = B·BW_eff / (weights + B·C_eff·kv/tok)   [tok/s]",
          ylabel="fitted plateau  $B/T_{mem}$  from the unified decode fit   [tok/s]",
          title="Decode bandwidth ceiling: theory vs measured\n"
                "(BW_eff calibrated per model; log-log spans the ~140× range)")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(3, lim); a.set_ylim(3, lim); a.grid(alpha=.3, which="both")
    a.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_tmax_validation.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)


def _table(fits, bw_model):
    out = os.path.join(FIGDIR, "portfolio_fits.csv")
    keys = ["id", "application", "model_id", "decode_ctx", "decode_batch",
            "pre_R2", "pre_p", "pre_relRMSE_pct",
            "dec_R2", "dec_relRMSE_pct", "dec_Tmem_ms", "dec_Cc_ms", "dec_p", "dec_P2_W",
            "dec_exponents_railed", "dec_Tmax_fit", "dec_Tmax_pred", "Dmem_GB", "BW_impl_GBs"]
    with open(out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys); wtr.writeheader()
        for f in fits:
            w = f["w"]
            row = {k: w.get(k) for k in ("id", "application", "model_id", "decode_ctx", "decode_batch")}
            if f["pre"]:
                row |= {"pre_R2": round(f["pre"]["R2"], 3), "pre_p": round(f["pre"]["p"], 2),
                        "pre_relRMSE_pct": round(f["pre"]["relRMSE"], 1)}
            if f["dec"]:
                d = f["dec"]
                row |= {"dec_R2": round(d["R2"], 3), "dec_relRMSE_pct": round(d["relRMSE"], 1),
                        "dec_Tmem_ms": round(d["T_mem_ms"], 2), "dec_Cc_ms": round(d["Cc_ms"], 3),
                        "dec_p": round(d["p"], 2), "dec_exponents_railed": d["edge"],
                        "dec_Tmax_fit": round(d["T_max"], 1)}
                if not d["edge"] and np.isfinite(d["P2"]):
                    row["dec_P2_W"] = round(d["P2"], 1)
            if "Tmax_pred" in f:
                row["dec_Tmax_pred"] = round(f["Tmax_pred"], 1)
            if "Dmem" in f:
                row["Dmem_GB"] = round(f["Dmem"] / 1e9, 2)
            if "BW_impl" in f:
                row["BW_impl_GBs"] = round(f["BW_impl"] / 1e9, 0)
            wtr.writerow(row)
    print("wrote", out)


if __name__ == "__main__":
    main()
