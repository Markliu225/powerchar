"""Standalone reproduction of fig_model_vs_measured.png from the bundled data.

Reads ../data/{prefill_freq_sweep.csv, decode_batch_sweep.csv, model_info.json},
fits the architecture-grounded P(T) model, and writes ../fig_model_vs_measured.png.

  python code/plot.py        # from inside fig_model_vs_measured_bundle/

Models (see THEORY.*.md):
  PREFILL (frequency sweep, compute-bound): P(T) = P0 + κ·T·(1+ρ·T)²   (the V²·f law with
      V=V0+γf, T∝f; coefficients tied by b²=4ac). Fit on un-throttled points (act≈req),
      choosing (P0,κ,ρ) that best match the EFFICIENCY curve (relative-power error).
  DECODE (batch sweep, memory-bound):      P(T) = a + s·T   (affine), fit on b≤32.
"""
from __future__ import annotations
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "fig_model_vs_measured.png")


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


def arr(rows, k):
    return np.array([float(r[k]) for r in rows])


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


info = json.load(open(os.path.join(DATA, "model_info.json")))
cap = info["power_cap_w"]

# ---- PREFILL: frequency sweep; clean = clock held (act >= req-30) ----
pf = [r for r in load("prefill_freq_sweep.csv") if r["workload"] == "prefill"]
req, act = arr(pf, "req_clk_mhz"), arr(pf, "act_clk_mhz")
Tp_all, Pp_all = arr(pf, "throughput_tok_s"), arr(pf, "power_avg_w")
Ts_all, Ps_all = arr(pf, "throughput_std"), arr(pf, "power_std_w")
clean = act >= req - 30
Tp, Pp = Tp_all[clean], Pp_all[clean]
Tpc, Ppc = Tp_all[~clean], Pp_all[~clean]
Tps, Pps = Ts_all[clean], Ps_all[clean]
best = None
for P0 in np.linspace(0, 110, 56):
    for rho in np.linspace(0, 1e-2, 1200):
        x = Tp * (1 + rho * Tp) ** 2
        k = np.sum(x * (Pp - P0) / Pp ** 2) / np.sum(x ** 2 / Pp ** 2)
        if k <= 0:
            continue
        Pm = P0 + k * x
        e = np.sum((Tp / Pm - Tp / Pp) ** 2)
        if best is None or e < best[0]:
            best = (e, P0, rho, k)
_, P0_p, rho_p, k_p = best
pre_model = lambda T: P0_p + k_p * T * (1 + rho_p * T) ** 2
r2_p = r2(Pp, pre_model(Tp)); r2_pe = r2(Tp / Pp, Tp / pre_model(Tp))

# ---- DECODE: batch sweep; clean = b<=32 (b=64 is thermal-edge) ----
dc = sorted(load("decode_batch_sweep.csv"), key=lambda r: float(r["batch"]))
Bd, Td_all, Pd_all = arr(dc, "batch"), arr(dc, "throughput_tok_s"), arr(dc, "power_avg_w")
dm = Bd <= 32
Td, Pd = Td_all[dm], Pd_all[dm]
Tdc, Pdc = Td_all[~dm], Pd_all[~dm]
Pds = arr(dc, "power_std_w")[dm]
s_d, A_d = np.polyfit(Td, Pd, 1)
dec_model = lambda T: A_d + s_d * T
r2_d = r2(Pd, dec_model(Td))

print(f"prefill V²f: P={P0_p:.0f}+κ·T·(1+ρT)²  κ={k_p:.3e} ρ={rho_p:.3e}  P-T R²={r2_p:.3f}  Eff R²={r2_pe:.3f}")
print(f"decode affine: P={A_d:.1f}+{s_d:.4f}·T  R²={r2_d:.3f}  asymptote 1/s={1/s_d:.1f} tok/J")

fig, ax = plt.subplots(2, 2, figsize=(15, 11))
TgP = np.linspace(Tp.min() * 0.97, Tp.max() * 1.02, 200)
TgD = np.linspace(Td.min() * 0.5, Td.max() * 1.02, 200)

a = ax[0, 0]
a.errorbar(Tp, Pp, xerr=Tps, yerr=Pps, fmt="o", color="C1", ms=8, mec="k", mew=.4, capsize=3, zorder=5,
           label="measured (freq sweep, 3× avg)")
a.scatter(Tpc, Ppc, facecolors="none", edgecolors="C1", s=90, zorder=5, label="cap-throttled (excl.)")
a.plot(TgP, pre_model(TgP), "k-", lw=2, label=f"physical V²f: P=P₀+κT(1+ρT)² (R²={r2_p:.3f})")
a.axhline(cap, color="r", ls="--", alpha=.5, label=f"cap {cap:.0f} W")
a.set_xlabel("throughput (tok/s)"); a.set_ylabel("power (W)"); a.set_ylim(0, cap * 1.08)
a.set_title("PREFILL · power vs throughput (frequency sweep)"); a.legend(fontsize=8); a.grid(alpha=.3)

a = ax[0, 1]
a.errorbar(Td, Pd, yerr=Pds, fmt="o", color="C0", ms=8, mec="k", mew=.4, capsize=3, zorder=5,
           label="measured (batch sweep)")
a.scatter(Tdc, Pdc, facecolors="none", edgecolors="C0", s=90, zorder=5, label="b=64 thermal-edge (excl.)")
a.plot(TgD, dec_model(TgD), "k-", lw=2, label=f"model P={A_d:.0f}+{s_d:.3f}·T (R²={r2_d:.3f})")
a.axhline(cap, color="r", ls="--", alpha=.5, label=f"cap {cap:.0f} W")
a.set_xlabel("throughput (tok/s)"); a.set_ylabel("power (W)"); a.set_ylim(0, cap * 1.08)
a.set_title("DECODE · power vs throughput (batch sweep)"); a.legend(fontsize=8); a.grid(alpha=.3)

a = ax[1, 0]
a.scatter(Pp, Tp / Pp, c="C1", s=90, edgecolor="k", lw=.4, zorder=5, label="measured")
a.scatter(Ppc, Tpc / Ppc, facecolors="none", edgecolors="C1", s=90, zorder=5, label="cap-throttled (excl.)")
a.plot(pre_model(TgP), TgP / pre_model(TgP), "k-", lw=2, label=f"model E=T/P(T) (R²={r2_pe:.3f})")
a.set_xlabel("power (W)"); a.set_ylabel("efficiency (tok/J)")
a.set_title("PREFILL · efficiency vs power"); a.legend(fontsize=8); a.grid(alpha=.3)

a = ax[1, 1]
a.scatter(Pd, Td / Pd, c="C0", s=90, edgecolor="k", lw=.4, zorder=5, label="measured")
a.scatter(Pdc, Tdc / Pdc, facecolors="none", edgecolors="C0", s=90, zorder=5, label="b=64 (excl.)")
a.plot(dec_model(TgD), TgD / dec_model(TgD), "k-", lw=2, label="model E=T/P(T)")
a.axhline(1 / s_d, color="gray", ls=":", lw=1.2, label=f"asymptote 1/s={1/s_d:.1f} tok/J")
a.set_xlabel("power (W)"); a.set_ylabel("efficiency (tok/J)")
a.set_title("DECODE · efficiency vs power"); a.legend(fontsize=8); a.grid(alpha=.3)

fig.suptitle("Phi-3-mini on V100 — architecture-grounded P(T) model vs measured", fontsize=13)
fig.tight_layout(); fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("wrote", OUT)
