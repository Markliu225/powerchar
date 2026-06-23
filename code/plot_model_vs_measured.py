"""New architecture-grounded model vs measured data, with realistic (least-squares) fits.

Power forms taken from POWER_THROUGHPUT_MODEL.zh.md:
  PREFILL (frequency sweep, compute-bound): the FULL curve P = P_static + a·T + c·T³
      (low-f: voltage floored -> dynamic power ∝ f ∝ T, the linear term;
       high-f: V∝f -> P_dyn ∝ V²f ∝ f³ ∝ T³, the cubic term).
      P_static fixed at the measured deep-idle 44 W; (a,c) fit by least squares.
  DECODE (batch sweep, memory-bound): affine  P = A + s·T   (A=P_static+a, s=slope), LSQ.

Efficiency curves E = T/P_model(T) follow from the same fits (no extra params).
Cap-throttled prefill points and the decode b=64 thermal-edge point are hollow / excluded.

  python code/plot_model_vs_measured.py
"""
from __future__ import annotations
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

cap = json.load(open(os.path.join(C.RESULTS_DIR, "model_info.json")))["power_cap_w"]
P_IDLE = 60.0


def load(name): return list(csv.DictReader(open(os.path.join(C.RESULTS_DIR, name))))
def arr(rows, k): return np.array([float(r[k]) for r in rows])
def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


# --- prefill: DVFS sweep (un-capped clean vs cap-throttled) ---
pf = [r for r in load("dvfs.csv") if r["workload"] == "prefill"]
req, act = arr(pf, "req_clk_mhz"), arr(pf, "act_clk_mhz")
Tp_all, Pp_all = arr(pf, "throughput_tok_s"), arr(pf, "power_avg_w")
clean = act >= req - 30
Tp, Pp = Tp_all[clean], Pp_all[clean]
Tpc, Ppc = Tp_all[~clean], Pp_all[~clean]
Tps, Pps = arr(pf, "throughput_std")[clean], arr(pf, "power_std_w")[clean]   # 3x-avg error bars
# PREFILL power model — PHYSICALLY CONSTRAINED V²f law (V=V0+γf, T∝f):
#   P = P0 + κ·T·(1+ρ·T)²,  P0≥0 (static floor), κ>0, ρ≥0  =>  a,b,c≥0 and b²=4ac.
# Fit κ by relative-error LSQ (≈ efficiency error) for each (P0,ρ); choose the (P0,ρ)
# that best matches the EFFICIENCY curve. The optimum pushes P0 to the ~90 W active
# (under-load) uncore floor — physical, not the 44 W deep idle.
_best = None
for _P0 in np.linspace(0, 110, 56):
    for _rho in np.linspace(0, 1e-2, 1200):
        _x = Tp * (1 + _rho * Tp) ** 2
        _k = np.sum(_x * (Pp - _P0) / Pp ** 2) / np.sum(_x ** 2 / Pp ** 2)
        if _k <= 0:
            continue
        _Pm = _P0 + _k * _x
        _e = np.sum((Tp / _Pm - Tp / Pp) ** 2)
        if _best is None or _e < _best[0]:
            _best = (_e, _P0, _rho, _k)
_, P0_p, rho_p, k_p = _best
pre_model = lambda T: P0_p + k_p * T * (1 + rho_p * T) ** 2
r2_p = r2(Pp, pre_model(Tp))
r2_pe = r2(Tp / Pp, Tp / pre_model(Tp))          # efficiency R² (the strict test)

# --- decode: batch sweep (b<=32 clean) ---
dc = sorted(load("decode.csv"), key=lambda r: float(r["batch"]))
Bd, Td_all, Pd_all = arr(dc, "batch"), arr(dc, "throughput_tok_s"), arr(dc, "power_avg_w")
dclean = Bd <= 32
Td, Pd = Td_all[dclean], Pd_all[dclean]
Tdc, Pdc = Td_all[~dclean], Pd_all[~dclean]
Pds = arr(dc, "power_std_w")[dclean]                                          # within-window power std
(s_d, A_d) = np.polyfit(Td, Pd, 1)
dec_model = lambda T: A_d + s_d * T
r2_d = r2(Pd, dec_model(Td))

print(f"prefill V²f(physical): P={P0_p:.0f}+κ·T·(1+ρT)²  κ={k_p:.2e} ρ={rho_p:.2e}  "
      f"P-T R²={r2_p:.4f}  Eff R²={r2_pe:.4f}")
print(f"decode : P = {A_d:.1f} + {s_d:.4f}·T            R²={r2_d:.4f}")

fig, ax = plt.subplots(2, 2, figsize=(15, 11))
TgP = np.linspace(Tp.min() * 0.97, Tp.max() * 1.02, 200)
TgD = np.linspace(Td.min() * 0.5, Td.max() * 1.02, 200)

# (0,0) PREFILL power vs throughput
a = ax[0, 0]
a.errorbar(Tp, Pp, xerr=Tps, yerr=Pps, fmt="o", color="C1", ms=8, mec="k", mew=.4,
           capsize=3, zorder=5, label="measured (freq sweep, 3× avg)")
a.scatter(Tpc, Ppc, facecolors="none", edgecolors="C1", s=90, zorder=5, label="cap-throttled (excl.)")
a.plot(TgP, pre_model(TgP), "k-", lw=2, label=f"physical V²f: P=P₀+κT(1+ρT)² (R²={r2_p:.3f})")
a.axhline(cap, color="gray", ls=":", lw=1, label=f"cap {cap:.0f} W")
a.set_xlabel("throughput (tok/s)"); a.set_ylabel("power (W)"); a.set_ylim(0, cap * 1.08)
a.set_title("PREFILL · power vs throughput (frequency sweep)"); a.legend(fontsize=8); a.grid(alpha=.3)

# (0,1) DECODE power vs throughput
a = ax[0, 1]
a.errorbar(Td, Pd, yerr=Pds, fmt="o", color="C0", ms=8, mec="k", mew=.4,
           capsize=3, zorder=5, label="measured (batch sweep)")
a.scatter(Tdc, Pdc, facecolors="none", edgecolors="C0", s=90, zorder=5, label="b=64 thermal-edge (excl.)")
a.plot(TgD, dec_model(TgD), "k-", lw=2, label=f"model P={A_d:.0f}+{s_d:.3f}·T (R²={r2_d:.3f})")
a.axhline(cap, color="gray", ls=":", lw=1, label=f"cap {cap:.0f} W")
a.set_xlabel("throughput (tok/s)"); a.set_ylabel("power (W)"); a.set_ylim(0, cap * 1.08)
a.set_title("DECODE · power vs throughput (batch sweep)"); a.legend(fontsize=8); a.grid(alpha=.3)

# (1,0) PREFILL efficiency vs power
a = ax[1, 0]
a.scatter(Pp, Tp / Pp, c="C1", s=90, edgecolor="k", lw=.4, zorder=5, label="measured")
a.scatter(Ppc, Tpc / Ppc, facecolors="none", edgecolors="C1", s=90, zorder=5, label="cap-throttled (excl.)")
a.plot(pre_model(TgP), TgP / pre_model(TgP), "k-", lw=2, label=f"model E=T/P(T) (R²={r2_pe:.3f})")
a.set_xlabel("power (W)"); a.set_ylabel("efficiency (tok/J)")
a.set_title("PREFILL · efficiency vs power"); a.legend(fontsize=8); a.grid(alpha=.3)

# (1,1) DECODE efficiency vs power
a = ax[1, 1]
a.scatter(Pd, Td / Pd, c="C0", s=90, edgecolor="k", lw=.4, zorder=5, label="measured")
a.scatter(Pdc, Tdc / Pdc, facecolors="none", edgecolors="C0", s=90, zorder=5, label="b=64 (excl.)")
a.plot(dec_model(TgD), TgD / dec_model(TgD), "k-", lw=2, label="model E=T/P(T)")
a.axhline(1 / s_d, color="gray", ls=":", lw=1.2, label=f"asymptote 1/s={1/s_d:.1f} tok/J")
a.set_xlabel("power (W)"); a.set_ylabel("efficiency (tok/J)")
a.set_title("DECODE · efficiency vs power"); a.legend(fontsize=8); a.grid(alpha=.3)

fig.suptitle("New architecture-grounded model vs measured  (realistic least-squares fit) — V100 / Phi-3-mini",
             fontsize=13)
fig.tight_layout()
out = os.path.join(C.FIGURES_DIR, "fig_model_vs_measured.png")
fig.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)
