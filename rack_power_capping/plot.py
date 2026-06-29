"""Optimal-vs-TDP across P:D ratios: GPU counts, per-GPU settings, throughput gain.

Per-GPU power is fixed by each policy (the ratio only changes the COUNTS):
  OPTIMAL : prefill @164 W, decode @169 W  (each phase at its efficiency sweet-spot)
  TDP     : prefill @250 W, decode @250 W  (nameplate full power)
Reads everything from solve.py (theoretical bandwidth-roofline decode, context solve.CTX).
  python3 plot.py   ->   fig_opt_vs_tdp.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import solve as S

HERE = os.path.dirname(os.path.abspath(__file__))
pP, _, _ = S.best_eff_prefill()
P_PRE_OPT, P_DEC_OPT = pP, S.P_DEC_OPT
P_TDP = 250.0
RATIOS = S.RATIOS
labels = [f"{p}:{d}" for p, d in RATIOS]
kW = S.W_RACK / 1000.0

opt = [S.solve(P, D, P_PRE_OPT, P_DEC_OPT) for P, D in RATIOS]
tdp = [S.solve(P, D, P_TDP, P_TDP) for P, D in RATIOS]
impr = [100 * (o["tot"] / t["tot"] - 1) for o, t in zip(opt, tdp)]

x = np.arange(len(RATIOS)); w = 0.38
fig, ax = plt.subplots(3, 1, figsize=(13, 14), gridspec_kw={"height_ratios": [1.1, 1, 1]})

# (0) throughput optimal vs TDP + improvement
a = ax[0]
a.bar(x - w / 2, [o["tot"] / 1000 for o in opt], w, color="#2ca02c",
      label=f"OPTIMAL (prefill@{P_PRE_OPT:.0f}W, decode@{P_DEC_OPT:.0f}W)")
a.bar(x + w / 2, [t["tot"] / 1000 for t in tdp], w, color="#d62728",
      label=f"TDP (prefill@{P_TDP:.0f}W, decode@{P_TDP:.0f}W)")
for i, im in enumerate(impr):
    a.annotate(f"+{im:.0f}%", (x[i], opt[i]["tot"] / 1000), textcoords="offset points",
               xytext=(0, 4), ha="center", fontsize=9, color="#2ca02c", weight="bold")
a.set_xticks(x); a.set_xticklabels(labels)
a.set_xlabel("prefill : decode token ratio   (decode-heavy ← → prefill-heavy)")
a.set_ylabel("rack throughput (k tok/s)")
a.set_title(f"Rack throughput — OPTIMAL vs TDP across P:D ratios   (C={S.CTX}, {kW:.0f} kW)\n"
            f"green = throughput gain of optimal over TDP")
a.legend(fontsize=9); a.grid(alpha=.3, axis="y")

# (1) OPTIMAL fleet composition
a = ax[1]
Np = [o["Np"] for o in opt]; Nd = [o["Nd"] for o in opt]
a.bar(x, Np, color="#1f77b4", label=f"prefill GPUs @{P_PRE_OPT:.0f}W")
a.bar(x, Nd, bottom=Np, color="#ff7f0e", label=f"decode GPUs @{P_DEC_OPT:.0f}W")
for i in range(len(x)):
    a.text(i, Np[i] + Nd[i] + max(Np[i]+Nd[i] for i in range(len(x))) * .02,
           f"{Np[i]:.0f}+{Nd[i]:.0f}\n={Np[i]+Nd[i]:.0f}", ha="center", fontsize=8)
a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel(f"GPUs in {kW:.0f} kW rack")
a.set_title(f"OPTIMAL fleet: how many GPUs in each phase (per-GPU {P_PRE_OPT:.0f}/{P_DEC_OPT:.0f} W)")
a.legend(fontsize=9); a.grid(alpha=.3, axis="y"); a.set_ylim(0, max(np.array(Np)+np.array(Nd)) * 1.20)

# (2) TDP fleet composition
a = ax[2]
Np2 = [t["Np"] for t in tdp]; Nd2 = [t["Nd"] for t in tdp]
a.bar(x, Np2, color="#1f77b4", alpha=.6, hatch="//", label=f"prefill GPUs @{P_TDP:.0f}W")
a.bar(x, Nd2, bottom=Np2, color="#ff7f0e", alpha=.6, hatch="//", label=f"decode GPUs @{P_TDP:.0f}W")
for i in range(len(x)):
    a.text(i, Np2[i] + Nd2[i] + max(Np2[i]+Nd2[i] for i in range(len(x))) * .02,
           f"{Np2[i]:.0f}+{Nd2[i]:.0f}\n={Np2[i]+Nd2[i]:.0f}", ha="center", fontsize=8)
a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel(f"GPUs in {kW:.0f} kW rack")
a.set_xlabel("prefill : decode token ratio")
a.set_title(f"TDP fleet: fewer GPUs at full power ({P_TDP:.0f}/{P_TDP:.0f} W) — same budget, less throughput")
a.legend(fontsize=9); a.grid(alpha=.3, axis="y"); a.set_ylim(0, max(np.array(Np2)+np.array(Nd2)) * 1.20)

fig.suptitle("Power-capping: OPTIMAL vs TDP — GPU counts, per-GPU power, throughput gain", fontsize=14)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_opt_vs_tdp.png"), dpi=130, bbox_inches="tight")
print("wrote fig_opt_vs_tdp.png")
