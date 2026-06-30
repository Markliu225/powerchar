"""Decode at FIXED full batch, SUSTAINED power swept via SM frequency: throughput + efficiency vs power.

x-axis = sustained (active) power = median of in-window NVML samples while decoding (what nvidia-smi
shows / what the power wall limits), NOT the window mean (which the eager idle-gaps under-read).
Each point = one even power target hit by tuning the SM clock (HBM clock fixed 877 MHz).
Data: decode_fixedbatch.csv (code/decode_powercap_sweep.py).
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "decode_fixedbatch.csv"))))
g = lambda k: np.array([float(r[k]) for r in rows])
PCOL = "power_w" if "power_w" in rows[0] else "power_avg_w"
P, T, clk = g(PCOL), g("throughput_tok_s"), g("sm_clk_avg")
E = T / P
o = np.argsort(P); P, T, clk, E = P[o], T[o], clk[o], E[o]
batch = rows[0]["batch"]
Tmax = T.max()
knee = float(P[T >= 0.97 * Tmax][0])
ie = int(np.argmax(E))

fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))

# (a) throughput vs power
a = ax[0]
sc = a.scatter(P, T, c=clk, cmap="viridis", s=70, edgecolor="k", lw=.5, zorder=5)
a.plot(P, T, "-", color="#888", lw=1.2, zorder=4)
a.plot(P, T[0] * P / P[0], "r--", lw=1.4, label="if compute-bound (T ∝ power)")
a.axvline(knee, color="#cc6600", ls=":", lw=1.3, label=f"bandwidth saturation ~{knee:.0f} W")
cb = fig.colorbar(sc, ax=a); cb.set_label("locked SM clock (MHz)")
a.set_xlabel("sustained power (W)"); a.set_ylabel("decode throughput (tok/s)")
a.set_title(f"DECODE throughput vs sustained power  (fixed batch={batch})\n"
            f"compute/clock-limited rise -> bandwidth ceiling ~{Tmax:.0f} tok/s above ~{knee:.0f} W")
a.legend(loc="lower right", fontsize=9); a.grid(alpha=.3); a.set_ylim(0, Tmax * 1.18); a.set_xlim(0, P.max() * 1.05)

# (b) efficiency vs power
a = ax[1]
a.scatter(P, E, c=clk, cmap="viridis", s=70, edgecolor="k", lw=.5, zorder=5)
a.plot(P, E, "-", color="#888", lw=1.2, zorder=4)
a.plot(P[ie], E[ie], "*", color="#cc6600", ms=18, zorder=6)
a.annotate(f"sweet spot ~{P[ie]:.0f} W\n{E[ie]:.1f} tok/J", (P[ie], E[ie]),
           textcoords="offset points", xytext=(12, -28), fontsize=9.5, color="#cc6600", weight="bold",
           arrowprops=dict(arrowstyle="->", color="#cc6600"))
a.set_xlabel("sustained power (W)"); a.set_ylabel("efficiency (tok/J)")
a.set_title("DECODE efficiency vs sustained power\n(tok/J peaks between the low-power floor and bandwidth saturation)")
a.grid(alpha=.3); a.set_ylim(0, max(E) * 1.18); a.set_xlim(0, P.max() * 1.05)

fig.suptitle("Phi-3-mini on V100 — decode at fixed full batch, sustained power swept via SM clock "
             f"(batch={batch}, ctx=256, HBM fixed 877 MHz)", fontsize=12)
fig.tight_layout()
out = os.path.join(HERE, "fig_decode_fixedbatch.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
print(f"sustained power {P[0]:.0f}->{P[-1]:.0f} W | throughput {T[0]:.0f}->{Tmax:.0f} tok/s | "
      f"saturates ~{knee:.0f} W | efficiency peak {E[ie]:.1f} tok/J @ {P[ie]:.0f} W")
