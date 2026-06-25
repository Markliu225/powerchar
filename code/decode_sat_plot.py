"""Diagnostic plot for the decode saturation sweeps (high-batch C=256/128 + long-context C=512/1024/2048).

Reads results/decode_saturation.csv and results/decode_saturation_long.csv. Four panels:
  (1) T vs batch (log-log)        — filled = clock held >=1200 MHz (clean), hollow = clock collapsed
  (2) per-seq rate T/B vs batch   — constant ~20 in the clean region (LINEAR, unsaturated)
  (3) sm clock vs batch           — the smoking gun: clock collapses at high batch despite -lgc lock
  (4) achieved bandwidth beta_eff — rises to ~190 GB/s (24% of peak) then FALLS (clock collapse, not a BW plateau)
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KV, W = 393216.0, 7.642e9
BETA_PEAK = 782.0

rows = []
for fn in ("decode_saturation.csv", "decode_saturation_long.csv"):
    p = os.path.join(ROOT, "results", fn)
    if os.path.exists(p):
        rows += list(csv.DictReader(open(p)))
ctxs = sorted({int(r["context"]) for r in rows})
cc = {128: "#9467bd", 256: "#1f77b4", 512: "#2ca02c", 1024: "#ff7f0e", 2048: "#d62728"}

fig, ax = plt.subplots(2, 2, figsize=(15, 11))
for C in ctxs:
    d = sorted([r for r in rows if int(r["context"]) == C], key=lambda r: float(r["batch"]))
    B = np.array([float(r["batch"]) for r in d]); T = np.array([float(r["throughput_tok_s"]) for r in d])
    clk = np.array([float(r["sm_clk_avg"]) for r in d]); beta = np.array([float(r["beta_eff_gbs"]) for r in d])
    held = clk >= 1200
    col = cc[C]; bstar = W / (C * KV)

    a = ax[0, 0]
    a.plot(B, T, "-", color=col, lw=1, alpha=.5)
    a.scatter(B[held], T[held], color=col, s=42, edgecolor="k", lw=.5, zorder=5, label=f"C={C} (clk≥1200)")
    a.scatter(B[~held], T[~held], facecolors="none", edgecolors=col, s=42, zorder=5)
    a.axvline(bstar, color=col, ls=":", lw=1, alpha=.5)

    ax[0, 1].plot(B, T / B, "o-", color=col, ms=4, lw=1.2, label=f"C={C}")
    ax[1, 0].plot(B, clk, "o-", color=col, ms=4, lw=1.2, label=f"C={C}")
    ax[1, 1].plot(B, beta, "o-", color=col, ms=4, lw=1.2, label=f"C={C}")

a = ax[0, 0]
a.set_xscale("log", base=2); a.set_yscale("log", base=2)
a.set_xlabel("batch"); a.set_ylabel("throughput (tok/s)")
a.set_title("Throughput vs batch — filled=clock held, hollow=clock collapsed\n(dotted = B*; clean region is LINEAR, peak coincides with clock collapse)")
a.legend(fontsize=8); a.grid(alpha=.3, which="both")

a = ax[0, 1]
a.set_xscale("log", base=2); a.set_xlabel("batch"); a.set_ylabel("per-seq rate T/B (tok/s/seq)")
a.axhline(20.5, color="gray", ls="--", alpha=.6, label="~20.5 (full-clock linear)")
a.set_title("Per-sequence rate: flat ~20 while clock held (linear/unsaturated),\nfalls only once the clock collapses"); a.legend(fontsize=8); a.grid(alpha=.3)

a = ax[1, 0]
a.set_xscale("log", base=2); a.set_xlabel("batch"); a.set_ylabel("avg SM clock (MHz)")
a.axhline(1290, color="gray", ls="--", alpha=.6, label="locked 1290 MHz")
a.set_title("SM clock COLLAPSES at high batch (despite -lgc 1290 lock)\n— this, not bandwidth, caps measured decode throughput")
a.legend(fontsize=8); a.grid(alpha=.3)

a = ax[1, 1]
a.set_xscale("log", base=2); a.set_xlabel("batch"); a.set_ylabel("achieved bandwidth β_eff (GB/s)")
a.axhline(BETA_PEAK, color="gray", ls="--", alpha=.6, label=f"peak HBM {BETA_PEAK:.0f}")
a.axhline(0.8 * BETA_PEAK, color="gray", ls=":", alpha=.5, label="80% peak (626)")
a.set_title("Achieved bandwidth peaks ~190 GB/s (24% of peak) then FALLS\n— never reaches the roofline; clock collapse pulls it back down")
a.legend(fontsize=8); a.grid(alpha=.3)

fig.suptitle("Decode high-batch sweep — throughput is capped by CLOCK COLLAPSE + memory capacity, not HBM bandwidth (V100, Phi-3, eager)", fontsize=12)
fig.tight_layout()
out = os.path.join(ROOT, "rack_power_capping", "fig_decode_saturation.png")
fig.savefig(out, dpi=130, bbox_inches="tight")

print("peak throughput & achieved BW per context (full-clock points only):")
for C in ctxs:
    d = [r for r in rows if int(r["context"]) == C]
    clean = [r for r in d if float(r["sm_clk_avg"]) >= 1200]
    bmax = max(clean, key=lambda r: float(r["beta_eff_gbs"]))
    pk = max(d, key=lambda r: float(r["throughput_tok_s"]))
    print(f"  C={C:>4}: peak T={float(pk['throughput_tok_s']):>5.0f}@b{pk['batch']:>3}(clk{pk['sm_clk_avg']}) | "
          f"max clean β_eff={float(bmax['beta_eff_gbs']):>5.0f} GB/s @b{bmax['batch']} = {100*float(bmax['beta_eff_gbs'])/BETA_PEAK:.0f}% peak")
print("wrote", out)
