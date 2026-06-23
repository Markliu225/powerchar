"""Measured data ONLY — no theory/fit curves.

One figure, two panels:
  left : GPU power (W) vs token throughput (tok/s)
  right: energy efficiency (tok/J) vs GPU power (W)
Prefill points come from the DVFS / frequency sweep (results/dvfs.csv); decode
points come from the batch sweep (results/decode.csv) — the experiment in which
each phase's P and T actually co-vary. Cap-clipped / thermal-edge points are
drawn hollow. Only the 250 W hardware cap is shown as a thin reference (not a model).

  python code/plot_measured_only.py
"""
from __future__ import annotations
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

cap = json.load(open(os.path.join(C.RESULTS_DIR, "model_info.json")))["power_cap_w"]


def load(name):
    return list(csv.DictReader(open(os.path.join(C.RESULTS_DIR, name))))


def arr(rows, k):
    return np.array([float(r[k]) for r in rows])


# prefill: DVFS sweep (split un-capped vs cap-throttled)
pf = [r for r in load("dvfs.csv") if r["workload"] == "prefill"]
req, act = arr(pf, "req_clk_mhz"), arr(pf, "act_clk_mhz")
Tpf, Ppf = arr(pf, "throughput_tok_s"), arr(pf, "power_avg_w")
pcap = act < req - 30

# decode: batch sweep (b<=32 clean, b=64 thermal-edge)
dc = sorted(load("decode.csv"), key=lambda r: float(r["batch"]))
Bd, Td, Pd = arr(dc, "batch"), arr(dc, "throughput_tok_s"), arr(dc, "power_avg_w")
dcap = Bd > 32

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

# ---- left: power vs throughput ----
axL.scatter(Tpf[~pcap], Ppf[~pcap], c="C1", s=95, edgecolor="k", lw=.4, zorder=5,
            label="prefill (frequency sweep)")
axL.scatter(Tpf[pcap], Ppf[pcap], facecolors="none", edgecolors="C1", s=95, zorder=5,
            label="prefill, cap-throttled")
axL.scatter(Td[~dcap], Pd[~dcap], c="C0", s=95, edgecolor="k", lw=.4, zorder=5,
            label="decode (batch sweep)")
axL.scatter(Td[dcap], Pd[dcap], facecolors="none", edgecolors="C0", s=95, zorder=5,
            label="decode, thermal-edge")
axL.axhline(cap, color="gray", ls=":", lw=1, label=f"hardware cap {cap:.0f} W")
axL.set_xscale("log")
axL.set_xlabel("token throughput  (tok/s)"); axL.set_ylabel("GPU power  (W)")
axL.set_ylim(0, cap * 1.08)
axL.set_title("Measured: power vs throughput")
axL.legend(fontsize=8.5, loc="upper left"); axL.grid(alpha=.3, which="both")

# ---- right: efficiency (tok/J) vs power ----
axR.scatter(Ppf[~pcap], Tpf[~pcap] / Ppf[~pcap], c="C1", s=95, edgecolor="k", lw=.4, zorder=5,
            label="prefill (frequency sweep)")
axR.scatter(Ppf[pcap], Tpf[pcap] / Ppf[pcap], facecolors="none", edgecolors="C1", s=95, zorder=5,
            label="prefill, cap-throttled")
axR.scatter(Pd[~dcap], Td[~dcap] / Pd[~dcap], c="C0", s=95, edgecolor="k", lw=.4, zorder=5,
            label="decode (batch sweep)")
axR.scatter(Pd[dcap], Td[dcap] / Pd[dcap], facecolors="none", edgecolors="C0", s=95, zorder=5,
            label="decode, thermal-edge")
axR.set_yscale("log")
axR.set_xlabel("GPU power  (W)"); axR.set_ylabel("energy efficiency  (tok/J = tok/s ÷ W)")
axR.set_title("Measured: efficiency vs power")
axR.legend(fontsize=8.5, loc="lower right"); axR.grid(alpha=.3, which="both")

fig.suptitle("Measured data only — V100 / Phi-3-mini  (no model / theory curves)", fontsize=12)
fig.tight_layout()
out = os.path.join(C.FIGURES_DIR, "fig_measured_only.png")
fig.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)
