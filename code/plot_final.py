"""Final figure set (4 standalone PNGs), measured vs theory.

Per-phase the throughput/power relationship only co-varies on the right knob:
  - PREFILL (compute-bound): use the DVFS / frequency sweep (results/dvfs.csv) —
    clock varies => T and P both move. Theory: P = P0 + k*T^gamma (ideal V∝f => gamma=3).
  - DECODE (memory-bound): use the batch sweep (results/decode.csv) — concurrency
    varies => T and P both move. Theory: P = P0 + k*T (linear).
(The other knob is degenerate for that phase: prefill batch-sweep is flat at the cap;
 decode frequency-sweep is throughput-flat.)

Figures:
  fig_prefill_power_vs_throughput.png   x=tok/s  y=power   (DVFS, measured vs cubic)
  fig_decode_power_vs_throughput.png    x=tok/s  y=power   (batch, measured vs linear)
  fig_prefill_efficiency_vs_power.png   x=power  y=tok/J   (DVFS)
  fig_decode_efficiency_vs_power.png    x=power  y=tok/J   (batch)

  python code/plot_final.py
"""
from __future__ import annotations
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

FIG = C.FIGURES_DIR
P_IDLE = 44.0
cap = json.load(open(os.path.join(C.RESULTS_DIR, "model_info.json")))["power_cap_w"]


def load_csv(name):
    return list(csv.DictReader(open(os.path.join(C.RESULTS_DIR, name))))


def f(rows, k):
    return np.array([float(r[k]) for r in rows])


def r2(y, pred):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


# ---- load: prefill from DVFS (frequency knob), decode from batch sweep ----
dv = [r for r in load_csv("dvfs.csv") if r["workload"] == "prefill"]
req = f(dv, "req_clk_mhz"); act = f(dv, "act_clk_mhz")
capped = act < req - 30
Tpf = f(dv, "throughput_tok_s"); Ppf = f(dv, "power_avg_w")
Tp, Pp = Tpf[~capped], Ppf[~capped]                      # clean (un-capped) prefill
Tpc, Ppc = Tpf[capped], Ppf[capped]

dec = sorted(load_csv("decode.csv"), key=lambda r: float(r["batch"]))
Bd = f(dec, "batch"); Td = f(dec, "throughput_tok_s"); Pd = f(dec, "power_avg_w")
dmask = Bd <= 32                                         # exclude thermal-edge b=64
Tdc, Pdc = Td[dmask], Pd[dmask]


# ---- fits ----
# prefill power-law P = P_IDLE + k*T^g  (baseline fixed at deep idle -> honest gamma)
g_pre, b_pre = np.polyfit(np.log(Tp), np.log(Pp - P_IDLE), 1)
k_pre = np.exp(b_pre)
pre_pred = P_IDLE + k_pre * Tp ** g_pre
r2_pre = r2(Pp, pre_pred)
# ideal cubic reference, anchored to the lowest clean point
k3 = (Pp[np.argmin(Tp)] - P_IDLE) / Tp.min() ** 3
# decode linear P = a + b*T
b1, b0 = np.polyfit(Tdc, Pdc, 1)                          # slope, intercept (note order)
a_dec, k_dec = b0, b1
dec_pred = a_dec + k_dec * Tdc
r2_dec = r2(Pdc, dec_pred)


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=130, bbox_inches="tight"); print("wrote", p); plt.close(fig)


# ===== 1a. PREFILL — power vs throughput (DVFS): measured vs theory =====
fig, ax = plt.subplots(figsize=(8.5, 6))
Tg = np.linspace(Tp.min() * 0.97, Tp.max() * 1.02, 200)
ax.scatter(Tp, Pp, c="C1", s=95, zorder=5, edgecolor="k", lw=.4, label="measured (clock 510–1260 MHz)")
ax.scatter(Tpc, Ppc, facecolors="none", edgecolors="C1", s=95, zorder=5, label="cap-clipped (clock throttled)")
ax.plot(Tg, P_IDLE + k_pre * Tg ** g_pre, "C1-", lw=2,
        label=f"measured fit:  P = {P_IDLE:.0f} + k·T^{g_pre:.2f}  (R²={r2_pre:.3f})")
Tg3 = np.linspace(Tp.min() * 0.97, (((cap - P_IDLE) / k3) ** (1 / 3)), 200)
ax.plot(Tg3, P_IDLE + k3 * Tg3 ** 3, "k--", lw=1.6, label="ideal theory  P ∝ T³  (needs V∝f)")
ax.axhline(cap, color="r", ls="--", alpha=.6, label=f"power cap {cap:.0f} W")
ax.set_xlabel("token throughput  (tok/s)"); ax.set_ylabel("GPU power  (W)")
ax.set_title("PREFILL — power vs throughput  (frequency sweep)\n"
             "measured exponent ≈%.1f, far below the ideal cubic: V barely scales with f on this V100" % g_pre)
ax.set_ylim(0, cap * 1.08); ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="upper left")
save(fig, "fig_prefill_power_vs_throughput.png")

# ===== 1b. DECODE — power vs throughput (batch): measured vs theory =====
fig, ax = plt.subplots(figsize=(8.5, 6))
Tg = np.linspace(0, Td.max() * 1.05, 200)
ax.scatter(Tdc, Pdc, c="C0", s=95, zorder=5, edgecolor="k", lw=.4, label="measured (batch 1–32)")
ax.scatter(Td[~dmask], Pd[~dmask], facecolors="none", edgecolors="C0", s=95, zorder=5,
           label="b=64 (thermal-edge, excluded)")
ax.plot(Tg, a_dec + k_dec * Tg, "C0-", lw=2,
        label=f"theory  P ∝ T  (linear)\nfit: P = {a_dec:.0f} + {k_dec:.3f}·T  (R²={r2_dec:.3f})")
ax.axhline(cap, color="r", ls="--", alpha=.6, label=f"power cap {cap:.0f} W")
ax.set_xlabel("token throughput  (tok/s)"); ax.set_ylabel("GPU power  (W)")
ax.set_title("DECODE — power vs throughput  (batch sweep)\nlinear law P ∝ T confirmed (R²=%.3f)" % r2_dec)
ax.set_ylim(0, cap * 1.08); ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="lower right")
save(fig, "fig_decode_power_vs_throughput.png")

# ===== 2a. PREFILL — efficiency vs power (DVFS) =====
fig, ax = plt.subplots(figsize=(8.5, 6))
Epf = Tpf / Ppf
ax.scatter(Pp, Tp / Pp, c="C1", s=95, zorder=5, edgecolor="k", lw=.4, label="measured (clock 510–1260 MHz)")
ax.scatter(Ppc, Tpc / Ppc, facecolors="none", edgecolors="C1", s=95, zorder=5, label="cap-clipped")
Tg = np.linspace(Tp.min(), Tp.max(), 200); Pg = P_IDLE + k_pre * Tg ** g_pre
ax.plot(Pg, Tg / Pg, "C1-", lw=2, label="theory  E = T / (P₀ + k·T^%.2f)" % g_pre)
ip = np.argmax(Tp / Pp)
ax.annotate(f"efficiency sweet-spot\n≈{(Tp/Pp)[ip]:.0f} tok/J @ {Pp[ip]:.0f} W",
            (Pp[ip], (Tp/Pp)[ip]), fontsize=8.5, color="dimgray",
            xytext=(10, -28), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_xlabel("GPU power  (W)"); ax.set_ylabel("energy efficiency  (tok/J = tok/s ÷ W)")
ax.set_title("PREFILL — efficiency vs power  (frequency sweep)\nsuper-linear P(T) ⇒ a sweet-spot: clocking up past it wastes energy/token")
ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="lower left")
save(fig, "fig_prefill_efficiency_vs_power.png")

# ===== 2b. DECODE — efficiency vs power (batch) =====
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.scatter(Pdc, Tdc / Pdc, c="C0", s=95, zorder=5, edgecolor="k", lw=.4, label="measured (batch 1–32)")
ax.scatter(Pd[~dmask], Td[~dmask] / Pd[~dmask], facecolors="none", edgecolors="C0", s=95, zorder=5,
           label="b=64 (excluded)")
Pg = np.linspace(a_dec + 1, Pd.max() * 1.02, 200)
ax.plot(Pg, (Pg - a_dec) / (k_dec * Pg), "C0-", lw=2,
        label="theory  E = (P−P₀)/(k·P),  P₀=%.0f" % a_dec)
ax.axhline(1 / k_dec, color="gray", ls=":", lw=1.2, label=f"asymptote 1/k = {1/k_dec:.1f} tok/J")
ax.set_xlabel("GPU power  (W)"); ax.set_ylabel("energy efficiency  (tok/J = tok/s ÷ W)")
ax.set_title("DECODE — efficiency vs power  (batch sweep)\nbatching raises power AND efficiency, toward the bandwidth asymptote")
ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="lower right")
save(fig, "fig_decode_efficiency_vs_power.png")

print(f"\nprefill DVFS fit: P=44+{k_pre:.2e}*T^{g_pre:.2f} (R2={r2_pre:.3f})")
print(f"decode batch fit: P={a_dec:.1f}+{k_dec:.4f}*T (R2={r2_dec:.3f}), asymptote {1/k_dec:.1f} tok/J")
