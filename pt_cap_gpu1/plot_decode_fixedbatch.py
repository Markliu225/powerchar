"""Decode at FIXED batch: ADDITIVE two-term model, three power stages, fit to the measured sweep.

Physical model (memory time + compute time, NOT a roofline max):

    Throughput(f_sm) = B / Time_per_token
    Time_per_token   = T_mem + T_comp(f_sm)
    T_mem            = D_mem / BW(f_mem)            CONSTANT  (memory-controller clock is fixed)
    T_comp(f_sm)     = O_comp / OPS(f_sm)           ~ 1/f_sm^p  (OPS degrades super-linearly at low clock)

Because the HBM clock is fixed, T_mem is a constant floor -> the throughput CEILING = B / T_mem.
We anchor T_mem to the measured plateau (max throughput), so ceiling == measured ceiling, then fit the
compute term T_comp (which is large at low clock and vanishes at f_max). Power: P = P_static + chi*x^theta.

Three stages as power (hence f_sm) rises:
  I   low power   : T_comp >> T_mem  -> pseudo-compute-bound, throughput rises ~linearly with power.
  II  mid power   : T_comp ~  T_mem  -> diminishing returns, task returns to memory-bound.
  III high power  : T_comp <  5% T_mem, clock pinned near f_max -> throughput PLATEAUS at B/T_mem;
                    extra power only raises voltage/heat, not throughput.
"""
from __future__ import annotations
import csv, io, os, re
import numpy as np
from scipy.optimize import curve_fit, brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
text = open(os.path.join(HERE, "decode_fixedbatch.csv")).read()

# ----------------------------------------------------------------------------- parse
P, T, clk = [], [], []
try:
    dr = list(csv.DictReader(io.StringIO(text)))
except Exception:
    dr = []
if dr and "throughput_tok_s" in (dr[0] or {}):
    for r in dr:
        P.append(float(r.get("power_w") or r.get("power_avg_w")))
        T.append(float(r["throughput_tok_s"])); clk.append(float(r["sm_clk_avg"]))
else:
    for line in text.splitlines():
        m = re.search(r"([\d.]+)\s*W\s*@\s*(\d+)\s*MHz.*?([\d.]+)\s*tok/s", line)
        if m:
            P.append(float(m.group(1))); clk.append(float(m.group(2))); T.append(float(m.group(3)))

P, T, clk = np.array(P), np.array(T), np.array(clk)
o = np.argsort(clk); P, T, clk = P[o], T[o], clk[o]

# ----------------------------------------------------------------------------- model
B    = 96.0
FMAX = 1530.0
Ttok = B / T                                   # measured time-per-token (s)

# T_mem = constant memory floor, anchored so ceiling == measured plateau (max throughput)
THR_CEIL = T.max()
T_MEM    = B / THR_CEIL                         # ~149 ms

# T_comp(f) = C*(1/x^p - 1)  -> = 0 at f_max (compute fully hidden), grows toward low clock
def t_comp(x, C, p):  return C * (1.0 / x ** p - 1.0)
def ttok(clk_mhz, C, p):
    x = clk_mhz / FMAX
    return T_MEM + t_comp(x, C, p)
def thr_model(clk_mhz, C, p):
    return B / ttok(clk_mhz, C, p)

(C_f, p_f), _ = curve_fit(ttok, clk, Ttok, p0=[0.03, 2.0], bounds=([0, 1], [2, 6]), maxfev=200000)
r2_thr = 1 - np.sum((T - thr_model(clk, C_f, p_f)) ** 2) / np.sum((T - T.mean()) ** 2)

# DVFS power  P = P_static + chi*(f/f_max)^theta
def p_model(clk_mhz, Pstat, chi, theta):
    return Pstat + chi * (clk_mhz / FMAX) ** theta
(Pstat_f, chi_f, th_f), _ = curve_fit(p_model, clk, P, p0=[35, 170, 2.4],
                                      bounds=([10, 40, 1.8], [50, 400, 3.2]), maxfev=200000)
r2_pw = 1 - np.sum((P - p_model(clk, Pstat_f, chi_f, th_f)) ** 2) / np.sum((P - P.mean()) ** 2)

# ----------------------------------------------------------------------------- stage boundaries
# I/II : T_comp = T_mem   ;   II/III : T_comp = 5% T_mem
f1 = brentq(lambda f: t_comp(f / FMAX, C_f, p_f) - T_MEM,        clk.min() * .5, FMAX)
f2 = brentq(lambda f: t_comp(f / FMAX, C_f, p_f) - 0.05 * T_MEM, clk.min() * .5, FMAX)
P1 = p_model(f1, Pstat_f, chi_f, th_f)
P2 = p_model(f2, Pstat_f, chi_f, th_f)

cg  = np.linspace(clk.min() * 0.92, FMAX, 400)
Pg  = p_model(cg, Pstat_f, chi_f, th_f)
thg = thr_model(cg, C_f, p_f)

print(f"ADDITIVE model  Throughput = B / (T_mem + T_comp)")
print(f"  T_mem (const memory floor, anchored) = {T_MEM*1e3:.0f} ms   ->  CEILING = B/T_mem = {THR_CEIL:.0f} tok/s")
print(f"  T_comp(f) = C*((f_max/f)^p - 1)   C={C_f*1e3:.1f} ms  p={p_f:.2f}     R^2(thr)={r2_thr:.4f}")
print(f"  effective mem BW = D_mem/T_mem ~ {(2*3.8e9+2*32*3072*2*256*96)/T_MEM/1e9:.0f} GB/s "
      f"(~{100*(2*3.8e9+2*32*3072*2*256*96)/T_MEM/1e9/900:.0f}% of 900 GB/s peak: memory-bound but latency-limited)")
print(f"  STAGE I  (compute-bound) : f < {f1:.0f} MHz  (P < {P1:.0f} W)")
print(f"  STAGE II (transition)    : {f1:.0f}-{f2:.0f} MHz  ({P1:.0f}-{P2:.0f} W)")
print(f"  STAGE III(plateau)       : f > {f2:.0f} MHz  (P > {P2:.0f} W)  -> throughput ~ {THR_CEIL:.0f} tok/s")
print(f"  POWER  P = {Pstat_f:.0f} + {chi_f:.0f}*(f/f_max)^{th_f:.2f}   R^2={r2_pw:.4f}")

# ============================================================================= figure
S1, S2, S3 = "#fde0c0", "#d9ead8", "#cfe2f3"           # stage background colours
fig, ax = plt.subplots(1, 3, figsize=(19, 5.8))

# (a) throughput vs POWER — three stages + ceiling --------------------------------
a = ax[0]
a.axvspan(0, P1, color=S1, alpha=.8, zorder=0)
a.axvspan(P1, P2, color=S2, alpha=.8, zorder=0)
a.axvspan(P2, P.max() * 1.06, color=S3, alpha=.8, zorder=0)
a.axhline(THR_CEIL, color="#444", ls="--", lw=1.5, zorder=2)
a.text(P.min() * 0.72, THR_CEIL + 6, f"ceiling = B/T_mem = {THR_CEIL:.0f} tok/s", fontsize=9, color="#444")
a.plot(Pg, thg, "-", color="#d62728", lw=2.6, zorder=4, label=f"model  B/(T_mem+T_comp)  (R²={r2_thr:.3f})")
sc = a.scatter(P, T, c=clk, cmap="viridis", s=78, edgecolor="k", lw=.5, zorder=5, label="measured")
for xx, lab, col in [((P.min()*.7+P1)/2, "I", "#9c4a10"), ((P1+P2)/2, "II", "#3a6b35"),
                     ((P2+P.max())/2, "III", "#1f5fa8")]:
    a.text(xx, THR_CEIL*0.22, lab, ha="center", fontsize=15, weight="bold", color=col, alpha=.6)
a.text((P.min()*.7+P1)/2, THR_CEIL*0.34, "compute-\nbound", ha="center", fontsize=7.5, color="#9c4a10")
a.text((P1+P2)/2, THR_CEIL*0.34, "diminishing\nreturns", ha="center", fontsize=7.5, color="#3a6b35")
a.text((P2+P.max())/2, THR_CEIL*0.34, "memory\nplateau", ha="center", fontsize=7.5, color="#1f5fa8")
cb = fig.colorbar(sc, ax=a); cb.set_label("SM clock (MHz)")
a.set_xlabel("power (W)"); a.set_ylabel("decode throughput (tok/s)")
a.set_title("(a) power → throughput : three stages + ceiling")
a.legend(loc="lower right", fontsize=8.5); a.grid(alpha=.25)
a.set_xlim(P.min() * 0.7, P.max() * 1.04); a.set_ylim(0, THR_CEIL * 1.16)

# (b) time-per-token decomposition vs SM clock ------------------------------------
a = ax[1]
ms = 1e3
xg = cg / FMAX
a.axvspan(cg.min(), f1, color=S1, alpha=.8, zorder=0)
a.axvspan(f1, f2, color=S2, alpha=.8, zorder=0)
a.axvspan(f2, FMAX, color=S3, alpha=.8, zorder=0)
a.plot(cg, np.full_like(cg, T_MEM) * ms, "--", color="#1f5fa8", lw=2.0, label=f"T_mem = {T_MEM*ms:.0f} ms (const)")
a.plot(cg, t_comp(xg, C_f, p_f) * ms, "-.", color="#2ca02c", lw=2.0, label="T_comp(f) ∝ 1/f$^p$")
a.plot(cg, ttok(cg, C_f, p_f) * ms, "-", color="#d62728", lw=2.4, label="T_per_token (total)")
a.scatter(clk, Ttok * ms, c=clk, cmap="viridis", s=58, edgecolor="k", lw=.5, zorder=6)
a.set_xlabel("SM clock (MHz)"); a.set_ylabel("time per token (ms)")
a.set_title("(b) T_mem (const floor) + T_comp(f)  — sets the 3 stages")
a.legend(loc="upper right", fontsize=8.5); a.grid(alpha=.25)
a.set_xlim(cg.min(), FMAX); a.set_ylim(0, Ttok.max() * ms * 1.08)

# (c) DVFS power vs SM clock ------------------------------------------------------
a = ax[2]
a.scatter(clk, P, c=clk, cmap="viridis", s=78, edgecolor="k", lw=.5, zorder=5, label="measured")
a.plot(cg, Pg, "-", color="#d62728", lw=2.2, zorder=4,
       label=f"P = {Pstat_f:.0f} + {chi_f:.0f}·(f/f_max)$^{{{th_f:.2f}}}$")
a.set_xlabel("SM clock (MHz)"); a.set_ylabel("sustained power (W)")
a.set_title(f"(c) DVFS power  (θ={th_f:.2f}, R²={r2_pw:.3f})")
a.legend(loc="upper left", fontsize=8.5); a.grid(alpha=.25)
a.set_xlim(clk.min() * 0.9, FMAX * 1.01); a.set_ylim(0, P.max() * 1.12)

fig.suptitle("Phi-3-mini on V100 — decode power→throughput: additive model (T_mem + T_comp), "
             "three stages, ceiling = B/T_mem  (batch=96, HBM fixed)", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = os.path.join(HERE, "fig_decode_fixedbatch.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
