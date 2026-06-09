"""Plot the DVFS sweep: power vs SM clock on a fixed full-occupancy load,
with a power-law fit P = P0 + k*f^alpha. This is where the 'cubic' lives."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results_clock.csv")))
f = np.array([float(r["act_sm_clk"]) for r in rows])
P = np.array([float(r["power_avg_w"]) for r in rows])
T = np.array([float(r["tflops"]) for r in rows])

# Fit only the DVFS-controlled region (drop points where the GPU couldn't reach
# the requested clock, i.e. duplicate top clocks -> voltage/power saturated).
keep = np.concatenate([[True], np.diff(f) > 20])
ff, PP = f[keep], P[keep]

# Fit P = P0 + k*f^alpha. Try a small grid of P0 (load static floor) and take the
# one giving the best straight line in log-log for (P-P0) vs f.
best = None
for P0 in np.linspace(0, 28, 57):
    if np.any(PP - P0 <= 0):
        continue
    a, b = np.polyfit(np.log(ff), np.log(PP - P0), 1)
    resid = np.sum((np.log(PP - P0) - (a * np.log(ff) + b)) ** 2)
    if best is None or resid < best[0]:
        best = (resid, P0, a, np.exp(b))
_, P0, alpha, k = best
print(f"fit: P = {P0:.1f} + {k:.3e} * f^{alpha:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].plot(f, P, "o", color="tab:purple", ms=8, label="measured power")
fx = np.linspace(f.min(), f.max(), 200)
ax[0].plot(fx, P0 + k * fx ** alpha, "--", color="gray",
           label=f"fit  P = {P0:.0f} + k·f^{alpha:.1f}")
# pure-cubic reference anchored at the lowest point's dynamic part
ax[0].plot(fx, P0 + (P[0] - P0) * (fx / f[0]) ** 3, ":", color="tab:red",
           label="pure cubic f³ reference")
ax[0].axhline(145, ls=":", color="k", lw=0.8, label="145W cap")
ax[0].set_xlabel("SM clock (MHz)")
ax[0].set_ylabel("GPU power (W)")
ax[0].set_title("Power vs CLOCK (fixed full-occupancy matmul)\n— THIS is the ~cubic relationship")
ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

ax[1].plot(f, T, "o-", color="tab:green")
ax[1].set_xlabel("SM clock (MHz)")
ax[1].set_ylabel("achieved compute (TFLOP/s)")
ax[1].set_title("Compute scales ~LINEARLY with clock\n(then saturates at the ~2.6GHz sustainable ceiling)")
ax[1].grid(alpha=0.3)

fig.suptitle("RTX 5060 — DVFS: power is ~cubic in frequency, compute is linear")
fig.tight_layout()
fig.savefig("curves_clock_dvfs.png", dpi=130)
print("wrote curves_clock_dvfs.png")
