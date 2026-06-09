"""Delivered tokens/s vs power (rate-limited prefill) + the shape fit."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results_rate.csv")))
t = np.array([float(r["delivered_tok_s"]) for r in rows])
p = np.array([float(r["power_avg_w"]) for r in rows])

# linear fit  P = a + b*t
b, a = np.polyfit(t, p, 1)
pred = a + b * t
ss = 1 - np.sum((p - pred) ** 2) / np.sum((p - p.mean()) ** 2)
# power-law fit P = c * t^k  (to test 'cubic')
k, lc = np.polyfit(np.log(t), np.log(p), 1)
print(f"linear fit:  P = {a:.1f} + {b*1000:.2f} mW/(tok/s) * tput   (R^2={ss:.3f})")
print(f"power-law:   P ∝ tput^{k:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))

ax[0].plot(t, p, "o-", color="tab:red", label="measured (rate-limited)")
tt = np.linspace(t.min(), t.max(), 100)
ax[0].plot(tt, a + b * tt, "--", color="gray",
           label=f"linear fit  P={a:.0f}+{b*1000:.1f}mW·tput  (R²={ss:.3f})")
ax[0].axhline(145, ls=":", color="k", lw=0.8, label="145W cap")
ax[0].set_xlabel("delivered prefill throughput (tokens/s)")
ax[0].set_ylabel("avg GPU power (W)")
ax[0].set_title("Delivered tokens/s vs power — steep, and LINEAR\n(P = idle + energy/token × tok/s, fixed clock)")
ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

# energy per token
ept = p / t * 1000.0   # mJ per token
ax[1].plot(t, ept, "o-", color="tab:green")
ax[1].set_xlabel("delivered prefill throughput (tokens/s)")
ax[1].set_ylabel("energy per token (mJ/token)")
ax[1].set_title("Energy/token falls as load rises\n(idle baseline amortized; floors at the compute cost)")
ax[1].grid(alpha=0.3)

fig.suptitle("RTX 5060 / Qwen2.5-1.5B fp16 — PREFILL delivered-throughput operating curve")
fig.tight_layout()
fig.savefig("curves_rate.png", dpi=130)
print("wrote curves_rate.png")
