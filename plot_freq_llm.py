"""Throughput-vs-power traced by sweeping CLOCK FREQUENCY on real LLM workloads.
Fits P = P0 + k*tput^alpha for each, and reports the high-end local exponent."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results_freq_llm.csv")))
W = {}
for r in rows:
    W.setdefault(r["workload"], []).append(
        (float(r["tok_s"]), float(r["power_avg_w"]), float(r["act_sm_clk"])))


def fit(tps, pw):
    tps, pw = np.array(tps), np.array(pw)
    best = None
    for P0 in np.linspace(0, min(pw) * 0.95, 60):
        a, b = np.polyfit(np.log(tps), np.log(pw - P0), 1)
        resid = np.sum((np.log(pw - P0) - (a * np.log(tps) + b)) ** 2)
        if best is None or resid < best[0]:
            best = (resid, P0, a, np.exp(b))
    return best[1], best[2], best[3]


styles = {"prefill_s2048": ("tab:red", "PREFILL 1x2048 (compute-bound)"),
          "decode_b1": ("tab:blue", "DECODE b=1 (memory-bound)")}

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

for name, (col, lab) in styles.items():
    pts = sorted(W[name])
    tps = [p[0] for p in pts]; pw = [p[1] for p in pts]
    P0, alpha, k = fit(tps, pw)
    # high-end local exponent (top 3 points)
    hi = np.polyfit(np.log(tps[-3:]), np.log(np.array(pw[-3:])), 1)[0]
    ax[0].plot(tps, pw, "o-", color=col,
               label=f"{lab}\n  fit α={alpha:.1f} (P0={P0:.0f}W); high-end P∝tput^{hi:.1f}")
    print(f"{name:<16} fit P={P0:.0f}+k*tput^{alpha:.2f}  high-end exponent={hi:.2f}")

ax[0].set_xlabel("token throughput (tokens/s)")
ax[0].set_ylabel("GPU power (W)")
ax[0].axhline(145, ls=":", color="k", lw=0.8, label="145W cap")
ax[0].set_title("Throughput vs power, swept by CLOCK FREQUENCY\nprefill: convex/accelerating ≈ cubic;  decode b=1: little throughput per watt")
ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7.5, loc="upper left")

# log-log view: a straight line of slope α is the power law
for name, (col, lab) in styles.items():
    pts = sorted(W[name])
    tps = [p[0] for p in pts]; pw = [p[1] for p in pts]
    ax[1].plot(tps, pw, "o-", color=col, label=lab.split(" (")[0])
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("token throughput (tokens/s, log)")
ax[1].set_ylabel("GPU power (W, log)")
ax[1].set_title("log-log: slope = power-law exponent\n(steeper at the high-clock end → ~quadratic-to-cubic)")
ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8)

fig.suptitle("RTX 5060 / Qwen2.5-1.5B fp16 — capturing the throughput↔power law via DVFS")
fig.tight_layout()
fig.savefig("curves_freq_llm.png", dpi=130)
print("wrote curves_freq_llm.png")
