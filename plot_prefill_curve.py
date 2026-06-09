"""Prefill throughput-vs-power, traced from low throughput upward by sweeping
prompt length. batch=1 and batch=8."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results_prefill_curve.csv")))
for r in rows:
    for k, v in r.items():
        try: r[k] = float(v)
        except (ValueError, TypeError): pass

by_b = {}
for r in rows:
    by_b.setdefault(int(r["batch"]), []).append(r)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
colors = {1: "tab:red", 8: "tab:purple"}

for B, rs in sorted(by_b.items()):
    rs = sorted(rs, key=lambda r: r["throughput_tok_s"])
    t = [r["throughput_tok_s"] for r in rs]
    p = [r["power_avg_w"] for r in rs]
    ax[0].plot(t, p, "o-", color=colors[B], label=f"batch={B} (sweep prompt len)")
    for r in rs:
        if int(r["seq_len"]) in (4, 16, 64, 256, 1024, 4096):
            ax[0].annotate(f"S={int(r['seq_len'])}", (r["throughput_tok_s"], r["power_avg_w"]),
                           fontsize=6.5, alpha=0.8, xytext=(3, 4), textcoords="offset points")
ax[0].axhline(145, ls=":", color="k", lw=0.8, label="145W cap")
ax[0].set_xlabel("prefill throughput (tokens/s)")
ax[0].set_ylabel("GPU power (W)")
ax[0].set_title("PREFILL throughput vs power (from low throughput up)\nrises from ~83W floor, accelerates, saturates near the cap")
ax[0].set_ylim(70, 150); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=9)

# log-log of the RISING region (throughput up to the peak), with dynamic power
ax[1].set_title("rising region, dynamic power (P − floor) vs throughput\n(log-log; slope ≈ exponent)")
floor = 81.0
for B, rs in sorted(by_b.items()):
    rs = [r for r in sorted(rs, key=lambda r: r["throughput_tok_s"])]
    # keep rising region: up to peak throughput
    peak_i = int(np.argmax([r["throughput_tok_s"] for r in rs]))
    rs = rs[:peak_i + 1]
    t = np.array([r["throughput_tok_s"] for r in rs])
    dp = np.array([r["power_avg_w"] for r in rs]) - floor
    m = dp > 0
    ax[1].plot(t[m], dp[m], "o-", color=colors[B], label=f"batch={B}")
    if m.sum() >= 3:
        a = np.polyfit(np.log(t[m]), np.log(dp[m]), 1)[0]
        print(f"batch={B}: dynamic power (P-{floor:.0f}) ∝ tput^{a:.2f}")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("prefill throughput (tokens/s, log)")
ax[1].set_ylabel("power above ~81W floor (W, log)")
ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=9)

fig.suptitle("RTX 5060 / Qwen2.5-1.5B fp16 — PREFILL throughput↔power from low load")
fig.tight_layout()
fig.savefig("curves_prefill_curve.png", dpi=130)
print("wrote curves_prefill_curve.png")
