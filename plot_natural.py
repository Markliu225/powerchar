"""Natural (no clock-lock) phase-isolated sweeps + the reconciliation:
the natural operating points sit at the TOP of the DVFS curve because the GPU
boosts to its clock ceiling immediately, so a load sweep never traverses the
cubic — it stays pinned at the top."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p):
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k, v in r.items():
            try: r[k] = float(v)
            except (ValueError, TypeError): pass
    return rows


pre = sorted(load("results_nat_prefill.csv"), key=lambda r: r["throughput_tok_s"])
dec = sorted(load("results_nat_decode.csv"), key=lambda r: r["throughput_tok_s"])
clk = sorted(load("results_clock.csv"), key=lambda r: r["act_sm_clk"])

fig, ax = plt.subplots(1, 3, figsize=(19, 5.8))

# Panel 1: PREFILL natural — throughput vs power, clock annotated
ax[0].plot([r["throughput_tok_s"] for r in pre], [r["power_avg_w"] for r in pre],
           "o-", color="tab:red")
for r in pre:
    ax[0].annotate(f"{r['sm_clk_avg']:.0f}MHz", (r["throughput_tok_s"], r["power_avg_w"]),
                   fontsize=6.5, alpha=0.8, xytext=(3, 4), textcoords="offset points")
ax[0].axhline(145, ls=":", color="k", lw=0.8, label="145W cap")
ax[0].set_title("PREFILL natural sweep (batch=8, vary prompt len)\npower FLAT ~138W; clock pinned ~2700MHz")
ax[0].set_xlabel("prefill throughput (tok/s)"); ax[0].set_ylabel("power (W)")
ax[0].set_ylim(0, 155); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

# Panel 2: DECODE natural — throughput vs power, clock annotated
ax[1].plot([r["throughput_tok_s"] for r in dec], [r["power_avg_w"] for r in dec],
           "o-", color="tab:blue")
for r in dec:
    ax[1].annotate(f"b{int(r['batch'])}\n{r['sm_clk_avg']:.0f}MHz",
                   (r["throughput_tok_s"], r["power_avg_w"]),
                   fontsize=6.5, alpha=0.8, xytext=(3, -2), textcoords="offset points")
ax[1].axhline(145, ls=":", color="k", lw=0.8, label="145W cap")
ax[1].set_title("DECODE natural sweep (ctx=512, vary batch, mem-bound)\npower rises via OCCUPANCY; clock pinned ~2800MHz")
ax[1].set_xlabel("decode throughput (tok/s)"); ax[1].set_ylabel("power (W)")
ax[1].set_ylim(0, 155); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)

# Panel 3: reconciliation — power vs CLOCK; natural points overlaid on DVFS cubic
ax[2].plot([r["act_sm_clk"] for r in clk], [r["power_avg_w"] for r in clk],
           "s-", color="gray", label="DVFS curve (clock-locked matmul)")
ax[2].scatter([r["sm_clk_avg"] for r in pre], [r["power_avg_w"] for r in pre],
              color="tab:red", zorder=5, label="prefill natural points")
ax[2].scatter([r["sm_clk_avg"] for r in dec], [r["power_avg_w"] for r in dec],
              color="tab:blue", zorder=5, label="decode natural points")
ax[2].axvspan(2640, 2820, color="orange", alpha=0.15)
ax[2].annotate("ALL natural points\nlive here (boost ceiling)\n— load never lowers the clock",
               (2300, 60), fontsize=8, color="darkorange")
ax[2].set_title("WHY no natural cubic:\nDVFS pins clock at the ceiling; load sweep stays at the top")
ax[2].set_xlabel("SM clock (MHz)"); ax[2].set_ylabel("power (W)")
ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8, loc="upper left")

fig.suptitle("RTX 5060 / Qwen2.5-1.5B fp16 — natural DVFS: the clock is already maxed, so load can't trace the cubic")
fig.tight_layout()
fig.savefig("curves_natural.png", dpi=130)
print("wrote curves_natural.png")

# numbers
def expo(rows):
    t = np.array([r["throughput_tok_s"] for r in rows]); p = np.array([r["power_avg_w"] for r in rows])
    return np.polyfit(np.log(t), np.log(p), 1)[0]
print(f"prefill natural: power ∝ tput^{expo(pre):.2f}  (clk {min(r['sm_clk_avg'] for r in pre):.0f}-{max(r['sm_clk_avg'] for r in pre):.0f} MHz)")
print(f"decode  natural: power ∝ tput^{expo(dec):.2f}  (clk {min(r['sm_clk_avg'] for r in dec):.0f}-{max(r['sm_clk_avg'] for r in dec):.0f} MHz)")
