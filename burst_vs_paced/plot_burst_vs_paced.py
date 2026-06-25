"""Plot burst-vs-paced: cumulative work vs time (who finishes first), plus temp & clock."""
from __future__ import annotations
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "timeline.csv"))))
meta = json.load(open(os.path.join(HERE, "meta.json")))
N = meta["n_total"]
col = {"A": "#d62728", "B": "#1f77b4"}
name = {"A": f"BURST (ignore cooling)  JCT={meta['jct_burst_s']}s",
        "B": f"PACED trickle (hold {meta['t_target_c']:.0f}°C)  JCT={meta['jct_paced_s']}s"}


def series(strat):
    d = [r for r in rows if r["strategy"] == strat and r["phase"] in ("run", "cool")]
    d.sort(key=lambda r: float(r["t_s"]))
    t = np.array([float(r["t_s"]) for r in d]); t -= t[0]
    return (t, np.array([float(r["work_done"]) for r in d]),
            np.array([float(r["temp_c"]) for r in d]),
            np.array([float(r["sm_clk_mhz"]) for r in d]),
            np.array([float(r["phase"] == "cool") for r in d]))


fig, ax = plt.subplots(3, 1, figsize=(13, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})

steady = {}
for S in ("A", "B"):
    t, work, temp, clk, iscool = series(S)
    ax[0].plot(t, work, color=col[S], lw=2.2, label=name[S])
    ax[0].plot(t[-1], work[-1], "o", color=col[S], ms=9, mec="k")
    # steady-state rate: linear fit to the last 45% (past the heat-soak transient)
    m = t >= t[-1] * 0.55
    if m.sum() >= 2:
        sl, ic = np.polyfit(t[m], work[m], 1)
        steady[S] = sl
        tail = np.linspace(t[-1] * 0.45, t[-1], 50)
        ax[0].plot(tail, sl * tail + ic, "--", color=col[S], lw=1.2, alpha=.9)
        ax[0].annotate(f"{'BURST' if S=='A' else 'PACED'} steady ≈ {sl:.1f} GEMM/s",
                       (tail[10], sl * tail[10] + ic), color=col[S], fontsize=9,
                       xytext=(0, -16 if S == "A" else 8), textcoords="offset points")
    ax[1].plot(t, temp, color=col[S], lw=1.8, label=f"{S} temp")
    ax[2].plot(t, clk, color=col[S], lw=1.8, label=f"{S} clock")
    # shade B's cooling (idle) stretches
    if S == "B":
        d = np.diff(np.concatenate([[0], iscool, [0]]))
        for si, ei in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            for a in ax:
                a.axvspan(t[min(si, len(t)-1)], t[min(ei-1, len(t)-1)], color="#1f77b4", alpha=.08)

a = ax[0]
a.axhline(N, color="gray", ls="--", alpha=.6, label=f"total workload = {N} GEMMs")
a.set_ylabel("cumulative GEMMs done")
faster = "BURST" if meta["jct_burst_s"] < meta["jct_paced_s"] else "PACED"
a.set_title(f"Fixed workload ({N} fp16 {meta['gemm_n']}² GEMMs) — who finishes first?\n"
            f"{faster} wins: burst {meta['jct_burst_s']}s vs paced {meta['jct_paced_s']}s "
            f"({meta['speedup_burst_over_paced']}× ; paced spent {meta['paced_cooling_s']}s cooling). "
            f"blue bands = paced idle-cooling")
a.legend(fontsize=9, loc="lower right"); a.grid(alpha=.3)

a = ax[1]
a.axhline(83, color="darkred", ls="--", alpha=.7, label="83°C throttle onset")
a.axhline(meta["t_target_c"], color="#1f77b4", ls=":", alpha=.6, label=f"paced target {meta['t_target_c']:.0f}°C")
a.set_ylabel("temperature (°C)"); a.legend(fontsize=8, loc="lower right"); a.grid(alpha=.3)

a = ax[2]
a.set_ylabel("SM clock (MHz)"); a.set_xlabel("time since job start (s)")
a.legend(fontsize=8, loc="lower right"); a.grid(alpha=.3)

fig.suptitle("Burst (power through the throttle) vs Paced (stay cool, idle to cool) — same total work", fontsize=13)
fig.tight_layout()
out = os.path.join(HERE, "fig_burst_vs_paced.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
print(f"BURST {meta['jct_burst_s']}s | PACED {meta['jct_paced_s']}s | "
      f"burst {meta['speedup_burst_over_paced']}x {'faster' if meta['jct_paced_s']>meta['jct_burst_s'] else 'slower'}")
