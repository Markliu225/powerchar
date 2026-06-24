"""Goodput vs power cap, from results/goodput_<slug>.csv (one or many models).

Goodput(cap, SLO) = max throughput among batch points whose latency <= SLO, at that cap.
Measurement and SLO are decoupled, so we plot a family of SLO thresholds:
  prefill:  TTFT SLO in {100, 250, 500} ms
  decode :  TPOT SLO in {20, 50, 100} ms   (= 50 / 20 / 10 tok/s per user)
One figure per model: left = prefill goodput vs cap, right = decode goodput vs cap.

  python code/plot_goodput.py
"""
from __future__ import annotations
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

TTFT_SLOS = [100, 250, 500]      # ms (prefill)
TPOT_SLOS = [20, 50, 100]        # ms (decode)


def load(p):
    with open(p) as f:
        return [{k: (float(v) if k not in ("phase",) else v) for k, v in r.items()} for r in csv.DictReader(f)]


def goodput_vs_cap(rows, phase, slo):
    caps = sorted({r["cap_w"] for r in rows if r["phase"] == phase})
    g = []
    for c in caps:
        pts = [r for r in rows if r["phase"] == phase and r["cap_w"] == c and r["latency_ms"] <= slo]
        g.append(max((r["throughput_tok_s"] for r in pts), default=0.0))
    return np.array(caps), np.array(g)


def main():
    files = sorted(glob.glob(os.path.join(C.RESULTS_DIR, "goodput_*.csv")))
    if not files:
        print("no results/goodput_*.csv — run goodput_cap_sweep.py first"); return
    for fp in files:
        slug = os.path.basename(fp)[8:-4]
        rows = load(fp)
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))
        for slo in TTFT_SLOS:
            caps, g = goodput_vs_cap(rows, "prefill", slo)
            if len(caps):
                axL.plot(caps, g / 1e3, "o-", label=f"TTFT ≤ {slo} ms")
        for slo in TPOT_SLOS:
            caps, g = goodput_vs_cap(rows, "decode", slo)
            if len(caps):
                axR.plot(caps, g / 1e3, "o-", label=f"TPOT ≤ {slo} ms")
        for ax, ttl in ((axL, "PREFILL goodput vs power cap"), (axR, "DECODE goodput vs power cap")):
            ax.set_xlabel("power cap (W)"); ax.set_ylabel("max goodput (k tok/s)")
            ax.set_title(ttl); ax.grid(alpha=.3); ax.legend(fontsize=9)
        fig.suptitle(f"{slug} — SLO-constrained goodput vs power cap (V100, deterministic -pl sweep)", fontsize=12)
        fig.tight_layout()
        out = os.path.join(C.FIGURES_DIR, f"goodput_{slug}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
