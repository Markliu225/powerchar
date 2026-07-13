"""Per-workload-class rack recipes — the figure for solve_workloads.py.

Two panels; workloads grouped by CLASS (decode-dominant on the left -> pure-prefill batch on
the right, the order defined by solve_workloads.WORKLOAD_CLASSES; labels carry the class name,
never a numeric ratio):
  (a) rack throughput OPT vs TDP, normalized to the TDP baseline, LINEAR axis — bar heights
      show the true gain (absolute tok/s spans ~124x across classes, so absolutes are printed
      on the bars instead of being the axis)
  (b) OPT fleet composition: prefill/decode GPU split, chosen caps, the physical slot wall

  python3 plot_workloads.py   ->   fig_workloads.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import solve_workloads as SW

HERE = os.path.dirname(os.path.abspath(__file__))
GREEN, RED, BLUE, ORANGE = "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"   # repo figure palette

# ---- solve every workload once (constrained + unconstrained), in class order ----
by_id = {w["id"]: w for w in SW.PORTFOLIO}
recs = []
for wid in SW.ORDERED_IDS:
    c = SW.load_workload(by_id[wid])
    o = SW.solve_opt(c)
    t = SW.solve_tdp(c)
    recs.append(dict(id=wid, cls=c["cls"], c=c, o=o, t=t))

x = np.arange(len(recs))
lab = [f"{r['id']}\n{r['cls']['short']}" for r in recs]
# class boundaries (positions between adjacent bars of different classes)
bounds = [i - 0.5 for i in range(1, len(recs)) if recs[i]["cls"]["key"] != recs[i - 1]["cls"]["key"]]


def class_separators(a):
    for b in bounds:
        a.axvline(b, color="lightgray", lw=0.9, zorder=0)


fig, ax = plt.subplots(2, 1, figsize=(13, 10.5), gridspec_kw={"height_ratios": [1.15, 1]})

# (a) throughput normalized to the TDP baseline — LINEAR axis, so bar heights show the true gain
a = ax[0]
wdt = 0.38


def fmt_tok(v):
    return f"{v/1e3:.0f}k" if v >= 9500 else (f"{v/1e3:.1f}k" if v >= 1000 else f"{v:.0f}")


rel = [r["o"]["tot"] / r["t"]["tot"] for r in recs]
a.bar(x - wdt / 2, rel, wdt, color=GREEN, label="OPT — caps float, full budget, slot-aware")
a.bar(x + wdt / 2, [1.0] * len(recs), wdt, color=RED, label="TDP baseline (= 1.0, every GPU @250 W)")
for i, r in enumerate(recs):
    a.annotate(f"+{100 * (rel[i] - 1):.0f}%", (x[i] - wdt / 2, rel[i]),
               textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5,
               color=GREEN, weight="bold")
    # absolute throughput printed on the bars (the axis is relative; absolutes span ~124x)
    a.text(x[i] - wdt / 2, rel[i] / 2, f"{fmt_tok(r['o']['tot'])} tok/s", rotation=90,
           ha="center", va="center", fontsize=7, color="white", weight="bold")
    a.text(x[i] + wdt / 2, 0.5, f"{fmt_tok(r['t']['tot'])} tok/s", rotation=90,
           ha="center", va="center", fontsize=7, color="white", weight="bold")
class_separators(a)
a.set_xticks(x); a.set_xticklabels(lab, fontsize=8)
a.set_ylim(0, max(rel) * 1.27)          # headroom so the legend clears the +% annotations
a.set_ylabel("rack throughput relative to TDP baseline (linear)")
a.set_title(f"Rack throughput per workload class — OPT vs TDP, normalized to TDP   ({SW.W_RACK/1e3:.0f} kW, "
            f"<= {SW.N_GPU_MAX} GPU slots)\nbar heights show the TRUE gain; absolute tok/s printed on the "
            "bars (classes span ~124x) · grouped by class: decode-dominant (left) -> pure-prefill batch (right)")
a.legend(fontsize=9, loc="upper left"); a.grid(alpha=.3, axis="y")

# (b) OPT fleet composition + chosen caps + the slot wall
a = ax[1]
Np = [r["o"]["Np"] for r in recs]; Nd = [r["o"]["Nd"] for r in recs]
a.bar(x, Np, color=BLUE, label="prefill GPUs")
a.bar(x, Nd, bottom=Np, color=ORANGE, label="decode GPUs")
a.axhline(SW.N_GPU_MAX, color="k", ls="--", lw=1.2)
a.text(-0.42, SW.N_GPU_MAX + 4.2, f"physical slot limit\nN_max={SW.N_GPU_MAX}",
       ha="left", fontsize=9, weight="bold")
class_separators(a)
for i, r in enumerate(recs):
    a.text(i, Np[i] + Nd[i] + 0.8, f"{Np[i]}+{Nd[i]}\n@{r['o']['p_p']:.0f}/{r['o']['p_d']:.0f}W",
           ha="center", fontsize=7.5)
a.set_xticks(x); a.set_xticklabels(lab, fontsize=8)
a.set_ylabel(f"GPUs in the {SW.W_RACK/1e3:.0f} kW rack")
a.set_title("OPT fleet per workload: prefill/decode split + chosen caps (@pre/dec W)\n"
            "even the prompt-heavy summarize (batch) class stacks 29/32 GPUs on decode — "
            "its 32k context caps decode at 6 tok/s/GPU")
a.legend(fontsize=9, loc="lower right"); a.grid(alpha=.3, axis="y")
a.set_ylim(0, SW.N_GPU_MAX * 1.28)

fig.suptitle("V100 rack recipes by REAL workload class — measured per-workload curves", fontsize=14)
fig.tight_layout()
out = os.path.join(HERE, "fig_workloads.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
