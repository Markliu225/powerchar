"""Plot prefill/decode power<->throughput UNDER POWER CAPS (from pt_cap.csv)."""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "pt_cap.csv"))))


def g(phase, k):
    return np.array([float(r[k]) for r in rows if r["phase"] == phase])


Tp, Pp, Kp = g("prefill", "throughput_tok_s"), g("prefill", "power_avg_w"), g("prefill", "cap_w")
Td, Pd, Kd = g("decode", "throughput_tok_s"), g("decode", "power_avg_w"), g("decode", "cap_w")

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# (left) power vs throughput, both phases, swept by cap
a = ax[0]
a.plot(Tp, Pp, "o-", color="C1", ms=7, mec="k", label="prefill (b8, S512)")
a.plot(Td, Pd, "s-", color="C0", ms=7, mec="k", label="decode (b48, ctx256)")
for t, p, c in zip(Tp, Pp, Kp):
    a.annotate(f"{c:.0f}W", (t, p), fontsize=7, xytext=(3, 4), textcoords="offset points", color="C1")
for t, p, c in zip(Td, Pd, Kd):
    a.annotate(f"{c:.0f}W", (t, p), fontsize=7, xytext=(3, -10), textcoords="offset points", color="C0")
a.set_xlabel("throughput (tok/s)"); a.set_ylabel("measured power (W)")
a.set_title("power vs throughput under power caps (label = cap)"); a.legend(); a.grid(alpha=.3)

# (right) throughput vs power cap -> prefill steep, decode flat
a = ax[1]
a.plot(Kp, Tp, "o-", color="C1", ms=7, mec="k", label="prefill")
a.set_xlabel("power cap (W)"); a.set_ylabel("prefill throughput (tok/s)", color="C1"); a.grid(alpha=.3)
a2 = a.twinx()
a2.plot(Kd, Td, "s-", color="C0", ms=7, mec="k", label="decode")
a2.set_ylabel("decode throughput (tok/s)", color="C0")
a.set_title("throughput vs power cap — prefill drops steeply, decode stays ~flat")
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)

fig.suptitle("Phi-3-mini on V100 GPU1 — prefill & decode under power capping", fontsize=13)
fig.tight_layout()
out = os.path.join(HERE, "fig_pt_cap.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
print(f"prefill: cap {Kp.min():.0f}-{Kp.max():.0f}W -> T {Tp.min():.0f}-{Tp.max():.0f} tok/s, P {Pp.min():.0f}-{Pp.max():.0f}W")
print(f"decode : cap {Kd.min():.0f}-{Kd.max():.0f}W -> T {Td.min():.0f}-{Td.max():.0f} tok/s, P {Pd.min():.0f}-{Pd.max():.0f}W")
