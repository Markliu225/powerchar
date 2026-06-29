"""Plot the decode cap×batch power<->throughput sweep (decode_pt.csv)."""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "decode_pt.csv"))))
caps = sorted({int(float(r["cap_w"])) for r in rows})
cmap = plt.get_cmap("viridis")
col = {c: cmap(i / max(len(caps) - 1, 1)) for i, c in enumerate(caps)}


def sub(cap):
    d = [r for r in rows if int(float(r["cap_w"])) == cap]
    d.sort(key=lambda r: float(r["batch"]))
    f = lambda k: np.array([float(r[k]) for r in d])
    return f("batch"), f("throughput_tok_s"), f("power_avg_w"), f("sm_clk_avg"), f("tok_per_joule")


fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# (0,0) P vs T, one band per cap (batch increases along each line)
a = ax[0, 0]
for c in caps:
    b, T, P, _, _ = sub(c)
    a.plot(T, P, "o-", color=col[c], ms=5, lw=1.4, label=f"cap {c}W")
a.set_xlabel("throughput (tok/s)"); a.set_ylabel("measured power (W)")
a.set_title("DECODE power vs throughput — each cap = a band, batch grows along it")
a.legend(fontsize=8); a.grid(alpha=.3)

# (0,1) T vs batch per cap (per-cap saturation)
a = ax[0, 1]
for c in caps:
    b, T, P, _, _ = sub(c)
    a.plot(b, T, "o-", color=col[c], ms=5, lw=1.4, label=f"cap {c}W")
a.set_xscale("log", base=2); a.set_xlabel("batch"); a.set_ylabel("throughput (tok/s)")
a.set_title("throughput vs batch (per power cap) — rises then saturates"); a.legend(fontsize=8); a.grid(alpha=.3)

# (1,0) efficiency vs power
a = ax[1, 0]
for c in caps:
    b, T, P, _, E = sub(c)
    a.plot(P, E, "o-", color=col[c], ms=5, lw=1.4, label=f"cap {c}W")
a.set_xlabel("power (W)"); a.set_ylabel("efficiency (tok/J)")
a.set_title("DECODE efficiency vs power"); a.legend(fontsize=8); a.grid(alpha=.3)

# (1,1) P-T FRONTIER: best throughput achievable at each power (envelope over cap×batch)
a = ax[1, 1]
P_all = np.array([float(r["power_avg_w"]) for r in rows])
T_all = np.array([float(r["throughput_tok_s"]) for r in rows])
a.scatter(T_all, P_all, c=[col[int(float(r["cap_w"]))] for r in rows], s=18, alpha=.5)
# envelope: max throughput in each power bin
bins = np.linspace(P_all.min(), P_all.max(), 16)
idx = np.digitize(P_all, bins)
fx, fy = [], []
for k in range(1, len(bins) + 1):
    m = idx == k
    if m.any():
        fy.append(P_all[m].mean()); fx.append(T_all[m].max())
a.plot(fx, fy, "k-o", lw=2, ms=5, label="P-T frontier (max T per power)")
a.set_xlabel("throughput (tok/s)"); a.set_ylabel("power (W)")
a.set_title("DECODE P-T frontier (envelope over cap×batch)"); a.legend(fontsize=8); a.grid(alpha=.3)

fig.suptitle(f"Phi-3-mini on V100 GPU1 — DECODE power↔throughput (power-cap × batch sweep, C={rows[0]['ctx']})", fontsize=13)
fig.tight_layout()
out = os.path.join(HERE, "fig_decode_pt.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
print(f"{len(rows)} points, caps {caps}, T {T_all.min():.0f}-{T_all.max():.0f} tok/s, P {P_all.min():.0f}-{P_all.max():.0f} W")
