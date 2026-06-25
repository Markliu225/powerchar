"""Plot the thermal-throttle timeline (throttle.csv + meta.json)."""
from __future__ import annotations
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "throttle.csv"))))
meta = json.load(open(os.path.join(HERE, "meta.json")))


def col(k, f=float):
    return np.array([f(r[k]) if r[k] != "" else np.nan for r in rows])


t = col("t_s"); temp = col("temp_c"); clk = col("sm_clk_mhz")
pwr = col("power_w"); util = col("util_pct"); th = col("throttle_thermal", int)
on, off = meta["load_on_s"], meta["load_off_s"]
slow, shut, gmax = meta.get("slowdown_c"), meta.get("shutdown_c"), meta.get("gpu_max_c")


def shade_load(a):
    a.axvspan(on, off, color="orange", alpha=.10, label="full load")


def shade_throttle(a):
    """red bands where NVML reports thermal throttling active."""
    first = True
    d = np.diff(np.concatenate([[0], th, [0]]))
    for s_i, e_i in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        a.axvspan(t[min(s_i, len(t) - 1)], t[min(e_i - 1, len(t) - 1)], color="red", alpha=.13,
                  label="thermal-throttle active" if first else None)
        first = False


fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

# main: temperature + SM clock
a = ax[0]
shade_load(a); shade_throttle(a)
a.plot(t, temp, color="#d62728", lw=2, label="temperature (°C)")
if gmax:
    a.axhline(gmax, color="#d62728", ls="--", lw=1.4, alpha=.9, label=f"max-operating / throttle onset {gmax}°C")
if slow:
    a.axhline(slow, color="darkred", ls="-.", lw=1.1, alpha=.7, label=f"HW slowdown {slow}°C (not reached)")
if shut:
    a.axhline(shut, color="darkred", ls=":", lw=1.0, alpha=.6, label=f"shutdown {shut}°C")
a.axvline(on, color="k", ls="-", lw=1, alpha=.5)
a.annotate("load ON", (on, temp.max()), xytext=(on + 1.5, temp.max() - 6), fontsize=9)
a.set_ylabel("temperature (°C)", color="#d62728"); a.set_ylim(45, (shut or 92) + 2)
a.set_title(f"{meta['name']} — thermal throttling: full load → temp pinned at {gmax}°C max-op → "
            f"clock auto-drops {1305}→{meta.get('min_clk_under_load_mhz')}MHz to hold it → load off → recovers")
a2 = a.twinx()
a2.plot(t, clk, color="#1f77b4", lw=2, label="SM clock (MHz)")
a2.set_ylabel("SM clock (MHz)", color="#1f77b4")
# merge legends
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
a.grid(alpha=.3)

# power + util
a = ax[1]
shade_load(a); shade_throttle(a)
a.plot(t, pwr, color="#2ca02c", lw=1.8, label="power (W)")
a.set_ylabel("power (W)", color="#2ca02c"); a.set_xlabel("time (s)")
a3 = a.twinx()
a3.plot(t, util, color="#9467bd", lw=1, alpha=.6, label="util (%)")
a3.set_ylabel("util (%)", color="#9467bd"); a3.set_ylim(0, 105)
a.legend(fontsize=8, loc="center right"); a.grid(alpha=.3)

fig.suptitle("Real-time thermal-throttle monitor", fontsize=13)
fig.tight_layout()
out = os.path.join(HERE, "fig_thermal_throttle.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
print(f"peak {meta['peak_temp_c']}°C | min clk under load {meta['min_clk_under_load_mhz']}MHz | "
      f"thermal-throttle samples {meta['thermal_throttle_samples']}")
