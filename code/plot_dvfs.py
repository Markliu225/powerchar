"""DVFS frequency-knob result: mechanism (T vs clock) and the P(T) law.

Measured by locking the SM clock 510->1530 MHz at a fixed workload (code/dvfs_sweep.py).
Clean points = clock held below the 250 W cap (act ~ req); the top requests were
cap-throttled (act < req) and are drawn hollow / excluded from fits.

  python code/plot_dvfs.py
"""
from __future__ import annotations
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C


def load():
    rows = list(csv.DictReader(open(os.path.join(C.RESULTS_DIR, "dvfs.csv"))))
    for r in rows:
        for k in ("req_clk_mhz", "act_clk_mhz", "throughput_tok_s", "power_avg_w"):
            r[k] = float(r[k])
    return rows


def powlaw(x, y):
    a = np.polyfit(np.log(x), np.log(y), 1)
    pred = np.exp(a[1]) * x ** a[0]
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return a[0], r2


def fit_PT(T, P, P0):
    g, b = np.polyfit(np.log(T), np.log(P - P0), 1)
    k = np.exp(b)
    pred = P0 + k * T ** g
    r2 = 1 - np.sum((P - pred) ** 2) / np.sum((P - P.mean()) ** 2)
    return g, k, r2


def main():
    rows = load()
    cap = json.load(open(os.path.join(C.RESULTS_DIR, "model_info.json")))["power_cap_w"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    for wl, color in (("prefill", "C1"), ("decode", "C0")):
        r = sorted([x for x in rows if x["workload"] == wl], key=lambda x: x["act_clk_mhz"])
        req = np.array([x["req_clk_mhz"] for x in r]); act = np.array([x["act_clk_mhz"] for x in r])
        T = np.array([x["throughput_tok_s"] for x in r]); P = np.array([x["power_avg_w"] for x in r])
        capped = act < req - 30
        cl = ~capped
        aT, r2T = powlaw(act[cl], T[cl])

        # left: mechanism T vs clock
        axL.plot(act[cl], T[cl] / 1e3, "o-", color=color, label=f"{wl}:  T ∝ f^{aT:.2f}  (R²={r2T:.2f})")
        if capped.any():
            axL.scatter(act[capped], T[capped] / 1e3, facecolors="none", edgecolors=color, s=70)

        # right: P vs T, with two baselines to show the exponent's P0-sensitivity
        order = np.argsort(T[cl]); Tc, Pc = T[cl][order], P[cl][order]
        axR.plot(Tc / 1e3, Pc, "o", color=color, ms=9, label=f"{wl} (clean)")
        if capped.any():
            axR.scatter(T[capped] / 1e3, P[capped], facecolors="none", edgecolors=color, s=80,
                        label=f"{wl} cap-clipped (excluded)")
        if wl == "prefill":
            Tg = np.linspace(Tc.min(), Tc.max(), 100)
            for P0, ls, lab in ((44, "--", "P₀=44 W (deep idle)"), (90, "-", "P₀=90 W (active floor)")):
                g, k, r2 = fit_PT(Tc, Pc, P0)
                axR.plot(Tg / 1e3, P0 + k * Tg ** g, ls, color="k", lw=1.6,
                         label=f"{lab}: P=P₀+k·T^{g:.2f} (R²={r2:.3f})")

    axL.set_xlabel("achieved SM clock f (MHz)"); axL.set_ylabel("throughput T (k tok/s)")
    axL.set_title("Mechanism — throughput vs clock\nprefill T∝f (compute-bound) · decode ~flat (memory-bound)")
    axL.legend(); axL.grid(alpha=.3)
    axR.axhline(cap, color="r", ls="--", alpha=.6, label=f"power cap {cap:.0f} W")
    axR.set_xlabel("throughput T (k tok/s)"); axR.set_ylabel("GPU power P (W)")
    axR.set_title("Power vs throughput (frequency knob)\nprefill exponent depends on baseline P₀ (1.5↔3); NOT a clean cubic")
    axR.legend(fontsize=8); axR.grid(alpha=.3)
    fig.suptitle("DVFS sweep — SM clock locked 510→1530 MHz, fixed workload  (V100 / Phi-3-mini)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(C.FIGURES_DIR, "dvfs_cubic.png")
    fig.savefig(out, dpi=130); print("wrote", out)


if __name__ == "__main__":
    main()
