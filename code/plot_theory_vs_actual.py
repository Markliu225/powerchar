"""Theory vs. actual: the P(T) laws against the measured points.

Theory (POWER_THROUGHPUT_MODEL.md, first-principles, NO data fitting):
  prefill  P = P0 + k_c * T^3   (compute-bound; core V^2 f, V∝f)
  decode   P = P0 + k_m * T     (memory-bound; fixed energy/bit)

Honest caveat shown on the figure: our measurements are a CONCURRENCY (batch) sweep
at a PINNED ~1530 MHz clock, not a frequency sweep.
  - Decode stays un-capped and un-saturated, so its measured points DO trace a line
    P ∝ T  -> the linear law is directly confirmed.
  - Prefill saturates the compute units at low batch and hits the 250 W POWER CAP,
    so its points sit FLAT at the cap and cannot traverse the cubic. The cubic is the
    *frequency-knob* law (raising T by clocking up costs f^3); the cap forbids it and
    we cannot lock the clock (no root). We draw the cubic as the would-be law and show
    it meeting the cap at the measured saturation throughput.

  python code/plot_theory_vs_actual.py
"""
from __future__ import annotations
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

P0 = 44.0            # measured idle/static floor (W)  -- nvidia-smi idle ~44.7 W


def load(name):
    with open(os.path.join(C.RESULTS_DIR, name)) as f:
        rows = [{k: float(v) for k, v in r.items() if k not in ("phase",)} for r in csv.DictReader(f)]
    return sorted(rows, key=lambda r: r["batch"])


def main():
    pre = load("prefill.csv")
    dec = load("decode.csv")
    cap = json.load(open(os.path.join(C.RESULTS_DIR, "model_info.json")))["power_cap_w"]

    Tp = np.array([r["throughput_tok_s"] for r in pre]); Pp = np.array([r["power_avg_w"] for r in pre])
    Bp = np.array([r["batch"] for r in pre])
    Td = np.array([r["throughput_tok_s"] for r in dec]); Pd = np.array([r["power_avg_w"] for r in dec])
    Bd = np.array([r["batch"] for r in dec])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))

    # ---------- Panel A: PREFILL — cubic theory vs capped measurement ----------
    T_sat = Tp.max()                                   # measured saturation throughput
    k_c = (cap - P0) / T_sat**3                        # anchor cubic to meet the cap at T_sat
    Tg = np.linspace(0, T_sat * 1.02, 400)
    axA.plot(Tg, P0 + k_c * Tg**3, "C3-", lw=2.2,
             label=r"theory (frequency knob):  $P = P_0 + k_c T^3$")
    axA.axhline(cap, color="r", ls="--", alpha=.6, label=f"power cap {cap:.0f} W")
    axA.fill_between([0, T_sat * 1.02], cap, cap * 1.08, color="r", alpha=.06)
    axA.scatter(Tp, Pp, c="C1", s=95, zorder=5, edgecolor="k", linewidth=.4,
                label="measured (batch sweep, clock pinned ~1530 MHz)")
    for t, p, b in zip(Tp, Pp, Bp):
        axA.annotate(f"b{int(b)}", (t, p), fontsize=7, xytext=(4, 4), textcoords="offset points")
    axA.axvline(T_sat, color="gray", ls=":", lw=1)
    axA.annotate("cubic meets the cap →\nthroughput saturates here",
                 (T_sat, cap * 0.74), fontsize=8, ha="right", color="dimgray")
    axA.annotate("measured points ride the CAP (flat),\n"
                 "pinned at max clock ⇒ they do NOT\ntrace the cubic (needs a clock sweep)",
                 (T_sat * 0.40, cap * 0.93), fontsize=8.5, color="C1")
    axA.set_xlabel("token throughput  T  (tok/s)"); axA.set_ylabel("GPU power  P  (W)")
    axA.set_ylim(0, cap * 1.10); axA.set_xlim(0, T_sat * 1.05)
    axA.set_title("PREFILL  (compute-bound):  theory  P ∝ T³")
    axA.legend(loc="lower right", fontsize=8.5); axA.grid(alpha=.3)

    # ---------- Panel B: DECODE — linear theory vs measurement ----------
    clean = (Bd <= 32)                                 # exclude thermal-edge b=64
    Tc, Pc = Td[clean], Pd[clean]
    (b0, b1), *_ = np.linalg.lstsq(np.c_[np.ones_like(Tc), Tc], Pc, rcond=None)
    pred = b0 + b1 * Tc
    ss = 1 - np.sum((Pc - pred)**2) / np.sum((Pc - Pc.mean())**2)
    Tg2 = np.linspace(0, Td.max() * 1.05, 200)
    axB.plot(Tg2, b0 + b1 * Tg2, "C3-", lw=2.2,
             label=f"theory  P = P₀ + k_m·T  (linear)\nfit: P = {b0:.0f} + {b1:.3f}·T,  R²={ss:.3f}")
    axB.axhline(cap, color="r", ls="--", alpha=.6, label=f"power cap {cap:.0f} W")
    axB.scatter(Td[clean], Pd[clean], c="C0", s=95, zorder=5, edgecolor="k", linewidth=.4,
                label="measured (batch sweep)")
    axB.scatter(Td[~clean], Pd[~clean], facecolors="none", edgecolors="C0", s=95, zorder=5,
                label="b=64 (thermal-edge, excluded)")
    for t, p, b in zip(Td, Pd, Bd):
        axB.annotate(f"b{int(b)}", (t, p), fontsize=7, xytext=(4, 4), textcoords="offset points")
    axB.annotate("measured points DO trace a line\n⇒ linear law  P ∝ T  confirmed",
                 (Td.max() * 0.05, cap * 0.86), fontsize=8.5, color="C0")
    axB.set_xlabel("token throughput  T  (tok/s)"); axB.set_ylabel("GPU power  P  (W)")
    axB.set_ylim(0, cap * 1.10); axB.set_xlim(0, Td.max() * 1.05)
    axB.set_title("DECODE  (memory-bound):  theory  P ∝ T")
    axB.legend(loc="lower right", fontsize=8.5); axB.grid(alpha=.3)

    fig.suptitle("Analytical P(T) laws vs. measured V100 / Phi-3-mini  "
                 "(measurements = concurrency sweep at fixed clock)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(C.FIGURES_DIR, "theory_vs_actual.png")
    fig.savefig(out, dpi=130); print("wrote", out)
    print(f"[decode] linear fit P = {b0:.1f} + {b1:.4f}*T  R2={ss:.4f}  (B<=32)")
    print(f"[prefill] cubic anchored k_c={k_c:.3e}, meets {cap:.0f}W at T_sat={T_sat:.0f} tok/s")


if __name__ == "__main__":
    main()
