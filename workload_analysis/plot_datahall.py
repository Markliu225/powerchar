"""Data-hall (1 MW) scale-up of the per-rack V-C results — Power Capping vs TDP on three platforms.

(a) per platform, hall throughput normalized to the TDP design (red = 1); the green bar's gain
    percentage is the paper's hall-level headline number, absolute tok/s printed inside the bars.
(b) per platform, TWO stacked bars (left = TDP, right = Power Capping): how the hall's racks are
    split across the seven production classes under each design. Power capping raises the per-rack
    output of the high-gain classes, so they need FEWER racks and the freed racks flow to the
    classes with the largest token share.

METHOD (each step deliberate, disclosed in the paper text):
 1. HALL LAYOUT: the SAME 128 rack slots on every platform (the NVIDIA DGX SuperPOD reference
    architecture tops out at a 128-rack configuration — citable). Hall power then follows the
    platform: V100 128x5 kW = 640 kW, RTX 5090 128x11.5 = ~1.47 MW, H200 128x14 = ~1.79 MW, all
    near the common 1-5 MW wholesale colocation lease range. Rack internals reuse the V-C per-class
    results as-is: TDP design = 20 GPUs at full speed per rack, Power Capping = Algorithm 1's
    output; nothing is re-solved at hall scale.
 2. PER-RACK THROUGHPUT per class x design is taken from the V-C rack results
    ({v100,5090,h200}/workload_rack_capping.csv, opt_tok_s / tdp_tok_s) — nothing is re-solved.
 3. LOAD MIX: the repo's research-grounded 7-class REQUEST shares (plot_profit_model.R_SHARE are
    request-count shares), converted to TOKEN demand w_j by multiplying each class's mean request
    length (Lp_mean + Ld_mean, workload_classes.csv) and renormalizing. Same mix for both designs.
 4. RACK ALLOCATION, independently per design: one rack per class, then every remaining rack ONE
    AT A TIME to the current bottleneck class (smallest sustainable hall load N_j·X_j/w_j) until
    the 128 slots are full — max-min optimal; converges to N_j ∝ w_j/X_j by itself (see allocate()
    for why a proportional pre-allocation was rejected).
 5. GAIN: hall throughput = the largest mixed load the hall sustains, T = min_j N_j·X_j/w_j
    (w normalized, so T is also the total token output). Gain = T_cap / T_tdp - 1.

python3 workload_analysis/plot_datahall.py -> fig_datahall.png + datahall.csv
"""
from __future__ import annotations
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_profit_model import R_SHARE           # noqa: E402  request-count shares (research-grounded)
from curves_lib import NAME                     # noqa: E402  class display names

N_RACKS = 128                                   # SuperPOD reference architecture: 128-rack layout
PLATFORMS = [("V100", "v100", 5.0), ("RTX 5090", "5090", 11.5), ("H200", "h200", 14.0)]
GREEN, RED, INK2 = "#2ca02c", "#d62728", "#52514e"
fmt_T = lambda v: f"{v/1e6:.1f}M tok/s" if v >= 1e6 else f"{v/1e3:.0f}k tok/s"


def load_mix():
    """Token-demand shares w_j: REQUEST shares x mean request length (Lp+Ld), renormalized.
    R_SHARE is request-count based (ServeGen request counts, Copilot completions/day, ...), so the
    length multiplication is required; a token-based mix would skip it."""
    rows = list(csv.DictReader(open(os.path.join(HERE, "workload_classes.csv"), encoding="utf-8")))
    L = {r["klass"]: float(r["Lp_mean"]) + float(r["Ld_mean"]) for r in rows}
    vol = {k: R_SHARE[k] * L[k] for k in R_SHARE}
    z = sum(vol.values())
    return {k: v / z for k, v in vol.items()}


def allocate(n_total, w, X):
    """Integer rack allocation, max-min optimal: one rack per class to start, then every remaining
    rack goes to the CURRENT BOTTLENECK class (smallest sustainable load N_j*X_j/w_j) — the only
    class whose extra rack raises the hall's serviceable mix. This pure greedy provably maximizes
    min_j N_j*X_j/w_j and lands within ~2% of the continuous optimum on all six design points.

    A proportional-floor pre-allocation (floor(N*share) each) was tried first and REJECTED: floors
    strand racks on non-bottleneck classes that the fill can never reclaim, costing up to 11% of
    hall throughput — and unevenly between the two designs, which distorted the gain by up to
    12 points (H200 read +37% against a +24.5% continuous bound). Greedy converges to the same
    proportional split on its own, without the stranding."""
    N = {k: 1 for k in w}
    while sum(N.values()) < n_total:
        k = min(w, key=lambda k: N[k] * X[k] / w[k])
        N[k] += 1
    return N


def hall_throughput(N, w, X):
    """Largest sustainable mixed load: T = min_j N_j*X_j/w_j (w normalized -> T = total tok/s)."""
    return min(N[k] * X[k] / w[k] for k in w)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    w = load_mix()
    classes = sorted(w, key=lambda k: R_SHARE[k], reverse=False)     # plotting order fixed below
    # class order & colors: by P:D ratio (the taxonomy's canonical order), tab10 — fixed, never cycled
    rows0 = list(csv.DictReader(open(os.path.join(HERE, "workload_classes.csv"), encoding="utf-8")))
    order = [r["klass"] for r in sorted(rows0, key=lambda r: float(r["ratio_agg"]))]
    cmap = plt.get_cmap("tab10")
    ccol = {k: cmap(i % 10) for i, k in enumerate(order)}

    res, out_rows = {}, []
    for label, sub, rack_kw in PLATFORMS:
        n_racks = N_RACKS
        hall_mw = n_racks * rack_kw / 1e3
        rr = {r["klass"]: r for r in csv.DictReader(
            open(os.path.join(HERE, sub, "workload_rack_capping.csv"), encoding="utf-8"))}
        X = {"cap": {k: float(rr[k]["opt_tok_s"]) for k in w},
             "tdp": {k: float(rr[k]["tdp_tok_s"]) for k in w}}
        N = {d: allocate(n_racks, w, X[d]) for d in ("cap", "tdp")}
        T = {d: hall_throughput(N[d], w, X[d]) for d in ("cap", "tdp")}
        res[label] = dict(n_racks=n_racks, hall_mw=hall_mw, N=N, T=T,
                          gain=T["cap"] / T["tdp"] - 1.0)
        for k in order:
            out_rows.append(dict(platform=label, klass=k, token_share=round(w[k], 4),
                                 racks_tdp=N["tdp"][k], racks_cap=N["cap"][k],
                                 rack_tok_s_tdp=X["tdp"][k], rack_tok_s_cap=X["cap"][k]))
        out_rows.append(dict(platform=label, klass="TOTAL", token_share=1.0,
                             racks_tdp=n_racks, racks_cap=n_racks,
                             rack_tok_s_tdp=round(T["tdp"], 1), rack_tok_s_cap=round(T["cap"], 1)))
        print(f"{label:9s} {n_racks:3d} racks ({hall_mw:.2f} MW) · hall T: TDP {fmt_T(T['tdp'])} "
              f"-> CAP {fmt_T(T['cap'])}  gain +{100*res[label]['gain']:.0f}%")

    with open(os.path.join(HERE, "datahall.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wr.writeheader()
        [wr.writerow(r) for r in out_rows]
    print("wrote datahall.csv")

    # ---------------------------------- figure: panel (a) only, compact ---------------------------
    x = np.arange(len(PLATFORMS))
    labels = [p[0] for p in PLATFORMS]
    fig, a = plt.subplots(figsize=(7.6, 5.2))
    wdt = 0.38
    rel = [res[l]["T"]["cap"] / res[l]["T"]["tdp"] for l in labels]
    a.bar(x - wdt / 2, rel, wdt, color=GREEN, label="Power Capping")
    a.bar(x + wdt / 2, [1.0] * len(labels), wdt, color=RED, label="TDP")
    for i, l in enumerate(labels):
        a.annotate(f"+{100 * (rel[i] - 1):.0f}%", (x[i] - wdt / 2, rel[i]),
                   textcoords="offset points", xytext=(0, 5), ha="center", fontsize=16,
                   color=GREEN, weight="bold")
        a.text(x[i] - wdt / 2, rel[i] / 2, fmt_T(res[l]["T"]["cap"]), rotation=90,
               ha="center", va="center", fontsize=12, color="white", weight="bold")
        a.text(x[i] + wdt / 2, 0.5, fmt_T(res[l]["T"]["tdp"]), rotation=90,
               ha="center", va="center", fontsize=12, color="white", weight="bold")
    a.set_xticks(x)
    a.set_xticklabels([f"{l}\n{res[l]['hall_mw']:.2f} MW" for l in labels], fontsize=13)
    a.set_ylim(0, max(rel) * 1.24)
    a.set_ylabel("hall throughput relative to the TDP design", fontsize=13)
    a.set_title("128-rack data hall — Power Capping vs TDP\n"
                "(same layout on every platform; ⚠ RTX 5090 = MOCK data)",
                fontsize=14)
    a.legend(fontsize=12, loc="upper right")
    a.tick_params(axis="y", labelsize=12); a.grid(alpha=.3, axis="y")
    fig.tight_layout()
    out = os.path.join(HERE, "fig_datahall.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {os.path.basename(out)}")


if __name__ == "__main__":
    main()
