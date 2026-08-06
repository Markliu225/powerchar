"""Rack power capping per use-case class: OPT (caps float) vs TDP under physical constraints.

Same rack problem as rack_power_capping/solve_workloads.py — 5 kW budget, <=32 physical
GPU slots, integer GPUs with >=1 per phase, caps confined to the measured [100, 250] W, decode
never above its saturation cap — but solved for the 7 PRODUCTION workload classes of
workload_classes.csv (the paper's II-C trace-based taxonomy, rho-bar 0.83..110.7 from ServeGen /
DynamoLLM-Azure'24 / Mooncake): each class contributes its trace P:D (Lp = ratio_agg, Ld = 1)
and the fitlib curves of the measured workload it maps onto (curves_lib.MAP, nearest decode-
context anchor). Solver AND curve construction are imported from solve_workloads.py (only Lp/Ld
overridden) — nothing is re-implemented, so the two artifact sets cannot drift.

OPT = per-phase caps float to spend the budget (decode never above its saturation cap, so a
saturation-limited class can strand watts — see opt_w_used); TDP = every GPU at nameplate
250 W (5 kW fits only 20 of 32 slots). Same 5 kW, capping buys 12 more GPUs -> more tokens.

python3 solve_rack_capping.py -> fig_workload_rack_capping.png + workload_rack_capping.csv
"""
from __future__ import annotations
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)                  # workload_analysis/
ROOT = os.path.dirname(PARENT)
sys.path.insert(0, os.path.join(ROOT, "rack_power_capping"))
sys.path.insert(0, PARENT)
import solve_workloads as SW                    # noqa: E402  (also puts fitlib on sys.path)
from curves_lib import NAME, MAP, CAVEAT         # noqa: E402  (single source: II-C taxonomy)

STAR = set(CAVEAT)                              # anchor mapping carries a caveat (starred)
BANDS = [("decode-heavy", 0.0, 0.5, "#d62728"),
         ("balanced", 0.5, 2.0, "#7f7f7f"),
         ("prefill-heavy", 2.0, np.inf, "#1f77b4")]
band_of = lambda r: next(b for b in BANDS if b[1] <= r < b[2])
ratio_str = lambda x: f"{x:.1f}:1" if x >= 1 else f"{x:.2f}:1"

GREEN, RED, BLUE, ORANGE = "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"   # repo figure palette
BLUE_LT, ORANGE_LT = "#aec7e8", "#ffbb78"       # TDP (lighter step of the same hues)
INK2 = "#52514e"


def main():
    rows = list(csv.DictReader(open(os.path.join(PARENT, "workload_classes.csv"), encoding="utf-8")))
    classes = sorted([dict(klass=r["klass"], r=float(r["ratio_agg"])) for r in rows],
                     key=lambda c: c["r"])      # decode-heavy -> prefill-heavy
    by_id = {w["id"]: w for w in SW.PORTFOLIO}  # curves via THE canonical loader; Lp/Ld overridden
    flat = sorted({w for e in MAP.values() for w in (e if isinstance(e, tuple) else (e,))})
    base = {wid: SW.load_workload(by_id[wid]) for wid in flat}
    anchor = lambda e: (SW.blend_workload([base[w] for w in e]) if isinstance(e, tuple)
                        else base[e])          # tuple entry = synthetic geometric-mean anchor

    recs, out = [], []
    for cl in classes:
        c = {**anchor(MAP[cl["klass"]]), "Lp": cl["r"], "Ld": 1.0}
        o, t, ceil = SW.solve_opt(c), SW.solve_tdp(c), SW.cont_bound(c)
        binds = []
        if o["Np"] + o["Nd"] >= SW.N_GPU_MAX:
            binds.append(f"N_max={SW.N_GPU_MAX}")
        if o["Np"] == 1 and c["Lp"] < c["Ld"]:
            binds.append("Np=1")
        if o["Nd"] == 1 and c["Ld"] < c["Lp"]:
            binds.append("Nd=1")
        # rack tok/J on the MEASURED draw (Σ per-GPU power_avg_w at the chosen caps), not the
        # provisioned cap sum — consistent with the tok/J efficiency curves (power, not the set cap)
        mw = lambda r: r["Np"] * float(c["pre_pwr_of"](r["p_p"])) + r["Nd"] * float(c["dec_pwr_of"](r["p_d"]))
        o_mw, t_mw = mw(o), mw(t)
        recs.append(dict(cl=cl, o=o, t=t))
        cav = CAVEAT.get(cl["klass"], "")
        out.append({"klass": cl["klass"], "band": band_of(cl["r"])[0], "ratio_agg": cl["r"],
                    "via_workload": (lambda e: "+".join(e) if isinstance(e, tuple) else e)(MAP[cl["klass"]]), "mapping_caveat": cav,
                    "opt_tok_s": round(o["tot"], 1), "tdp_tok_s": round(t["tot"], 1),
                    "gain_pct": round(100 * (o["tot"] / t["tot"] - 1), 1),
                    "opt_N_prefill": o["Np"], "opt_N_decode": o["Nd"],
                    "opt_cap_prefill_w": round(o["p_p"]), "opt_cap_decode_w": round(o["p_d"]),
                    "opt_w_provisioned": round(o["w_used"]), "opt_w_measured": round(o_mw),
                    "tdp_N_prefill": t["Np"], "tdp_N_decode": t["Nd"],
                    "opt_rack_tok_per_j": round(o["tot"] / o_mw, 3),
                    "tdp_rack_tok_per_j": round(t["tot"] / t_mw, 3),
                    "opt_pct_of_cont_bound": round(100 * o["tot"] / ceil, 1),
                    "constraint_binds": "+".join(binds)})

    path = os.path.join(HERE, "workload_rack_capping.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        [w.writerow(r) for r in out]
    print(f"wrote {os.path.basename(path)}")

    # ---------------- figure: (a) throughput OPT vs TDP · (b) GPU count & split ----------------
    x = np.arange(len(recs))
    lab = [f"{NAME[r['cl']['klass']]}\n{ratio_str(r['cl']['r'])}" for r in recs]
    bounds = [i - 0.5 for i in range(1, len(recs))
              if band_of(recs[i]["cl"]["r"])[0] != band_of(recs[i - 1]["cl"]["r"])[0]]
    fmt_tok = lambda v: f"{v/1e3:.0f}k" if v >= 9500 else (f"{v/1e3:.1f}k" if v >= 1000
                                                           else f"{v:.0f}")

    fig, ax = plt.subplots(2, 1, figsize=(17, 13.5), gridspec_kw={"height_ratios": [1.15, 1]})

    # (a) throughput normalized to TDP — LINEAR, bar heights show the true gain
    a = ax[0]
    wdt = 0.38
    rel = [r["o"]["tot"] / r["t"]["tot"] for r in recs]
    a.bar(x - wdt / 2, rel, wdt, color=GREEN,
          label="Power Capping")
    a.bar(x + wdt / 2, [1.0] * len(recs), wdt, color=RED,
          label="TDP")
    for i, r in enumerate(recs):
        a.annotate(f"+{100 * (rel[i] - 1):.0f}%", (x[i] - wdt / 2, rel[i]),
                   textcoords="offset points", xytext=(0, 6), ha="center", fontsize=16,
                   color=GREEN, weight="bold")
        a.text(x[i] - wdt / 2, rel[i] / 2, f"{fmt_tok(r['o']['tot'])} tok/s", rotation=90,
               ha="center", va="center", fontsize=14.5, color="white", weight="bold")
        a.text(x[i] + wdt / 2, 0.5, f"{fmt_tok(r['t']['tot'])} tok/s", rotation=90,
               ha="center", va="center", fontsize=14.5, color="white", weight="bold")
    for b in bounds:
        a.axvline(b, color="lightgray", lw=0.9, zorder=0)
    a.set_xticks(x); a.set_xticklabels(lab, fontsize=16)
    for tick, r in zip(a.get_xticklabels(), recs):   # label color = P:D band (fig_workload_pd)
        tick.set_color(band_of(r["cl"]["r"])[3])
    a.set_ylim(0, max(rel) * 1.38)
    a.set_ylabel("rack throughput relative to TDP baseline (linear)", fontsize=17)
    a.set_title("(a) Rack throughput — Power Capping vs TDP", fontsize=18)
    a.legend(fontsize=15.5, loc="upper left"); a.tick_params(axis="y", labelsize=16); a.grid(alpha=.3, axis="y")

    # (b) GPU count change: TDP (left, light) vs OPT (right, solid), stacked prefill/decode
    a = ax[1]
    NpT = [r["t"]["Np"] for r in recs]; NdT = [r["t"]["Nd"] for r in recs]
    NpO = [r["o"]["Np"] for r in recs]; NdO = [r["o"]["Nd"] for r in recs]
    a.bar(x - wdt / 2, NpT, wdt, color=BLUE_LT)
    a.bar(x - wdt / 2, NdT, wdt, bottom=NpT, color=ORANGE_LT)
    a.bar(x + wdt / 2, NpO, wdt, color=BLUE)
    a.bar(x + wdt / 2, NdO, wdt, bottom=NpO, color=ORANGE)
    a.axhline(SW.N_GPU_MAX, color="k", ls="--", lw=1.2)
    a.text(-0.42, SW.N_GPU_MAX + 6.4, f"physical slot limit N_max={SW.N_GPU_MAX}",
           ha="left", fontsize=16, weight="bold")
    for b in bounds:
        a.axvline(b, color="lightgray", lw=0.9, zorder=0)
    for i, r in enumerate(recs):
        a.text(i - wdt / 2, NpT[i] + NdT[i] + 0.7, f"{NpT[i]}+{NdT[i]}", ha="center",
               fontsize=15, color=INK2)
        a.text(i + wdt / 2, NpO[i] + NdO[i] + 0.7,
               f"{NpO[i]}+{NdO[i]}\n@{r['o']['p_p']:.0f}/{r['o']['p_d']:.0f}W",
               ha="center", fontsize=15)
    a.set_xticks(x); a.set_xticklabels(lab, fontsize=16)
    for tick, r in zip(a.get_xticklabels(), recs):
        tick.set_color(band_of(r["cl"]["r"])[3])
    a.set_ylabel(f"GPUs in the {SW.W_RACK/1e3:.0f} kW rack", fontsize=17)
    a.set_title("(b) GPU count and phase split", fontsize=18)
    a.legend(handles=[Patch(fc=BLUE, label="prefill GPUs (Power Capping)"),
                      Patch(fc=ORANGE, label="decode GPUs (Power Capping)"),
                      Patch(fc=BLUE_LT, label="prefill GPUs (TDP)"),
                      Patch(fc=ORANGE_LT, label="decode GPUs (TDP)")],
             fontsize=15.5, loc="upper right", ncols=2)
    a.tick_params(axis="y", labelsize=16); a.grid(alpha=.3, axis="y")
    a.set_ylim(0, SW.N_GPU_MAX * 1.48)

    fig.suptitle("V100 rack power capping by production workload class — trace P:D ratios on the "
                 "mapped workload curves", fontsize=19)
    fig.text(0.5, 0.005,
             "classes ordered decode-heavy -> prefill-heavy; label color = P:D band (red decode-heavy / "
             "gray balanced / blue prefill-heavy)\n"
             "P:D = trace aggregate ratio (ServeGen NSDI'26 · DynamoLLM-Azure'24 HPCA'25 · Mooncake FAST'25)",
             ha="center", fontsize=14, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    outp = os.path.join(HERE, "fig_workload_rack_capping.png")
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    print(f"wrote {os.path.basename(outp)}")

    hdr = f"{'class':<18}{'P:D':>8} |{'OPT k/s':>9}{'Np':>4}{'Nd':>4}{'pp':>5}{'pd':>5} |{'TDP k/s':>9}{'Np':>4}{'Nd':>4} |{'gain':>7}"
    print("\n" + hdr)
    for r, ln in zip(recs, out):
        print(f"{NAME[r['cl']['klass']]:<18}{ratio_str(r['cl']['r']):>8} |{ln['opt_tok_s']/1e3:>9.2f}"
              f"{ln['opt_N_prefill']:>4}{ln['opt_N_decode']:>4}{ln['opt_cap_prefill_w']:>5}"
              f"{ln['opt_cap_decode_w']:>5} |{ln['tdp_tok_s']/1e3:>9.2f}{ln['tdp_N_prefill']:>4}"
              f"{ln['tdp_N_decode']:>4} |{ln['gain_pct']:>6.1f}%")


if __name__ == "__main__":
    main()
