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
import palette                                  # noqa: E402  (single source: the paper palette)

STAR = set(CAVEAT)                              # anchor mapping carries a caveat (starred)
BANDS = [(b, lo, hi, palette.BAND_C[b]) for b, lo, hi in
         [("decode-heavy", 0.0, 0.5), ("balanced", 0.5, 2.0), ("prefill-heavy", 2.0, np.inf)]]
band_of = lambda r: next(b for b in BANDS if b[1] <= r < b[2])
ratio_str = lambda x: f"{x:.1f}:1" if x >= 1 else f"{x:.2f}:1"

GREEN, RED = palette.OK, palette.BAD            # capped vs TDP (the semantic pair)
BLUE, ORANGE = palette.PRE_C, palette.DEC_C     # prefill vs decode GPU counts
BLUE_LT, ORANGE_LT = palette.PAL["cyan"], palette.PAL["sand"]   # TDP: lighter step of both hues
INK2 = palette.INK2


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
    lab = [NAME[r['cl']['klass']] for r in recs]      # name only, in black: the P:D ratio is in
                                                     # the table and the band shows as the dividers
    bounds = [i - 0.5 for i in range(1, len(recs))
              if band_of(recs[i]["cl"]["r"])[0] != band_of(recs[i - 1]["cl"]["r"])[0]]

    fig, ax = plt.subplots(2, 1, figsize=(12.5, 15.0), gridspec_kw={"height_ratios": [1, 1.05]})

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
                   textcoords="offset points", xytext=(0, 8), ha="center", fontsize=28,
                   color=GREEN, weight="bold")
    for b in bounds:
        a.axvline(b, color="lightgray", lw=0.9, zorder=0)
    a.set_xticks(x); a.set_xticklabels([])       # categories are labelled once, under panel (b):
                                                 # both panels share the column, and at this type
                                                 # size labelling twice would eat the figure
    a.set_ylim(0, max(rel) * 1.26)   # just enough for the "+x%" labels and the key above
                                     # them: 1.62 left the bars floating in white space
    a.set_ylabel("throughput vs TDP", fontsize=24)
    a.set_title("(a) Rack throughput — Power Capping vs TDP", fontsize=30)
    a.tick_params(axis="y", labelsize=26); a.grid(alpha=.3, axis="y")

    # (b) GPU count change: TDP (left, light) vs OPT (right, solid), stacked prefill/decode
    a = ax[1]
    NpT = [r["t"]["Np"] for r in recs]; NdT = [r["t"]["Nd"] for r in recs]
    NpO = [r["o"]["Np"] for r in recs]; NdO = [r["o"]["Nd"] for r in recs]
    a.bar(x - wdt / 2, NpT, wdt, color=BLUE_LT)
    a.bar(x - wdt / 2, NdT, wdt, bottom=NpT, color=ORANGE_LT)
    a.bar(x + wdt / 2, NpO, wdt, color=BLUE)
    a.bar(x + wdt / 2, NdO, wdt, bottom=NpO, color=ORANGE)
    a.axhline(SW.N_GPU_MAX, color="k", ls="--", lw=1.2)
    a.text(-0.45, SW.N_GPU_MAX - 1.2, f"N_max={SW.N_GPU_MAX}", ha="left", va="top",
           fontsize=24, weight="bold")
    for b in bounds:
        a.axvline(b, color="lightgray", lw=0.9, zorder=0)
    for i, r in enumerate(recs):
        a.text(i - wdt / 2, NpT[i] + NdT[i] + 0.7, f"{NpT[i]}+{NdT[i]}", ha="center",
               fontsize=19, color=INK2)
        a.text(i + wdt / 2, SW.N_GPU_MAX + 1.0,
               f"{NpO[i]}+{NdO[i]}\n{r['o']['p_p']:.0f}/{r['o']['p_d']:.0f}W",
               ha="center", fontsize=19)      # 19 pt: at 23 the caps line of one class
                                             # ran into the next one at this width
    a.set_xticks(x)
    a.set_xticklabels(lab, fontsize=29, color="black", rotation=40, ha="right",
                      rotation_mode="anchor")   # slanted: no class name fits horizontally this big
    a.set_ylabel(f"GPUs ({SW.W_RACK/1e3:.0f} kW rack)", fontsize=24)
    a.set_title("(b) GPU count and phase split", fontsize=30)
    a.tick_params(axis="y", labelsize=26); a.grid(alpha=.3, axis="y")
    a.set_ylim(0, SW.N_GPU_MAX * 1.30)

    # ONE key for both panels, below everything: a key inside a panel forces an empty band above
    # the bars for it to sit in — which is exactly the whitespace this figure did not want.
    fig.legend(handles=[Patch(fc=GREEN, label="(a) Power Capping"),
                        Patch(fc=RED, label="(a) TDP"),
                        Patch(fc=BLUE, label="(b) prefill — capped"),
                        Patch(fc=ORANGE, label="(b) decode — capped"),
                        Patch(fc=BLUE_LT, label="(b) prefill — TDP"),
                        Patch(fc=ORANGE_LT, label="(b) decode — TDP")],
               loc="lower center", ncol=2, fontsize=27, frameon=False, bbox_to_anchor=(0.5, 0.105),
               columnspacing=0.8, handletextpad=0.4, handlelength=1.3)
    fig.suptitle("V100 rack power capping by production workload class\n"
                 "trace P:D ratios on the mapped workload curves", fontsize=25)
    fig.text(0.5, 0.005,
             "classes ordered decode-heavy -> prefill-heavy; dividers mark the band boundaries\n"
             "under each capped bar: prefill+decode GPUs, and their prefill/decode caps\n"
             "P:D = trace aggregate ratio (ServeGen NSDI'26 · DynamoLLM-Azure'24 · Mooncake FAST'25)",
             ha="center", fontsize=19, color=INK2)
    fig.tight_layout(rect=(0, 0.235, 1, 1))
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
