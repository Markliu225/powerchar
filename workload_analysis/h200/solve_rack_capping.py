"""Rack power capping per use-case class on H200: OPT vs TDP under physical constraints.

Same pipeline as ../solve_rack_capping.py (V100), retargeted at the H200 dataset and an
H200-scaled rack scenario. Solver AND curve construction are imported from the canonical
rack_power_capping/v100/solve_workloads.py; this file only retargets its scenario globals
(the functions read them at call time) and overrides Lp/Ld with each class's measured ratio.

SCENARIO (H200-scaled from the V100 5 kW / 32-slot experiment, disclosed on the figure):
  W_RACK   = 14 kW   -- the V100 budget scaled by the TDP ratio 700/250, so the slot-wall
                        tension is identical: nameplate TDP affords exactly 20 of 32 slots
  N_GPU_MAX = 32 slots;  P_TDP = 700 W;  caps confined to the MEASURED [200, 700] W
  decode never above its saturation cap (0.995 of T at 700 W), integer GPUs, >=1 per phase

With the v-next H200 data (prefill clock-swept, so its efficiency sweet spot lands ~530-575 W
in range), the 32-slot wall binds for all classes present — prefill no longer needs ~650-700 W,
so even Code fills the slots (16+16) rather than being budget-bound. Classes whose mapped workload
has no H200 data are skipped automatically (the subtitle names any that were). As of 2026-07-15 all
10 use-case classes are present (classify-qwen7b re-measured -> Extract back). NOTE: classify-qwen7b
prefill is still v3 cap-swept (under-enforces <367 W) and its decode is near-flat (fit R^2<0), so
Extract's curves are the shakiest of the 10 — see h200/README.md; its recipe operates in the
enforced region so the +59% gain (decode-bottleneck packing) is structural, not an artifact.

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
sys.path.insert(0, os.path.join(ROOT, "rack_power_capping", "v100"))
sys.path.insert(0, PARENT)
import solve_workloads as SW                    # noqa: E402  THE rack solver (also adds fitlib)
import plot_power_curves as V                   # noqa: E402  shared taxonomy / mapping / caveats

# ---- retarget the canonical solver at the H200 dataset + scenario ------------------------------
# SW's functions read DATA / F_MAX / P_TDP / CAP_LO / CAP_HI as module globals at call time;
# W_RACK / N_GPU_MAX are def-time default args, so they are passed explicitly below.
W_RACK, N_GPU_MAX = 14000.0, 32                 # 5 kW x (700/250); TDP -> exactly 20 of 32 slots
SW.DATA = os.path.join(ROOT, "data_h200")
SW.F_MAX = SW.fitlib.resolve_f_max(SW.DATA)
SW.P_TDP, SW.CAP_LO, SW.CAP_HI = 700.0, 200.0, 700.0
SW.sweet_spot.__defaults__ = (SW.CAP_HI,)       # hi= default was bound to 250 at def time

NAME, MAP, BANDS = V.NAME, V.MAP, V.BANDS
CAVEAT = {**V.CAVEAT,                           # Extract fit note updated to the H200 numbers
          "Extract 信息抽取": "Dolly 3.1:1 on production-scale curves (class shape 100:1); "
                              "decode fit poor on H200 (R2<0, relRMSE ~23%) - read the dots"}
band_of = lambda r: next(b for b in BANDS if b[1] <= r < b[2])
ratio_str = lambda x: f"{x:.1f}:1" if x >= 1 else f"1:{1/x:.0f}"

GREEN, RED, BLUE, ORANGE = "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"   # repo figure palette
BLUE_LT, ORANGE_LT = "#aec7e8", "#ffbb78"       # TDP (lighter step of the same hues)
INK2 = "#52514e"


def main():
    rows = list(csv.DictReader(open(os.path.join(PARENT, "workload_ratios.csv"))))
    avail = lambda wid: os.path.exists(os.path.join(SW.DATA, f"{wid}_prefill.csv"))
    classes = sorted([dict(klass=r["klass"], r=float(r["ratio_agg"])) for r in rows
                      if avail(MAP[r["klass"]])], key=lambda c: c["r"])   # Extract dropped: classify-qwen7b absent
    by_id = {w["id"]: w for w in SW.PORTFOLIO}  # curves via THE canonical loader; Lp/Ld overridden
    base = {wid: SW.load_workload(by_id[wid]) for wid in sorted({MAP[c["klass"]] for c in classes})}

    recs, out = [], []
    for cl in classes:
        c = {**base[MAP[cl["klass"]]], "Lp": cl["r"], "Ld": 1.0}
        o = SW.solve_opt(c, W=W_RACK, n_max=N_GPU_MAX)
        t = SW.solve_tdp(c, W=W_RACK, n_max=N_GPU_MAX)
        ceil = SW.cont_bound(c, W=W_RACK)
        binds = []
        if o["Np"] + o["Nd"] >= N_GPU_MAX:
            binds.append(f"N_max={N_GPU_MAX}")
        if o["Np"] == 1 and c["Lp"] < c["Ld"]:
            binds.append("Np=1")
        if o["Nd"] == 1 and c["Ld"] < c["Lp"]:
            binds.append("Nd=1")
        # rack tok/J on the MEASURED draw (Σ per-GPU power_avg_w at the chosen caps), not the
        # provisioned cap sum — consistent with the tok/J efficiency curves (power, not the set cap)
        mw = lambda r: r["Np"] * float(c["pre_pwr_of"](r["p_p"])) + r["Nd"] * float(c["dec_pwr_of"](r["p_d"]))
        o_mw, t_mw = mw(o), mw(t)
        recs.append(dict(cl=cl, o=o, t=t))
        out.append({"klass": cl["klass"], "band": band_of(cl["r"])[0], "ratio_agg": cl["r"],
                    "via_workload": MAP[cl["klass"]],
                    "mapping_caveat": CAVEAT.get(cl["klass"], ""),
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
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        [w.writerow(r) for r in out]
    print(f"wrote {os.path.basename(path)}")

    # ---------------- figure: (a) throughput OPT vs TDP · (b) GPU count & split ----------------
    x = np.arange(len(recs))
    lab = [f"{NAME[r['cl']['klass']]}{'*' if r['cl']['klass'] in CAVEAT else ''}\n"
           f"{ratio_str(r['cl']['r'])}" for r in recs]
    bounds = [i - 0.5 for i in range(1, len(recs))
              if band_of(recs[i]["cl"]["r"])[0] != band_of(recs[i - 1]["cl"]["r"])[0]]
    fmt_tok = lambda v: f"{v/1e3:.0f}k" if v >= 9500 else (f"{v/1e3:.1f}k" if v >= 1000
                                                           else f"{v:.0f}")

    fig, ax = plt.subplots(2, 1, figsize=(13, 10.5), gridspec_kw={"height_ratios": [1.15, 1]})

    # (a) throughput normalized to TDP — LINEAR, bar heights show the true gain
    a = ax[0]
    wdt = 0.38
    rel = [r["o"]["tot"] / r["t"]["tot"] for r in recs]
    a.bar(x - wdt / 2, rel, wdt, color=GREEN,
          label="OPT — caps float, <=14 kW (decode capped at saturation), slot-aware")
    a.bar(x + wdt / 2, [1.0] * len(recs), wdt, color=RED,
          label="TDP baseline (= 1.0, every GPU @700 W)")
    for i, r in enumerate(recs):
        a.annotate(f"+{100 * (rel[i] - 1):.0f}%", (x[i] - wdt / 2, rel[i]),
                   textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5,
                   color=GREEN, weight="bold")
        a.text(x[i] - wdt / 2, rel[i] / 2, f"{fmt_tok(r['o']['tot'])} tok/s", rotation=90,
               ha="center", va="center", fontsize=7, color="white", weight="bold")
        a.text(x[i] + wdt / 2, 0.5, f"{fmt_tok(r['t']['tot'])} tok/s", rotation=90,
               ha="center", va="center", fontsize=7, color="white", weight="bold")
    for b in bounds:
        a.axvline(b, color="lightgray", lw=0.9, zorder=0)
    a.set_xticks(x); a.set_xticklabels(lab, fontsize=8)
    for tick, r in zip(a.get_xticklabels(), recs):   # label color = P:D band (fig_workload_pd)
        tick.set_color(band_of(r["cl"]["r"])[3])
    a.set_ylim(0, max(rel) * 1.27)
    a.set_ylabel("rack throughput relative to TDP baseline (linear)")
    a.set_title(f"Rack throughput per use-case class on H200 — OPT vs TDP, normalized to TDP   "
                f"({W_RACK/1e3:.0f} kW, <= {N_GPU_MAX} GPU slots, caps in "
                f"[{SW.CAP_LO:.0f},{SW.CAP_HI:.0f}] W)\nbar heights show the TRUE gain; "
                "absolute tok/s printed on the bars (classes span orders of magnitude)")
    a.legend(fontsize=9, loc="upper left"); a.grid(alpha=.3, axis="y")

    # (b) GPU count change: TDP (left, light) vs OPT (right, solid), stacked prefill/decode
    a = ax[1]
    NpT = [r["t"]["Np"] for r in recs]; NdT = [r["t"]["Nd"] for r in recs]
    NpO = [r["o"]["Np"] for r in recs]; NdO = [r["o"]["Nd"] for r in recs]
    a.bar(x - wdt / 2, NpT, wdt, color=BLUE_LT)
    a.bar(x - wdt / 2, NdT, wdt, bottom=NpT, color=ORANGE_LT)
    a.bar(x + wdt / 2, NpO, wdt, color=BLUE)
    a.bar(x + wdt / 2, NdO, wdt, bottom=NpO, color=ORANGE)
    a.axhline(N_GPU_MAX, color="k", ls="--", lw=1.2)
    a.text(-0.42, N_GPU_MAX + 3.6, f"physical slot limit N_max={N_GPU_MAX}",
           ha="left", fontsize=9, weight="bold")
    for b in bounds:
        a.axvline(b, color="lightgray", lw=0.9, zorder=0)
    for i, r in enumerate(recs):
        a.text(i - wdt / 2, NpT[i] + NdT[i] + 0.7, f"{NpT[i]}+{NdT[i]}", ha="center",
               fontsize=7.5, color=INK2)
        a.text(i + wdt / 2, NpO[i] + NdO[i] + 0.7,
               f"{NpO[i]}+{NdO[i]}\n@{r['o']['p_p']:.0f}/{r['o']['p_d']:.0f}W",
               ha="center", fontsize=7.5)
    a.set_xticks(x); a.set_xticklabels(lab, fontsize=8)
    for tick, r in zip(a.get_xticklabels(), recs):
        tick.set_color(band_of(r["cl"]["r"])[3])
    a.set_ylabel(f"GPUs in the {W_RACK/1e3:.0f} kW rack")
    n_wall = sum(1 for r in recs if r["o"]["Np"] + r["o"]["Nd"] >= N_GPU_MAX)
    n_min = min(r["o"]["Np"] + r["o"]["Nd"] for r in recs)
    present = {r["cl"]["klass"] for r in recs}
    dropped = [NAME[r["klass"]] for r in rows if r["klass"] not in present]
    drop_note = f"  ·  {', '.join(dropped)} omitted (no H200 data)" if dropped else \
                f"  ·  all {len(recs)}/10 use-case classes present"
    a.set_title("GPU count & phase split per class — left bar TDP (every GPU @700 W), right bar "
                f"OPT (Np+Nd @pre/dec cap W)\nsame 14 kW: TDP affords 20 GPUs; OPT hits the "
                f"32-slot wall in {n_wall}/{len(recs)} classes (min fleet {n_min}); "
                f"split follows the TOKEN-COST ratio{drop_note}")
    a.legend(handles=[Patch(fc=BLUE, label="prefill GPUs (OPT)"),
                      Patch(fc=ORANGE, label="decode GPUs (OPT)"),
                      Patch(fc=BLUE_LT, label="prefill GPUs (TDP)"),
                      Patch(fc=ORANGE_LT, label="decode GPUs (TDP)")],
             fontsize=8.5, loc="upper right", ncols=2)
    a.grid(alpha=.3, axis="y")
    a.set_ylim(0, N_GPU_MAX * 1.3)

    fig.suptitle("H200 rack power capping by use-case class — measured P:D ratios on the mapped "
                 "workload curves", fontsize=14)
    fig.text(0.5, 0.005,
             "classes ordered decode-heavy -> prefill-heavy; label color = P:D band (red decode-heavy / "
             "gray balanced / blue prefill-heavy, as in fig_workload_pd.png); P:D = aggregate ratio from "
             "workload_ratios.csv\nscenario scaled from the V100 experiment by the TDP ratio 700/250 "
             "(5 kW -> 14 kW, same 32 slots); curves data_h200, optimizer identical to "
             "rack_power_capping/v100 (solve_workloads.py)  ·  * = mapping caveat, see WORKLOADS.zh.md §2",
             ha="center", fontsize=7.4, color=INK2)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
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
