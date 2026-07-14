"""Composite-workload power-capping economics — the NONLINEAR return curve, V100 & H200.

The repo's per-class economics (rack_power_capping/v100/economics.py) fixes the rack power
budget so ENERGY CANCELS between the capped and uncapped fleets — its payback is linear and
electricity-independent. This asks the opposite, real question: for the COMPOSITE workload (all
10 use-case classes, dataset-size-weighted as a proxy for real proportions — dominated by the
Chat+Code production traces), what does profit look like as you choose the operating power cap,
once electricity is a real cost?

THE MECHANISM (why the curve is nonlinear).  Serve a fixed demand mix. At per-GPU cap p, each
class's mapped workload delivers T_pre(p), T_dec(p) tok/s (the fitted curves). To serve one
composite request you rent
        g(p) = Σ_i φ_i ( Lp_i / T_pre_i(p) + Ld_i / T_dec_i(p) )       [GPU-seconds]
GPU-seconds (φ_i = real request share; Lp_i, Ld_i = measured prefill/decode tokens per request).
Cost per request has two terms pulling opposite ways in p:
        energy = PUE · elec · Σφ_i(Lp_i·Ppre_i/Tpre_i + Ld_i·Pdec_i/Tdec_i)   billed on the MEASURED
                 draw P(p) (utility charges consumption, not the set cap) — rises with p but the
                 GPU-seconds fall -> U-shaped, min near the efficiency sweet spot
        capex  = (GPU$/life) · g(p)             GPU-time only, falls monotonically with p (more tok/GPU)

NOT MODELLED — latency/SLO.  Revenue is held cap-independent, i.e. capping is assumed free of
service-quality cost. But ~92% of this mix is interactive (Chat 68% + Code 24%), whose TTFT /
inter-token latency degrades as caps drop. So the profit-optimal cap here is an UPPER BOUND on how
far to cap interactive traffic, not a target (the rack doc punts on the same, WORKLOADS.zh.md §7).
Revenue is fixed by the demand (tokens sold), so
        profit/Mtok(p) = revenue/Mtok - 1e6·(PUE·elec·p/3.6e6 + GPU$/life)·g(p) / tok
is NONLINEAR with an interior optimum whose location moves with the elec:capex balance — cheap
power -> optimum at max throughput (TDP-like); dear power -> optimum at the efficiency sweet spot.

TWO CAP POLICIES.  UNIFORM = one per-GPU cap for both phases (fig 1's single-knob curve).
DISAGGREGATED = each phase capped on its OWN curve (what the rack recipes actually do): each
class-phase independently minimizes (capex + energy)/T(cap). Disaggregation matters most on H200,
where decode saturates far below TDP, so its decode GPUs can be capped down for free.

python3 plot_composite_economics.py -> fig_composite_economics.png          (profit vs cap)
                                       fig_composite_elec_sensitivity.png   (profit/benefit vs elec)
                                       composite_economics.csv
"""
from __future__ import annotations
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PORT = os.path.join(ROOT, "pt_cap_gpu1", "portfolio")
sys.path.insert(0, PORT)
sys.path.insert(0, HERE)
import fitlib                                            # noqa: E402
import plot_power_curves as V                            # noqa: E402  NAME/MAP (pure data)

# ---- economic knobs (documented, all parameterizable) ----------------------------------------
PRICE_IN_PER_MTOK = 0.05          # $ per 1e6 prefill (input) tokens   (repo economics.py)
PRICE_OUT_PER_MTOK = 0.20         # $ per 1e6 decode  (output) tokens
PUE = 1.3                         # facility power / IT power (cooling etc.) — part of "电价等等"
GPU_LIFETIME_YR = 3.0             # straight-line amortization horizon at 24/7
UTIL = 1.0                        # fraction of peak throughput actually sold (capex/tok ∝ 1/UTIL)
GPU_USD = {"V100": 2500.0, "H200": 30000.0}   # market/planning prices (V100 per repo; H200 ~$25-40k)
ELEC_CURVES = [0.05, 0.20, 0.50, 1.00]        # $/kWh shown as separate curves in fig 1
ELEC_REP = 0.50                   # $/kWh for the cost-decomposition panels
ELEC_MAX = 1.50                   # $/kWh axis top for the sensitivity figure (incl. demand/carbon)
# ----------------------------------------------------------------------------------------------

DEVICES = {"V100": dict(data=os.path.join(PORT, "data"), cap=(100.0, 250.0), tdp=250.0),
           "H200": dict(data=os.path.join(ROOT, "data_h200"), cap=(200.0, 700.0), tdp=700.0)}
SEC_PER_YR = 365.0 * 86400.0
JOULE_PER_KWH = 3.6e6
_pin, _pout = PRICE_IN_PER_MTOK / 1e6, PRICE_OUT_PER_MTOK / 1e6
NAME, MAP = V.NAME, V.MAP
INK, INK2, MUTE, GRID = V.INK, V.INK2, V.MUTE, V.GRID
TDP_C, UNI_C, DIS_C, CAP_C, EN_C = "#d62728", "#1f77b4", "#2ca02c", "#2ca02c", "#d62728"


def _read(path):
    rows = [r for r in csv.DictReader(open(path)) if float(r["throughput_tok_s"]) > 0]
    cap = np.array([float(r["cap_w"]) for r in rows])
    pwr = np.array([float(r["power_avg_w"]) for r in rows])          # MEASURED avg draw
    P = cap if np.ptp(cap) > 1e-6 else pwr                           # power axis: cap if swept, else measured
    return (P, np.array([float(r["throughput_tok_s"]) for r in rows]),
            np.array([float(r["sm_clk_avg"]) for r in rows]), pwr)


def _batch(data_dir, wid):
    return float(next(csv.DictReader(open(os.path.join(data_dir, f"{wid}_decode.csv"))))["batch"])


def phase_arrays(data_dir, wid, f_max, cap_lo, cap_hi, grid):
    """Fitted T_pre/T_dec AND MEASURED power P_pre/P_dec on the grid (arrays now, no closures).
    Energy is billed on the MEASURED draw (what the utility charges), not the set cap: on H200 the
    low-cap set point under-enforces (draw > cap up to ~1.9x), on V100 memory-bound decode draws
    below cap — using the set cap would mis-bill energy and one-sidedly bias the capping benefit."""
    cp, Tp, Fp, Wp = _read(os.path.join(data_dir, f"{wid}_prefill.csv"))
    cd, Td, Fd, Wd = _read(os.path.join(data_dir, f"{wid}_decode.csv"))
    preT, _ = fitlib.fit_prefill_unified(cp, Tp, Fp, f_max)
    decT, _ = fitlib.fit_decode_additive(cd, Td, Fd, _batch(data_dir, wid), f_max)
    q = np.clip(grid, cap_lo, cap_hi)
    op, od = np.argsort(cp), np.argsort(cd)                          # interp needs sorted xp
    Ppre = np.interp(q, cp[op], Wp[op])                              # measured avg power vs cap
    Pdec = np.interp(q, cd[od], Wd[od])
    return np.maximum(preT(q), 1e-9), np.maximum(decT(q), 1e-9), Ppre, Pdec


def load_mix():
    """Composite: dataset-size share φ_i (n, a proxy for real proportions) and tokens/request Lp_i, Ld_i."""
    rows = list(csv.DictReader(open(os.path.join(HERE, "workload_ratios.csv"))))
    ntot = sum(float(r["n"]) for r in rows)
    return [dict(klass=r["klass"], phi=float(r["n"]) / ntot,
                 Lp=float(r["pre_mean"]), Ld=float(r["dec_mean"])) for r in rows]


def build(dev, mix, grid):
    """Per-class fitted phase throughputs + measured power on the grid, plus composite constants.
    Classes whose mapped workload is absent for this device (e.g. Extract/classify-qwen7b on the
    v-next H200 data) are skipped and φ renormalized over the classes actually present."""
    d = DEVICES[dev]
    f_max = fitlib.resolve_f_max(d["data"])
    avail = [m for m in mix if os.path.exists(os.path.join(d["data"], f"{MAP[m['klass']]}_prefill.csv"))]
    zphi = sum(m["phi"] for m in avail)                              # renormalize the retained mix
    cls = []
    tok = rev = 0.0
    dropped = [NAME[m["klass"]] for m in mix if m not in avail]
    for m in avail:
        phi = m["phi"] / zphi
        Tpre, Tdec, Ppre, Pdec = phase_arrays(d["data"], MAP[m["klass"]], f_max,
                                              d["cap"][0], d["cap"][1], grid)
        cls.append(dict(phi=phi, Lp=m["Lp"], Ld=m["Ld"], Tpre=Tpre, Tdec=Tdec,
                        Ppre=Ppre, Pdec=Pdec))
        tok += phi * (m["Lp"] + m["Ld"])
        rev += phi * (m["Lp"] * _pin + m["Ld"] * _pout)
    if dropped:
        print(f"  [{dev}] dropped (no data): {', '.join(dropped)}  -> mix renormalized over {len(avail)} classes")
    g = sum(c["phi"] * (c["Lp"] / c["Tpre"] + c["Ld"] / c["Tdec"]) for c in cls)    # GPU-s/req (capex)
    ej = sum(c["phi"] * (c["Lp"] / c["Tpre"] * c["Ppre"]                            # measured J/req (PUE=1)
                         + c["Ld"] / c["Tdec"] * c["Pdec"]) for c in cls)
    return dict(cls=cls, tok=tok, rev=rev, rev_mtok=1e6 * rev / tok, g=g, ej=ej,
                capex_rate=GPU_USD[dev] / (GPU_LIFETIME_YR * SEC_PER_YR) / UTIL)


def profit_uniform(b, grid, elec):
    """$/Mtok profit vs uniform cap at electricity `elec`; also capex & (measured) energy split."""
    capex = 1e6 * b["capex_rate"] * b["g"] / b["tok"]               # $/Mtok (GPU-time only), array
    energy = 1e6 * PUE * elec / JOULE_PER_KWH * b["ej"] / b["tok"]  # $/Mtok on MEASURED draw, array
    return b["rev_mtok"] - capex - energy, capex, energy


def profit_disagg(b, grid, elec):
    """$/Mtok profit with each class-phase capped independently at its cost-min cap (measured energy)."""
    k = PUE * elec / JOULE_PER_KWH
    cost = 0.0
    for c in b["cls"]:
        pre = (b["capex_rate"] + k * c["Ppre"]) / c["Tpre"]        # $/prefill-token vs cap
        dec = (b["capex_rate"] + k * c["Pdec"]) / c["Tdec"]        # $/decode-token  vs cap
        cost += c["phi"] * (c["Lp"] * pre.min() + c["Ld"] * dec.min())
    return b["rev_mtok"] - 1e6 * cost / b["tok"]


def main():
    mix = load_mix()
    grid = {dev: np.linspace(*DEVICES[dev]["cap"], 1501) for dev in DEVICES}
    B = {dev: build(dev, mix, grid[dev]) for dev in DEVICES}

    # ============ figure 1: profit vs uniform cap (the nonlinear operating curve) ============
    fig, ax = plt.subplots(2, 2, figsize=(15, 10.4), gridspec_kw={"width_ratios": [1.3, 1]})
    for row, dev in enumerate(DEVICES):
        b = B[dev]; gd = grid[dev]; lo, hi = DEVICES[dev]["cap"]; tdp = DEVICES[dev]["tdp"]
        i_tdp = int(np.argmin(np.abs(gd - tdp)))

        a = ax[row][0]                          # profit/Mtok vs cap, one curve per elec price
        a.axhline(b["rev_mtok"], color=MUTE, lw=1, ls=":")
        a.text(lo, b["rev_mtok"], f"revenue {b['rev_mtok']:.3f} ", va="bottom", ha="left",
               fontsize=7.5, color=MUTE)
        cmap = plt.cm.viridis(np.linspace(0.1, 0.8, len(ELEC_CURVES)))
        opt_profits = []
        for elec, col in zip(ELEC_CURVES, cmap):
            pr, _, _ = profit_uniform(b, gd, elec)
            a.plot(gd, pr, color=col, lw=2.1, zorder=3)
            i = int(np.argmax(pr)); opt_profits.append(pr[i])
            a.plot(gd[i], pr[i], "o", ms=7, color=col, mec="white", mew=1, zorder=5)
            a.annotate(f"${elec:.2f}/kWh · opt {gd[i]:.0f}W", (hi, pr[i_tdp]),   # right-edge label
                       textcoords="offset points", xytext=(9, 0), va="center", ha="left",
                       fontsize=7.6, color=col, annotation_clip=False)
        a.axvline(tdp, color=TDP_C, lw=1.2, ls="--", alpha=.7)
        ymax = b["rev_mtok"] * 1.04
        ymin = min(opt_profits) - 0.55 * (ymax - min(opt_profits))
        a.annotate("TDP\n(no cap)", (tdp, ymax), textcoords="offset points", xytext=(-4, -2),
                   va="top", ha="right", fontsize=7.5, color=TDP_C)
        a.set_ylim(ymin, ymax); a.set_xlim(lo - (hi - lo) * .04, hi + (hi - lo) * .30)
        note = "optimum slides down as power gets dearer" if dev == "V100" else \
               "optimum pinned at TDP: CapEx dominates + uniform cap can't spare decode"
        a.set_title(f"{dev}  ·  GPU ${GPU_USD[dev]:,.0f}, {GPU_LIFETIME_YR:.0f}-yr amort  —  {note}",
                    fontsize=10, color=INK)
        a.set_xlabel("uniform per-GPU power cap (W)", fontsize=9, color=INK2)
        a.set_ylabel("profit ($ / 1e6 tokens sold)", fontsize=9, color=INK2)
        a.grid(alpha=.4, color=GRID, lw=.7); a.tick_params(labelsize=8, colors=MUTE)
        [s.set_visible(False) for s in (a.spines["top"], a.spines["right"])]

        a = ax[row][1]                          # cost decomposition at ELEC_REP
        _, cx, en = profit_uniform(b, gd, ELEC_REP)
        a.stackplot(gd, cx, en, colors=[CAP_C, EN_C], alpha=.45,
                    labels=["CapEx / Mtok (↓ with cap)", f"energy / Mtok (↑ with cap)"])
        a.plot(gd, cx + en, color=INK, lw=2, label="total cost / Mtok")
        a.axhline(b["rev_mtok"], color=MUTE, lw=1, ls=":", label=f"revenue {b['rev_mtok']:.3f}")
        i = int(np.argmin(cx + en))
        a.plot(gd[i], (cx + en)[i], "v", ms=9, color=INK, zorder=6)
        a.annotate(f"min cost {gd[i]:.0f}W", (gd[i], (cx + en)[i]), textcoords="offset points",
                   xytext=(-10, 9), ha="right", fontsize=7.8, color=INK)
        a.set_xlim(lo, hi); a.set_ylim(0, max(b["rev_mtok"], (cx + en)[i_tdp]) * 1.4)
        a.set_title(f"{dev} — why it curves @ ${ELEC_REP}/kWh: CapEx↓ vs energy↑", fontsize=10, color=INK)
        a.set_xlabel("uniform per-GPU power cap (W)", fontsize=9, color=INK2)
        a.set_ylabel("$ / 1e6 tokens", fontsize=9, color=INK2)
        a.legend(fontsize=7.4, loc="upper right", ncols=1, framealpha=.9)
        a.grid(alpha=.4, color=GRID, lw=.7); a.tick_params(labelsize=8, colors=MUTE)
        [s.set_visible(False) for s in (a.spines["top"], a.spines["right"])]

    fig.suptitle("Composite-workload power-capping economics — the nonlinear profit curve   "
                 "(10 classes, dataset-size-weighted ≈ Chat 68% + Code 24% + 8-class tail)\n"
                 "revenue fixed by demand (latency/SLO NOT modelled); cost = amortized CapEx (↓ with cap) "
                 "+ PUE·elec·MEASURED-watts (↑ with cap) → interior optimum that moves with the power price",
                 fontsize=12.5, color=INK)
    fig.text(0.5, 0.006, FOOT1, ha="center", fontsize=7.3, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(os.path.join(HERE, "fig_composite_economics.png"), dpi=130, bbox_inches="tight")
    print("wrote fig_composite_economics.png")

    # ============ figure 2: electricity sensitivity (uniform vs disaggregated vs TDP) ============
    elec_fine = np.linspace(0.02, ELEC_MAX, 300)
    fig2, ax2 = plt.subplots(2, 2, figsize=(15, 10.2))
    rows_out = []
    for row, dev in enumerate(DEVICES):
        b = B[dev]; gd = grid[dev]; tdp = DEVICES[dev]["tdp"]
        i_tdp = int(np.argmin(np.abs(gd - tdp)))
        p_tdp, p_uni, p_dis = [], [], []
        for e in elec_fine:
            pr, _, _ = profit_uniform(b, gd, e)
            p_tdp.append(pr[i_tdp]); p_uni.append(pr.max()); p_dis.append(profit_disagg(b, gd, e))
        p_tdp, p_uni, p_dis = map(np.array, (p_tdp, p_uni, p_dis))

        a = ax2[row][0]                         # profit vs elec: TDP / uniform-opt / disagg-opt
        a.axhline(0, color=MUTE, lw=.8, ls=":")
        a.fill_between(elec_fine, p_tdp, p_dis, color=DIS_C, alpha=.13)
        a.plot(elec_fine, p_dis, color=DIS_C, lw=2.4, label="capped, disaggregated (per-phase optimal)")
        a.plot(elec_fine, p_uni, color=UNI_C, lw=2, ls="-", label="capped, uniform (one cap)")
        a.plot(elec_fine, p_tdp, color=TDP_C, lw=2, label="no cap (TDP)")
        be = np.interp(0.0, p_dis[::-1], elec_fine[::-1]) if p_dis.min() < 0 < p_dis.max() else None
        a.set_title(f"{dev} — profit vs power price   (revenue {b['rev_mtok']:.3f} $/Mtok, fixed)",
                    fontsize=10.5, color=INK)
        a.set_xlabel("electricity price ($/kWh, incl. PUE)", fontsize=9, color=INK2)
        a.set_ylabel("profit ($ / 1e6 tokens)", fontsize=9, color=INK2)
        a.legend(fontsize=8, loc="upper right"); a.set_xlim(0, ELEC_MAX)
        a.grid(alpha=.4, color=GRID, lw=.7); a.tick_params(labelsize=8, colors=MUTE)
        [s.set_visible(False) for s in (a.spines["top"], a.spines["right"])]

        a = ax2[row][1]                         # capping benefit vs elec (nonlinear growth)
        a.plot(elec_fine, p_dis - p_tdp, color=DIS_C, lw=2.4, label="disaggregated benefit")
        a.plot(elec_fine, p_uni - p_tdp, color=UNI_C, lw=2, label="uniform-cap benefit")
        a.fill_between(elec_fine, 0, p_dis - p_tdp, color=DIS_C, alpha=.12)
        a.set_title(f"{dev} — value of capping grows NONLINEARLY with power price", fontsize=10.5, color=INK)
        a.set_xlabel("electricity price ($/kWh, incl. PUE)", fontsize=9, color=INK2)
        a.set_ylabel("capping benefit ($/Mtok vs TDP)", fontsize=9, color=INK2)
        a.legend(fontsize=8, loc="upper left"); a.set_xlim(0, ELEC_MAX); a.set_ylim(bottom=0)
        a.grid(alpha=.4, color=GRID, lw=.7); a.tick_params(labelsize=8, colors=MUTE)
        [s.set_visible(False) for s in (a.spines["top"], a.spines["right"])]

        for e in ELEC_CURVES:
            pr, _, _ = profit_uniform(b, gd, e); i = int(np.argmax(pr))
            rows_out.append(dict(device=dev, elec_usd_kwh=e, revenue_usd_mtok=round(b["rev_mtok"], 4),
                                 tdp_cap_w=round(tdp), profit_tdp_usd_mtok=round(float(pr[i_tdp]), 4),
                                 uniform_opt_cap_w=round(float(gd[i])),
                                 profit_uniform_usd_mtok=round(float(pr.max()), 4),
                                 profit_disagg_usd_mtok=round(float(profit_disagg(b, gd, e)), 4),
                                 uniform_benefit_usd_mtok=round(float(pr.max() - pr[i_tdp]), 4),
                                 disagg_benefit_usd_mtok=round(float(profit_disagg(b, gd, e) - pr[i_tdp]), 4)))

    fig2.suptitle("Composite-workload capping economics vs electricity price — V100 & H200   "
                  "(dataset-size-weighted mix ≈ Chat 68% + Code 24%; latency/SLO NOT modelled)\n"
                  "no-cap profit falls linearly with power price; capping bends it up (convex) — "
                  "disaggregation (cap decode on its own curve) wins most where uniform capping can't",
                  fontsize=12.5, color=INK)
    fig2.text(0.5, 0.006, FOOT2, ha="center", fontsize=7.3, color=INK2)
    fig2.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig2.savefig(os.path.join(HERE, "fig_composite_elec_sensitivity.png"), dpi=130, bbox_inches="tight")
    print("wrote fig_composite_elec_sensitivity.png")

    path = os.path.join(HERE, "composite_economics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader()
        [w.writerow(r) for r in rows_out]
    print(f"wrote {os.path.basename(path)}")


FOOT1 = ("mix weight φ = dataset SIZE (n from workload_ratios.csv), a proxy for real proportions (dominated by the "
         "Chat+Code production traces; the 8 Dolly classes are the small tail); tokens/request = measured pre_mean/dec_mean; "
         "per-phase throughput & MEASURED power from the mapped workload's fits/data (V100 portfolio / data_h200)\n"
         "energy billed on measured draw, not set cap  ·  UNIFORM cap = one knob for both phases (low-cap tail off-scale)  ·  "
         "NOT modelled: latency/SLO — 92% of the mix is interactive, so a lower cap costs service quality not priced here; "
         "token-count vs mapped-curve context scales differ (WORKLOADS.zh.md §2) → biases profit optimistic")
FOOT2 = ("profit/Mtok = revenue - amortized GPU CapEx (GPU-time) - PUE·elec·MEASURED-watts, at the profit-optimal cap  ·  "
         "DISAGGREGATED caps each class-phase on its own curve (what the rack recipes do); on H200 decode saturates below TDP "
         "so it can be capped down  ·  knobs: GPU $2.5k(V100)/$30k(H200), 3-yr amort, PUE 1.3, $0.05/$0.20 per Mtok in/out  ·  "
         "latency/SLO NOT modelled → optimal cap is an UPPER bound on how far to cap interactive traffic  ·  elec axis folds in PUE")

if __name__ == "__main__":
    main()
