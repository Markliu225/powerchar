"""Cumulative profit over TIME — power-capping vs TDP, composite workload, V100 & H200.

THE QUESTION (as asked): serve the COMPOSITE demand (all 10 use-case classes at their real
proportions — dataset-size weighting, dominated by the Chat+Code production traces). Two fleets
draw the SAME rack power; plot each fleet's cumulative net profit vs operating time, per device.

  x = months of operation      y = cumulative net profit ($) = token revenue - energy - CapEx

THE TWO FLEETS (same IT power budget P, both serve the composite mix):
  TDP  : every GPU at nameplate power.        Fewer, hotter GPUs.
  CAP  : every phase at its max-tok/J point.  More GPUs, each throttled to its efficiency sweet
         spot, so the fleet delivers MORE tokens per watt -> more revenue/day at the same power.
Per composite request a fleet spends g(π) GPU-seconds and ej(π) measured Joules (from the fitted
per-phase curves + MEASURED draw); at power P it serves R = P/ej req/s using N = P·g/ej GPUs.
Capping lowers ej -> higher R (more revenue) but higher N (more CapEx). Same power both fleets, so
the ENERGY bill is identical between them and the CAP-vs-TDP crossover is a pure CapEx-vs-revenue
trade: a higher token price shrinks the payback of the extra GPUs.

WHY THE CURVES ARE NONLINEAR — two time-varying factors, both real, both convex:
  (1) TOKEN PRICE decays over time. LLM inference $/token has fallen fast (roughly halving every
      ~1-2 yr). So per-request revenue = rev0 · 2^(-t/half-life): the daily revenue RATE shrinks,
      the cumulative-revenue curve is concave, and profit rises fast early then FLATTENS (and can
      roll over once the decayed price drops below the operating cost). Starting the price HIGH
      gives a fast early payback; the decay bends the curve.
  (2) ELECTRICITY price rises over time (convex; AI-driven grid stress), so the cumulative energy
      bill ∫elec(t)dt is superlinear — a second, smaller downward bend. Energy is identical
      between the two fleets here (same power), so it shifts both curves together.
Neither factor is linear, so the profit-vs-time curve is not a straight line — it bends over.

NOT MODELLED — latency/SLO. Revenue is cap-independent, i.e. capping is assumed free of service
quality; ~92% of this mix is interactive, so the CAP fleet's edge is an UPPER bound (WORKLOADS §7).
All knobs (start price, half-life, elec path, PUE, GPU $) are documented constants below.

  python3 plot_profit_over_time.py  ->  fig_profit_over_time.png  +  profit_over_time.csv
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
import fitlib                                              # noqa: E402
import plot_composite_economics as CE                      # noqa: E402  (phase_arrays, load_mix, knobs)

# ---- economic knobs ---------------------------------------------------------------------------
PRICE_IN0_PER_MTOK = 0.30      # $/1e6 prefill tokens at t=0 (HIGH start -> fast early payback)
PRICE_OUT0_PER_MTOK = 1.20     # $/1e6 decode tokens at t=0
PRICE_HALFLIFE_MO = 18.0       # token price halves every N months (LLM inference price trend)
PUE = CE.PUE                   # facility power / IT power
GPU_USD = CE.GPU_USD           # {"V100": 2500, "H200": 30000}
P_BUDGET = {"V100": 5000.0, "H200": 12000.0}   # IT power budget (W); profit scales linearly with it
HORIZON_MO = {"V100": 42.0, "H200": 42.0}      # x-axis span (months)
ELEC0, ELEC_END, ELEC_CONVEX = 0.10, 0.45, 2.0   # $/kWh at t=0 -> at horizon, convex exponent>1
# -----------------------------------------------------------------------------------------------

DEVICES = CE.DEVICES
NAME, MAP = CE.NAME, CE.MAP
SEC_PER_DAY, DAYS_PER_MO = 86400.0, 30.44
_pin0, _pout0 = PRICE_IN0_PER_MTOK / 1e6, PRICE_OUT0_PER_MTOK / 1e6
GREEN, RED, GOLD, MUTE = "#2ca02c", "#d62728", "#cc6600", "#52514e"

FOOT = ("KEY ASSUMPTION — elastic demand: both fleets draw the same power, so the CAP fleet's edge is SELLING its +40%/+19% extra tokens at the market price.\n"
        "If demand is fixed (unsellable surplus), capping to add GPUs does NOT help and the sign can flip (see plot_composite_economics.py).  "
        "CAP = every phase at its max-tok/J point (steepest daily cash-flow, not the only cap choice).\n"
        "CapEx = GPU sticker only (the CAP fleet's extra servers/racks/NICs are NOT costed → optimistic for CAP); single purchase, no GPU replacement/discounting.  "
        "Latency/SLO NOT modelled (~92% of the mix is interactive) → the CAP edge is an upper bound.")


def price_decay(mo):
    return 0.5 ** (mo / PRICE_HALFLIFE_MO)                 # token-price multiplier vs t=0


def elec_path(mo, horizon):
    return ELEC0 + (ELEC_END - ELEC0) * np.clip(mo / horizon, 0, None) ** ELEC_CONVEX


def _cumtrapz(y, x):
    return np.concatenate([[0.0], np.cumsum((y[1:] + y[:-1]) / 2 * np.diff(x))])


def fleet(dev):
    """Composite per-request (g, ej, rev0) for the TDP and CAP fleets, + chosen mean caps.
    g = GPU-seconds/request; ej = measured IT Joules/request; rev0 = $/request at t=0 prices."""
    d = DEVICES[dev]
    f_max = fitlib.resolve_f_max(d["data"])
    lo, hi, tdp = d["cap"][0], d["cap"][1], d["tdp"]
    grid = np.linspace(lo, hi, 1501)
    i_tdp = int(np.argmin(np.abs(grid - tdp)))
    mix = CE.load_mix()
    avail = [m for m in mix if os.path.exists(os.path.join(d["data"], f"{MAP[m['klass']]}_prefill.csv"))]
    zphi = sum(m["phi"] for m in avail)

    acc = {p: dict(g=0.0, ej=0.0, wcap_pre=0.0, wcap_dec=0.0) for p in ("TDP", "CAP")}
    rev0 = 0.0
    for m in avail:
        phi = m["phi"] / zphi
        Lp, Ld = m["Lp"], m["Ld"]
        Tpre, Tdec, Ppre, Pdec = CE.phase_arrays(d["data"], MAP[m["klass"]], f_max, lo, hi, grid)
        i_pre = int(np.argmin(Ppre / Tpre))            # min J/token -> max tok/J (efficiency sweet spot)
        i_dec = int(np.argmin(Pdec / Tdec))
        for p, ip, idc in (("TDP", i_tdp, i_tdp), ("CAP", i_pre, i_dec)):
            a = acc[p]
            a["g"] += phi * (Lp / Tpre[ip] + Ld / Tdec[idc])
            a["ej"] += phi * (Lp / Tpre[ip] * Ppre[ip] + Ld / Tdec[idc] * Pdec[idc])
            a["wcap_pre"] += phi * grid[ip]
            a["wcap_dec"] += phi * grid[idc]
        rev0 += phi * (Lp * _pin0 + Ld * _pout0)
    for p in acc:
        acc[p].update(rev0=rev0)
    return acc


def _cross(x, y, y0=0.0):
    s = np.sign(y - y0)
    idx = np.where(np.diff(s) != 0)[0]
    if not len(idx):
        return None
    i = idx[0]
    a0, a1 = y[i] - y0, y[i + 1] - y0
    return x[i] + (x[i + 1] - x[i]) * (-a0) / (a1 - a0)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))
    rows = []
    for ax, dev in zip(axes, ("V100", "H200")):
        P = P_BUDGET[dev]
        fl = fleet(dev)
        mo = np.linspace(0, HORIZON_MO[dev], 700)
        days = mo * DAYS_PER_MO
        cum_decay = _cumtrapz(price_decay(mo), days)                       # ∫ price-multiplier dt [days]
        cum_elec = _cumtrapz(elec_path(mo, HORIZON_MO[dev]), days)         # ∫ elec dt [$/kWh·days]
        energy_cum = P / 1000.0 * 24.0 * PUE * cum_elec                    # $ cumulative (same both fleets)

        E = {}
        for pol in ("TDP", "CAP"):
            a = fl[pol]
            R = P / a["ej"]; N = P * a["g"] / a["ej"]
            capex = N * GPU_USD[dev]
            rev_day0 = R * a["rev0"] * SEC_PER_DAY                         # $/day at t=0 prices
            profit = -capex + rev_day0 * cum_decay - energy_cum
            E[pol] = dict(R=R, N=N, capex=capex, rev_day0=rev_day0, profit=profit)

        yk_cap, yk_tdp = E["CAP"]["profit"] / 1e3, E["TDP"]["profit"] / 1e3
        ax.plot(mo, yk_cap, color=GREEN, lw=2.4,
                label=f"POWER CAP — {E['CAP']['N']:.0f} GPUs @ {fl['CAP']['wcap_pre']:.0f}/{fl['CAP']['wcap_dec']:.0f} W (pre/dec)")
        ax.plot(mo, yk_tdp, color=RED, lw=2.4,
                label=f"NO CAP (TDP) — {E['TDP']['N']:.0f} GPUs @ {DEVICES[dev]['tdp']:.0f} W")
        ax.axhline(0, color="k", ls="--", lw=1)

        be_cap, be_tdp = _cross(mo, E["CAP"]["profit"]), _cross(mo, E["TDP"]["profit"])
        for be in (be_cap, be_tdp):
            if be is not None:
                ax.plot(be, 0, "o", color="k", ms=5, zorder=6)
        cross = _cross(mo, E["CAP"]["profit"] - E["TDP"]["profit"])
        if cross is not None:
            yc = float(np.interp(cross, mo, yk_cap))
            ax.plot(cross, yc, "*", color=GOLD, ms=17, zorder=7)
            right = cross > 0.5 * HORIZON_MO[dev]            # star on the right -> label down-left, else up-right
            off = (-46, -30, "right", "top") if right else (44, 34, "left", "bottom")
            ax.annotate(f"capping overtakes TDP ~{cross:.1f} mo\n"
                        f"(+{E['CAP']['N']-E['TDP']['N']:.0f} GPUs, +\\${(E['CAP']['capex']-E['TDP']['capex'])/1e3:.0f}k)",
                        (cross, yc), textcoords="offset points", xytext=off[:2], ha=off[2], va=off[3],
                        fontsize=8.6, color=GOLD, weight="bold", arrowprops=dict(arrowstyle="->", color=GOLD))

        pk = int(np.argmax(E["CAP"]["profit"]))          # note the roll-over if the curve peaks in-window
        if 0.03 * len(mo) < pk < 0.97 * len(mo):
            ax.annotate(f"profit peaks ~{mo[pk]:.0f} mo\n(price decayed below cost)", (mo[pk], yk_cap[pk]),
                        xytext=(mo[pk] - HORIZON_MO[dev] * 0.02, yk_cap[pk] + max(yk_cap) * 0.06),
                        ha="right", fontsize=7.8, color=GREEN)

        ax.text(0.015, 0.975,
                f"same {P/1e3:.0f} kW IT budget · both fleets · +{100*(E['CAP']['R']/E['TDP']['R']-1):.0f}% tokens/day\n"
                f"token price \\${PRICE_IN0_PER_MTOK}/\\${PRICE_OUT0_PER_MTOK} per Mtok at t=0, "
                f"halving every {PRICE_HALFLIFE_MO:.0f} mo\n"
                f"elec \\${ELEC0:.2f}→\\${ELEC_END:.2f}/kWh (convex ×{ELEC_CONVEX:.0f}) × PUE {PUE}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.9, color=MUTE,
                bbox=dict(boxstyle="round", fc="#fbfbf7", ec="#ccc", alpha=.92))

        ax.set_title(f"{dev}   (GPU \\${GPU_USD[dev]:,.0f})", fontsize=12, weight="bold")
        ax.set_xlabel("months of operation", fontsize=10)
        ax.set_ylabel("cumulative net profit (k\\$)", fontsize=10)
        ax.set_xlim(0, HORIZON_MO[dev]); ax.grid(alpha=.3)
        ax.legend(loc="lower right", fontsize=8.4)

        rows.append(dict(device=dev, budget_kw=P / 1e3, cap_gpus=round(E["CAP"]["N"]), tdp_gpus=round(E["TDP"]["N"]),
                         cap_capex_usd=round(E["CAP"]["capex"]), tdp_capex_usd=round(E["TDP"]["capex"]),
                         rev_day0_cap_usd=round(E["CAP"]["rev_day0"], 1), rev_day0_tdp_usd=round(E["TDP"]["rev_day0"], 1),
                         tokens_per_day_rel=round(E["CAP"]["R"] / E["TDP"]["R"], 3),
                         cap_breakeven_mo=round(be_cap, 2) if be_cap else None,
                         tdp_breakeven_mo=round(be_tdp, 2) if be_tdp else None,
                         cap_overtakes_tdp_mo=round(cross, 2) if cross else None,
                         cap_profit_peak_mo=round(float(mo[pk]), 1)))

    fig.suptitle("Cumulative profit over time — POWER CAP vs NO CAP (TDP), composite workload   "
                 "(10 use-case classes at real proportions ≈ Chat 68% + Code 24%)\n"
                 "curves BEND because token price decays and electricity rises over time (both nonlinear); "
                 "capping serves more tokens/day → steeper, overtakes TDP early",
                 fontsize=12, y=1.03)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.text(0.5, 0.015, FOOT, ha="center", va="bottom", fontsize=7.6, color=MUTE)
    out = os.path.join(HERE, "fig_profit_over_time.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)

    with open(os.path.join(HERE, "profit_over_time.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    print("wrote profit_over_time.csv")
    for r in rows:
        print(f"  {r['device']}: cap {r['cap_gpus']} vs TDP {r['tdp_gpus']} GPUs; +{100*(r['tokens_per_day_rel']-1):.0f}% tok/day; "
              f"crossover {r['cap_overtakes_tdp_mo']} mo; cap payback {r['cap_breakeven_mo']} mo; peak {r['cap_profit_peak_mo']} mo")


if __name__ == "__main__":
    main()
