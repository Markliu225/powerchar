"""Economics layer on top of solve.py — payback of the EXTRA GPUs a power cap lets you deploy.

THE TRADE.  Same rack power budget, two fleets:
    TDP : N_tdp GPUs @250 W      (fewer GPUs, each at full nameplate power)
    OPT : N_opt GPUs @~145 W     (more GPUs, each at its efficiency sweet spot)
OPT buys (N_opt - N_tdp) EXTRA GPUs but serves more tokens. Crucially, BOTH fleets draw the SAME
power, so the extra tokens cost NO extra energy -> the only extra cost is the extra GPUs' CapEx and
the extra-token revenue is (nearly) pure upside.

        payback = (extra GPUs x GPU price) / (extra tokens/s x token price)

Prefill (input) and decode (output) tokens are billed at different prices. Energy is identical for
both fleets, so it cancels in this marginal payback (printed only as context). All $ assumptions are
knobs below; payback scales ~linearly with GPU price / token price / 1-over-utilisation.
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import solve as S

# ---- economic knobs --------------------------------------------------------------------------
GPU_USD = 2500.0                 # one Tesla V100-32GB (used market ~$1.5-3k; new was ~$8-10k)
PRICE_IN_PER_MTOK = 0.05         # $ per 1e6 prefill (input) tokens
PRICE_OUT_PER_MTOK = 0.20        # $ per 1e6 decode  (output) tokens
UTIL = 1.0                       # fraction of peak throughput actually sold (payback ∝ 1/UTIL)
ELEC_USD_PER_KWH = 0.12          # context only — equal for both fleets, cancels out
# ----------------------------------------------------------------------------------------------

SEC_PER_DAY = 86400.0
_pin = PRICE_IN_PER_MTOK / 1e6
_pout = PRICE_OUT_PER_MTOK / 1e6


def revenue_per_s(tot_tok_s, P, D):
    """Split balanced throughput into prefill(input)/decode(output) tokens, price each, x utilisation."""
    fp, fd = P / (P + D), D / (P + D)
    return UTIL * tot_tok_s * (fp * _pin + fd * _pout)


REP_RATIO = (1, 10)            # representative workload mix for the absolute-curve comparison


def plot(rows):
    """LEFT: absolute cumulative net-profit curves — POWER CAP vs NO CAP (TDP) for one workload mix.
    Both start negative (full fleet CapEx), climb at their token-revenue rate; capping starts lower
    (more GPUs) but is steeper (more tokens) -> overtakes. RIGHT: 1-year net profit at every mix."""
    here = os.path.dirname(os.path.abspath(__file__))
    months = np.linspace(0, 24, 400); days = months * 30.44
    r = next(x for x in rows if (x["P"], x["D"]) == REP_RATIO)
    GREEN, RED = "#2ca02c", "#d62728"
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.35, 1]})

    # (a) ABSOLUTE net profit: capping vs no-capping (representative ratio)
    a = ax[0]
    net_opt = (-r["opt_capex_usd"] + r["opt_net_usd_day"] * days) / 1e3
    net_tdp = (-r["tdp_capex_usd"] + r["tdp_net_usd_day"] * days) / 1e3
    a.plot(months, net_opt, color=GREEN, lw=2.3, label=f"POWER CAP — {r['opt_gpus']} GPUs @~145 W")
    a.plot(months, net_tdp, color=RED, lw=2.3, label=f"NO CAP (TDP) — {r['tdp_gpus']} GPUs @250 W")
    a.axhline(0, color="k", ls="--", lw=1.1)
    be_opt = r["opt_capex_usd"] / r["opt_net_usd_day"] / 30.44       # each fleet's own break-even (mo)
    be_tdp = r["tdp_capex_usd"] / r["tdp_net_usd_day"] / 30.44
    cross = (r["opt_capex_usd"] - r["tdp_capex_usd"]) / (r["opt_net_usd_day"] - r["tdp_net_usd_day"]) / 30.44
    yc = (-r["opt_capex_usd"] + r["opt_net_usd_day"] * cross * 30.44) / 1e3
    a.plot([be_tdp, be_opt], [0, 0], "o", color="k", ms=5, zorder=6)
    a.plot(cross, yc, "*", color="#cc6600", ms=16, zorder=7)
    a.annotate(f"capping overtakes no-cap  ~{cross:.1f} mo\n(extra \\${r['extra_capex_usd']/1e3:.0f}k repaid)",
               (cross, yc), xytext=(cross + 1.7, yc - 36), fontsize=9.5, color="#cc6600", weight="bold",
               arrowprops=dict(arrowstyle="->", color="#cc6600"))
    a.text(0.3, -r["opt_capex_usd"] / 1e3 + 4,
           f"day-0 CapEx:  cap -\\${r['opt_capex_usd']/1e3:.0f}k   vs   no-cap -\\${r['tdp_capex_usd']/1e3:.0f}k\n"
           f"own break-even:  cap {be_opt:.1f} mo  ·  no-cap {be_tdp:.1f} mo", fontsize=8.5, color="dimgray")
    a.set_xlabel("months of operation"); a.set_ylabel("cumulative net profit (k\\$) = token revenue - energy - CapEx")
    a.set_title(f"Cumulative net profit — power-cap vs NO power-cap  (P:D={REP_RATIO[0]}:{REP_RATIO[1]}, {S.W_RACK/1e3:.0f} kW rack)\n"
                f"capping invests more up front but earns faster -> ahead from ~{cross:.0f} months on")
    a.legend(loc="lower right", fontsize=9.5); a.grid(alpha=.3); a.set_xlim(0, 24)

    # (b) 1-year net profit, capping vs no-cap, every mix
    a = ax[1]
    x = np.arange(len(rows)); w = 0.4
    p_opt = [(-rr["opt_capex_usd"] + rr["opt_net_usd_day"] * 365) / 1e3 for rr in rows]
    p_tdp = [(-rr["tdp_capex_usd"] + rr["tdp_net_usd_day"] * 365) / 1e3 for rr in rows]
    a.bar(x - w / 2, p_opt, w, color=GREEN, label="power cap")
    a.bar(x + w / 2, p_tdp, w, color=RED, label="no cap (TDP)")
    a.set_xticks(x); a.set_xticklabels([f"{rr['P']}:{rr['D']}" for rr in rows], rotation=45, fontsize=8)
    a.set_xlabel("prefill : decode token ratio"); a.set_ylabel("net profit after 1 year (k\\$)")
    a.set_title("1-year net profit — capping wins at every mix")
    a.legend(fontsize=9); a.grid(alpha=.3, axis="y")

    fig.suptitle("Power-capping economics: capping vs no-capping (same rack power budget)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(here, "fig_payback.png"), dpi=130, bbox_inches="tight")
    print(f"wrote fig_payback.png  (P:D={REP_RATIO[0]}:{REP_RATIO[1]}: cap break-even {be_opt:.1f}mo, "
          f"no-cap {be_tdp:.1f}mo, capping overtakes at {cross:.1f}mo)")


def main():
    pol = S.policies()
    energy_day = S.W_RACK / 1000.0 * 24.0 * ELEC_USD_PER_KWH          # same for both fleets
    print(f"knobs:  GPU=${GPU_USD:,.0f}   in=${PRICE_IN_PER_MTOK}/Mtok  out=${PRICE_OUT_PER_MTOK}/Mtok   "
          f"util={UTIL:.0%}   elec=${ELEC_USD_PER_KWH}/kWh")
    print(f"both fleets draw {S.W_RACK/1000:.0f} kW -> energy ${energy_day:,.1f}/day each (identical -> cancels)\n")
    print(f"{'P:D':>7} | {'+GPU':>4} {'extra CapEx':>11} | {'OPT k/s':>8} {'TDP k/s':>8} {'Δtok/s':>8} | "
          f"{'Δrev/day':>9} {'Δrev/yr':>9} | {'payback':>8}")
    rows = []
    for (P, D) in S.RATIOS:
        o = S.solve(P, D, *pol["OPTIMAL"]); t = S.solve(P, D, *pol["TDP"])
        n_opt, n_tdp = o["Np"] + o["Nd"], t["Np"] + t["Nd"]
        n_extra = n_opt - n_tdp
        capex = n_extra * GPU_USD
        dT = o["tot"] - t["tot"]
        drev_day = revenue_per_s(dT, P, D) * SEC_PER_DAY
        payback = capex / drev_day if drev_day > 0 else float("inf")
        # absolute per-fleet net cash rate = token revenue - own energy bill (actual draw)
        opt_net_day = revenue_per_s(o["tot"], P, D) * SEC_PER_DAY - (o["Wp"] + o["Wd"]) / 1e3 * 24 * ELEC_USD_PER_KWH
        tdp_net_day = revenue_per_s(t["tot"], P, D) * SEC_PER_DAY - (t["Wp"] + t["Wd"]) / 1e3 * 24 * ELEC_USD_PER_KWH
        print(f"{P}:{D:<5} | {n_extra:>4d} ${capex:>10,.0f} | {o['tot']/1e3:>8.1f} {t['tot']/1e3:>8.1f} {dT:>8.0f} | "
              f"${drev_day:>8,.0f} ${drev_day*365/1e3:>7,.0f}k | {payback:>5.0f} d")
        rows.append({"P": P, "D": D, "opt_gpus": n_opt, "tdp_gpus": n_tdp, "extra_gpus": n_extra,
                     "opt_capex_usd": round(n_opt * GPU_USD), "tdp_capex_usd": round(n_tdp * GPU_USD),
                     "extra_capex_usd": round(capex), "opt_tok_s": round(o["tot"]), "tdp_tok_s": round(t["tot"]),
                     "extra_tok_s": round(dT), "opt_net_usd_day": round(opt_net_day, 1),
                     "tdp_net_usd_day": round(tdp_net_day, 1), "extra_rev_usd_day": round(drev_day, 1),
                     "extra_rev_usd_year": round(drev_day * 365), "payback_days": round(payback, 1)})

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "economics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    print("\nwrote economics.csv  (payback = cost of the extra GPUs / revenue of the extra tokens;"
          " same power both ways, so energy cancels)")
    plot(rows)


if __name__ == "__main__":
    main()
