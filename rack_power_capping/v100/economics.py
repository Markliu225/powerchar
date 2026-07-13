"""Economics layer on top of solve_workloads.py — payback of the EXTRA GPUs, per workload class.

THE TRADE.  Same rack power budget, two fleets for each workload class:
    TDP : every GPU @250 W          (fewer GPUs at full nameplate power)
    OPT : slot-aware capped fleet   (more GPUs, caps float; from solve_workloads.solve_opt)
OPT buys extra GPUs but serves more tokens at the SAME power draw, so the extra tokens cost no
extra energy — the only extra cost is the extra GPUs' CapEx:

        payback = (extra GPUs x GPU price) / (extra tokens/s x token price)

Prefill (input) and decode (output) tokens are billed at different prices; each class's revenue
split follows its canonical request shape. The punchline is PER-CLASS: high-throughput classes
(classification, RAG, chat) repay the extra CapEx in months, while decode-starved classes
(long CoT reasoning, 32k summarization) may never repay it at these token prices — the same
capping recipe, wildly different economics.
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import solve_workloads as SW

# ---- economic knobs --------------------------------------------------------------------------
GPU_USD = 2500.0                 # one Tesla V100-32GB (used market ~$1.5-3k; new was ~$8-10k)
PRICE_IN_PER_MTOK = 0.05         # $ per 1e6 prefill (input) tokens
PRICE_OUT_PER_MTOK = 0.20        # $ per 1e6 decode  (output) tokens
UTIL = 1.0                       # fraction of peak throughput actually sold (payback ∝ 1/UTIL)
ELEC_USD_PER_KWH = 0.12          # context only — same budget both fleets, cancels in the margin
# ----------------------------------------------------------------------------------------------

SEC_PER_DAY = 86400.0
_pin = PRICE_IN_PER_MTOK / 1e6
_pout = PRICE_OUT_PER_MTOK / 1e6
REP_ID = "chat-phi3"             # representative class member for the absolute-curve panel
GREEN, RED = "#2ca02c", "#d62728"
HERE = os.path.dirname(os.path.abspath(__file__))


def revenue_per_s(tot_tok_s, fp, fd):
    """Split balanced throughput into prefill/decode tokens by the class shape, price each."""
    return UTIL * tot_tok_s * (fp * _pin + fd * _pout)


def plot(rows):
    """LEFT: cumulative net profit, cap vs no-cap, for the representative chat rack.
    RIGHT: payback of the extra GPUs per workload (log y — spans months to decades)."""
    r = next(x for x in rows if x["id"] == REP_ID)
    months = np.linspace(0, 24, 400); days = months * 30.44
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.2, 1]})

    a = ax[0]
    net_opt = (-r["opt_capex_usd"] + r["opt_net_usd_day"] * days) / 1e3
    net_tdp = (-r["tdp_capex_usd"] + r["tdp_net_usd_day"] * days) / 1e3
    a.plot(months, net_opt, color=GREEN, lw=2.3, label=f"POWER CAP — {r['opt_gpus']} GPUs (slot-aware caps)")
    a.plot(months, net_tdp, color=RED, lw=2.3, label=f"NO CAP (TDP) — {r['tdp_gpus']} GPUs @250 W")
    a.axhline(0, color="k", ls="--", lw=1.1)
    cross = (r["opt_capex_usd"] - r["tdp_capex_usd"]) / (r["opt_net_usd_day"] - r["tdp_net_usd_day"]) / 30.44
    yc = (-r["opt_capex_usd"] + r["opt_net_usd_day"] * cross * 30.44) / 1e3
    a.plot(cross, yc, "*", color="#cc6600", ms=16, zorder=7)
    a.annotate(f"capping overtakes no-cap ~{cross:.1f} mo\n(extra \\${r['extra_capex_usd']/1e3:.0f}k repaid)",
               (cross, yc), xytext=(cross + 2.2, yc - 60), fontsize=9.5, color="#cc6600", weight="bold",
               arrowprops=dict(arrowstyle="->", color="#cc6600"))
    a.set_xlabel("months of operation")
    a.set_ylabel("cumulative net profit (k\\$) = token revenue - energy - CapEx")
    a.set_title(f"Cumulative net profit — cap vs no-cap   ({r['class_en']} rack: {REP_ID}, "
                f"{SW.W_RACK/1e3:.0f} kW / {SW.N_GPU_MAX} slots)")
    a.legend(loc="lower right", fontsize=9.5); a.grid(alpha=.3); a.set_xlim(0, 24)

    a = ax[1]
    x = np.arange(len(rows))
    pb = [r_["payback_days"] for r_ in rows]
    a.bar(x, pb, 0.55, color=GREEN)
    a.axhline(365, color="k", ls="--", lw=1.1)
    a.text(-0.42, 400, "1 year", ha="left", fontsize=9)
    for i, r_ in enumerate(rows):
        a.text(i, pb[i] * 1.10, f"{pb[i]:.0f}d", ha="center", fontsize=8)
    a.set_yscale("log")
    a.set_xticks(x)
    a.set_xticklabels([f"{r_['id']}\n{r_['class_short']}" for r_ in rows], fontsize=6.8, rotation=25)
    a.set_ylabel("payback of the extra GPUs (days, log)")
    a.set_title("Payback per workload class — months for token-rich classes,\n"
                "decades for decode-starved ones (long CoT, 32k summarization)")
    a.grid(alpha=.3, axis="y", which="both")

    fig.suptitle("Power-capping economics by workload class (same rack power budget both fleets)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_payback.png"), dpi=130, bbox_inches="tight")
    print(f"wrote fig_payback.png  ({REP_ID}: capping overtakes no-cap at {cross:.1f} mo)")


def main():
    energy_day = SW.W_RACK / 1000.0 * 24.0 * ELEC_USD_PER_KWH
    print(f"knobs:  GPU=${GPU_USD:,.0f}   in=${PRICE_IN_PER_MTOK}/Mtok  out=${PRICE_OUT_PER_MTOK}/Mtok   "
          f"util={UTIL:.0%}   elec=${ELEC_USD_PER_KWH}/kWh")
    print(f"both fleets budgeted {SW.W_RACK/1000:.0f} kW -> energy <= ${energy_day:,.1f}/day each\n")
    print(f"{'class':<22}{'workload':<18} | {'+GPU':>4} {'extra CapEx':>11} | {'Δtok/s':>8} | "
          f"{'Δrev/day':>9} | {'payback':>9}")
    rows = []
    by_id = {w["id"]: w for w in SW.PORTFOLIO}
    for wid in SW.ORDERED_IDS:
        c = SW.load_workload(by_id[wid])
        o = SW.solve_opt(c)
        t = SW.solve_tdp(c)
        fp, fd = c["Lp"] / (c["Lp"] + c["Ld"]), c["Ld"] / (c["Lp"] + c["Ld"])
        n_opt, n_tdp = o["Np"] + o["Nd"], t["Np"] + t["Nd"]
        capex = (n_opt - n_tdp) * GPU_USD
        dT = o["tot"] - t["tot"]
        drev_day = revenue_per_s(dT, fp, fd) * SEC_PER_DAY
        payback = capex / drev_day if drev_day > 0 else float("inf")
        opt_net_day = revenue_per_s(o["tot"], fp, fd) * SEC_PER_DAY - o["w_used"] / 1e3 * 24 * ELEC_USD_PER_KWH
        tdp_net_day = revenue_per_s(t["tot"], fp, fd) * SEC_PER_DAY - t["w_used"] / 1e3 * 24 * ELEC_USD_PER_KWH
        print(f"{c['cls']['en']:<22}{wid:<18} | {n_opt - n_tdp:>4d} ${capex:>10,.0f} | {dT:>8.0f} | "
              f"${drev_day:>8,.2f} | {payback:>6,.0f} d")
        rows.append({"id": wid, "class": c["cls"]["key"], "class_en": c["cls"]["en"],
                     "class_short": c["cls"]["short"],
                     "opt_gpus": n_opt, "tdp_gpus": n_tdp, "extra_gpus": n_opt - n_tdp,
                     "opt_capex_usd": round(n_opt * GPU_USD), "tdp_capex_usd": round(n_tdp * GPU_USD),
                     "extra_capex_usd": round(capex), "opt_tok_s": round(o["tot"]),
                     "tdp_tok_s": round(t["tot"]), "extra_tok_s": round(dT),
                     "opt_net_usd_day": round(opt_net_day, 1), "tdp_net_usd_day": round(tdp_net_day, 1),
                     "extra_rev_usd_day": round(drev_day, 2), "payback_days": round(payback, 1)})

    with open(os.path.join(HERE, "economics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    print("\nwrote economics.csv  (payback = extra-GPU CapEx / extra-token revenue; energy cancels)")
    plot(rows)


if __name__ == "__main__":
    main()
