"""MIXED-workload profit under THE PAPER'S economic model — power capping vs TDP, V100 & H200.

Implements the paper's 经济模型 section, eq. (1)-(15), at the CLUSTER level:
  per rack (eq. 1-7, 9-15):
    K=m*c_g · D=(K-S)/n · E=e*beta*P*8760 · M=mu*K · C=D+E+M · Q=X*T_yr
    pi=(kappa*pi_p+pi_d)/(1+kappa) · pi(t)=pi*e^(-lambda*t)
    Phi(t)=pi*Q*(1-e^(-lambda*t))/lambda - C*t   (accrual; depreciation inside C)
    ROI(n)=Phi(n)/K · T_pb = cash payback (revenue - E - M reaching K; eq. 13 when lambda=0)
    c_tok=C/Q · g=(pi-c_tok)/pi
  mixture (eq. 8 + the rack-partition rule):
    the real workload is a MIX of J P:D classes; class j holds token-volume share w_j
    (composition measured per ServeGen, Xiang et al., NSDI 2026 — here w_j defaults to the
    token-volume share of the measured datasets in workload_ratios.csv, a documented proxy;
    override via W_OVERRIDE when production shares are available).
    The system partitions racks by class (each rack serves ONE class); class j receives racks
    in proportion to its demand: N_j ∝ w_j / X_j, i.e. enough racks that class-j capacity
    covers its share of demand — the produced token mix then equals the demand mix, and the
    model is linear over racks, so cluster profit = sum of rack profits (per-class kappa_j
    pricing per rack == eq. 8, since all classes share one price schedule).

CLUSTER NORMALIZATION: both policies get the SAME IT power (1 MW => 200 V100 racks @5 kW,
~71.4 H200 racks @14 kW) and allocate racks to classes by the rule above with their OWN
per-rack throughputs. Capping fills 32 slots/rack vs TDP's 20, serves the same demand mix,
and sells more total tokens (all capacity sold, paper §3).

INPUTS per class per policy from the rack solver ({v100,h200}/workload_rack_capping.csv):
m=N_prefill+N_decode; X = solver 'tot' = lam*(Lp+Ld), prefill+decode tokens/s under the class
token balance; P = MEASURED rack draw (OPT: opt_w_measured; TDP: tdp_tok_s/tdp_rack_tok_per_j,
both columns rounded -> <=0.5% round-trip error, <=$22/yr in E). kappa_j = the class's measured
aggregate P:D (ratio_agg) — the same shape the solver balanced the rack against.

Y-AXIS of the figure = cumulative ACCRUAL profit Phi(t) of the 1 MW cluster (M$): revenue
to date (price decaying at lambda) minus (depreciation+electricity+maintenance)*t. Depreciation
spreads K/n over the lifetime, so curves start at 0 — capex is NOT a t=0 step; the t=n value
is eq. (11) summed over the demand-allocated racks. ▼ = CASH payback (K recovered from
revenue - E - M; eq. 13 generalized to lambda>0).

NOT MODELLED (paper §3/§5): latency/SLO; demand growth/decline (utilization = 100%).

python3 plot_profit_model.py -> fig_profit_model.png + profit_model.csv
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- model parameters (paper §5; all documented in the figure) --------------------------------
C_G = {"V100": 2500.0, "H200": 27000.0}   # $/GPU deployed (sticker; server/NIC share foldable)
N_YR = 5.0                 # service life n (paper: 4-6 yr, hyperscaler practice)
SALVAGE = 0.0              # S, conservative baseline (paper §2)
ELEC = 0.10                # e, $/kWh
PUE = 1.1                  # beta (hyperscaler range 1.09-1.15; 2025 survey mean 1.54)
MU = 0.04                  # mu, maintenance fraction of K per year (paper: 3-5%)
PI_P_MTOK = 0.30           # pi_p, $/1e6 input tokens  (small-model serving price, t=0)
PI_D_MTOK = 1.20           # pi_d, $/1e6 output tokens (4x input, in the 4-6x vendor range)
LAMBDA = np.log(2.0) * 12.0 / 18.0   # price decay rate /yr  (halving every 18 months)
T_YR_S = 365.0 * 86400.0   # seconds/yr (~3.15e7, paper eq. 6)
CLUSTER_MW = 1.0           # cluster IT power both policies are normalized to
W_OVERRIDE = None          # {klass: token-share} to replace the dataset-derived demand mix
# -----------------------------------------------------------------------------------------------

DEV = {"V100": dict(csv="v100", budget_kw=5.0, tdp_w=250),
       "H200": dict(csv="h200", budget_kw=14.0, tdp_w=700)}
NAME = {"Generation 创作生成": "Generation", "General QA 常识问答": "General QA",
        "Brainstorming 头脑风暴": "Brainstorming", "Open QA 开放问答": "Open QA",
        "Classification 分类": "Classification", "Summarization 摘要": "Summarization",
        "Extract 信息抽取": "Extract", "Chat 多轮对话": "Chat (dialogue)",
        "Closed QA 闭卷问答": "Closed QA", "Code 代码补全": "Code (completion)"}
GREEN, RED, MUTE, GRID = "#2ca02c", "#d62728", "#52514e", "#e1e0d9"
_pi_p, _pi_d = PI_P_MTOK / 1e6, PI_D_MTOK / 1e6
ratio_str = lambda x: f"{x:.1f}:1" if x >= 1 else f"1:{1/x:.0f}"


def econ(m, X, P_w, kappa, c_g):
    """All per-rack model quantities (eq. 1-7, 9-15)."""
    K = m * c_g
    D = (K - SALVAGE) / N_YR
    E = ELEC * PUE * (P_w / 1000.0) * 8760.0
    M = MU * K
    C = D + E + M
    Q = X * T_YR_S
    pi = (kappa * _pi_p + _pi_d) / (1.0 + kappa)
    R0 = pi * Q
    return dict(m=m, X=X, P_w=P_w, kappa=kappa, K=K, D=D, E=E, M=M, C=C, Q=Q, pi=pi,
                R0=R0, rho0=R0 - C, c_tok=C / Q, gross=(pi - C / Q) / pi if pi > 0 else 0.0)


def phi(R0, C, t):
    """Eq. (11): cumulative accrual profit at time t (years)."""
    t = np.asarray(t, float)
    return R0 * (1.0 - np.exp(-LAMBDA * t)) / LAMBDA - C * t


def payback(R0, K, E, M):
    """Cash payback: first t with R0*(1-e^-lt)/l - (E+M)t >= K (eq. 13 generalized)."""
    tg = np.linspace(0, 3 * N_YR, 6000)
    cash = R0 * (1.0 - np.exp(-LAMBDA * tg)) / LAMBDA - (E + M) * tg - K
    ix = np.where(cash >= 0)[0]
    return float(tg[ix[0]]) if len(ix) else None


def load_mix():
    """Demand mix w_j = token-VOLUME share of the measured datasets (n * (pre_mean+dec_mean)).
    A proxy for production composition (measure per ServeGen, NSDI'26); W_OVERRIDE replaces it."""
    if W_OVERRIDE is not None:
        z = sum(W_OVERRIDE.values())
        return {k: v / z for k, v in W_OVERRIDE.items()}
    rows = list(csv.DictReader(open(os.path.join(HERE, "workload_ratios.csv"), encoding="utf-8")))
    vol = {r["klass"]: float(r["n"]) * (float(r["pre_mean"]) + float(r["dec_mean"])) for r in rows}
    z = sum(vol.values())
    return {k: v / z for k, v in vol.items()}


def load(dev):
    """Per-class per-policy rack economics from the solver CSV."""
    rows = list(csv.DictReader(open(os.path.join(HERE, DEV[dev]["csv"], "workload_rack_capping.csv"),
                                    encoding="utf-8")))
    out = []
    for r in rows:
        kappa = float(r["ratio_agg"])
        m_opt = int(r["opt_N_prefill"]) + int(r["opt_N_decode"])
        m_tdp = int(r["tdp_N_prefill"]) + int(r["tdp_N_decode"])
        p_tdp = float(r["tdp_tok_s"]) / float(r["tdp_rack_tok_per_j"])   # measured draw (W)
        out.append(dict(
            klass=r["klass"], kappa=kappa,
            CAP=econ(m_opt, float(r["opt_tok_s"]), float(r["opt_w_measured"]), kappa, C_G[dev]),
            TDP=econ(m_tdp, float(r["tdp_tok_s"]), p_tdp, kappa, C_G[dev])))
    return out


def cluster(dev, classes, w):
    """Demand-partitioned cluster per CLUSTER_MW of IT power, per policy.
    Racks per class: N_j ∝ w_j / X_j (class capacity matches its demand share), total rack
    count fixed by the power budget — same racks & power both policies."""
    n_racks = CLUSTER_MW * 1e6 / (DEV[dev]["budget_kw"] * 1e3)
    res = {}
    for pol in ("CAP", "TDP"):
        need = {c["klass"]: w[c["klass"]] / c[pol]["X"] for c in classes}
        z = sum(need.values())
        f = {k: v / z for k, v in need.items()}                     # rack share per class
        agg = {q: n_racks * sum(f[c["klass"]] * c[pol][q] for c in classes)
               for q in ("K", "D", "E", "M", "C", "Q", "R0", "X", "P_w", "m")}
        kin = n_racks * sum(f[c["klass"]] * c[pol]["X"] * c[pol]["kappa"] / (1 + c[pol]["kappa"])
                            for c in classes)
        kout = n_racks * sum(f[c["klass"]] * c[pol]["X"] / (1 + c[pol]["kappa"]) for c in classes)
        phi_n = float(phi(agg["R0"], agg["C"], N_YR))
        res[pol] = dict(agg, f=f, n_racks=n_racks, kappa_bar=kin / kout,
                        pi=agg["R0"] / agg["Q"], rho0=agg["R0"] - agg["C"],
                        phi_n=phi_n, roi_n=phi_n / agg["K"],
                        t_pb=payback(agg["R0"], agg["K"], agg["E"], agg["M"]),
                        c_tok=agg["C"] / agg["Q"],
                        gross=(agg["R0"] / agg["Q"] - agg["C"] / agg["Q"]) / (agg["R0"] / agg["Q"]))
    return res


fmt_m = lambda v: f"${v/1e6:.1f}M" if abs(v) >= 1e6 else f"${v/1e3:.0f}k"
fmt_pb = lambda t: "-" if t is None else (f"{t:.1f} yr" if t >= 1 else f"{t*12:.0f} mo")


def figure(results):
    t = np.linspace(0, N_YR, 400)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    for ax, dev in zip(axes, ("V100", "H200")):
        cl = results[dev]
        yc = phi(cl["CAP"]["R0"], cl["CAP"]["C"], t) / 1e6
        yt = phi(cl["TDP"]["R0"], cl["TDP"]["C"], t) / 1e6
        ax.plot(t, yc, color=GREEN, lw=2.6, zorder=3,
                label=f"POWER CAP — 32 GPU/rack ({cl['CAP']['m']:.0f} GPUs)")
        ax.plot(t, yt, color=RED, lw=2.6, zorder=3,
                label=f"NO CAP (TDP {DEV[dev]['tdp_w']} W) — 20 GPU/rack ({cl['TDP']['m']:.0f} GPUs)")
        ax.axhline(0, color="k", ls="--", lw=0.9)
        for pol, c_ in (("CAP", GREEN), ("TDP", RED)):              # cash payback on the t-axis
            pb = cl[pol]["t_pb"]
            if pb is not None and pb <= N_YR:
                ax.plot(pb, 0, "v", color=c_, ms=7, mec="white", mew=0.8, zorder=5)
        d5 = yc[-1] - yt[-1]
        ax.text(0.97, 0.30, f"ΔΦ₅ = {'+' if d5 >= 0 else '−'}{fmt_m(abs(d5)*1e6).replace('$', '')}  "
                f"({100*(yc[-1]/yt[-1]-1):+.0f}%)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5,
                color=GREEN if d5 >= 0 else RED, weight="bold")
        f = cl["CAP"]["f"]
        top = sorted(f.items(), key=lambda kv: -kv[1])[:3]
        esc = lambda v: fmt_m(v).replace("$", "\\$")
        ax.text(0.02, 0.975,
                f"ROI$_{{{N_YR:.0f}}}$   CAP {cl['CAP']['roi_n']:+.1f} · TDP {cl['TDP']['roi_n']:+.1f}\n"
                f"payback  CAP {fmt_pb(cl['CAP']['t_pb'])} · TDP {fmt_pb(cl['TDP']['t_pb'])}\n"
                f"capex  CAP {esc(cl['CAP']['K'])} · TDP {esc(cl['TDP']['K'])}\n"
                f"rack shares: " + " · ".join(f"{NAME[k].split(' (')[0]} {100*v:.0f}%" for k, v in top) +
                f" · rest {100*(1-sum(v for _, v in top)):.0f}%",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.2, color=MUTE,
                bbox=dict(boxstyle="round", fc="#fbfbf7", ec="#ccc", alpha=.92))
        ax.set_title(f"{dev} — {cl['CAP']['n_racks']:.0f} racks × {DEV[dev]['budget_kw']:.0f} kW "
                     f"(GPU ${C_G[dev]:,.0f})", fontsize=12, weight="bold")
        ax.set_xlabel("years of operation", fontsize=10)
        ax.set_ylabel(f"cumulative profit Φ(t) of the {CLUSTER_MW:.0f} MW cluster (M$)", fontsize=10)
        ax.set_xlim(0, N_YR)
        ax.grid(alpha=.35, color=GRID, lw=0.8)
        ax.legend(loc="lower right", bbox_to_anchor=(0.985, 0.05), fontsize=8.8, frameon=False)
        [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]
    fig.suptitle(
        f"Mixed-workload cumulative profit — POWER CAP vs TDP, {CLUSTER_MW:.0f} MW IT cluster, "
        f"{N_YR:.0f} yr   (paper economic model eq. 1-15)\n"
        f"racks partitioned by class, N_j ∝ w_j/X_j (capacity matches demand share); "
        f"w = measured token-volume mix (Chat+Code-dominated; composition method: ServeGen NSDI'26)   ·   "
        f"n={N_YR:.0f} yr S=0 · e=\\${ELEC:.2f}/kWh × PUE {PUE} · μ={MU:.0%} · "
        f"π=\\${PI_P_MTOK}/\\${PI_D_MTOK}/Mtok · λ={LAMBDA:.2f}/yr",
        fontsize=11)
    fig.text(0.5, 0.012,
             "y-axis = ACCRUAL cumulative profit: revenue to date (price decaying at λ) − (depreciation + electricity + maintenance)·t. "
             "Depreciation spreads K/n over the lifetime ⇒ curves start at 0 (capex is not a t=0 step); Φ(5) is eq. (11) summed over racks.\n"
             "▼ = CASH payback: cumulative (revenue − E − M) reaches the upfront K (eq. 13, λ>0). "
             "Same IT power & rack count both policies; both serve the same demand MIX and sell all output (100% utilization, latency/SLO not priced).",
             ha="center", fontsize=7.4, color=MUTE)
    fig.tight_layout(rect=(0, 0.065, 1, 0.985))
    out = os.path.join(HERE, "fig_profit_model.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    w = load_mix()
    results, all_rows = {}, []
    for dev in ("V100", "H200"):
        classes = load(dev)
        cl = cluster(dev, classes, w)
        results[dev] = cl
        print(f"\n{dev}: {cl['CAP']['n_racks']:.0f} racks/{CLUSTER_MW:.0f} MW · demand mix "
              f"kappa_bar(prod) CAP {cl['CAP']['kappa_bar']:.2f}:1")
        for pol in ("CAP", "TDP"):
            e_ = cl[pol]
            print(f"  {pol}: {e_['m']:.0f} GPUs, capex {fmt_m(e_['K'])}, X {e_['X']/1e6:.2f}M tok/s, "
                  f"Phi5 {fmt_m(e_['phi_n'])}, ROI5 {e_['roi_n']:+.2f}, payback {fmt_pb(e_['t_pb'])}")
        d5 = cl["CAP"]["phi_n"] - cl["TDP"]["phi_n"]
        print(f"  capping earns {fmt_m(d5)} ({100*(cl['CAP']['phi_n']/cl['TDP']['phi_n']-1):+.1f}%) "
              f"over TDP in {N_YR:.0f} yr per {CLUSTER_MW:.0f} MW")
        # CSV: cluster rows + per-class per-rack rows (w and rack shares included)
        for pol in ("CAP", "TDP"):
            e_ = cl[pol]
            all_rows.append(dict(device=dev, klass=f"MIXED cluster/{CLUSTER_MW:.0f}MW", policy=pol,
                                 w_token_share=1.0, f_rack_share=1.0, kappa=round(e_["kappa_bar"], 2),
                                 m_gpus=round(e_["m"]), X_tok_s=round(e_["X"], 1), P_w=round(e_["P_w"]),
                                 K_usd=round(e_["K"]), D_usd_yr=round(e_["D"]), E_usd_yr=round(e_["E"]),
                                 M_usd_yr=round(e_["M"]), C_usd_yr=round(e_["C"]),
                                 Q_tok_yr=f"{e_['Q']:.4g}", pi_usd_mtok=round(e_["pi"] * 1e6, 4),
                                 R0_usd_yr=round(e_["R0"]), rho0_usd_yr=round(e_["rho0"]),
                                 Phi_n_usd=round(e_["phi_n"]), ROI_n=round(e_["roi_n"], 3),
                                 T_pb_yr=round(e_["t_pb"], 2) if e_["t_pb"] is not None else "",
                                 c_tok_usd_mtok=round(e_["c_tok"] * 1e6, 4),
                                 gross_margin=round(e_["gross"], 3)))
        for c in classes:
            for pol in ("CAP", "TDP"):
                e_ = c[pol]
                pn = float(phi(e_["R0"], e_["C"], N_YR))
                all_rows.append(dict(device=dev, klass=c["klass"], policy=pol,
                                     w_token_share=round(w[c["klass"]], 4),
                                     f_rack_share=round(cl[pol]["f"][c["klass"]], 4),
                                     kappa=round(c["kappa"], 2), m_gpus=e_["m"],
                                     X_tok_s=round(e_["X"], 1), P_w=round(e_["P_w"]),
                                     K_usd=round(e_["K"]), D_usd_yr=round(e_["D"]),
                                     E_usd_yr=round(e_["E"]), M_usd_yr=round(e_["M"]),
                                     C_usd_yr=round(e_["C"]), Q_tok_yr=f"{e_['Q']:.4g}",
                                     pi_usd_mtok=round(e_["pi"] * 1e6, 4),
                                     R0_usd_yr=round(e_["R0"]), rho0_usd_yr=round(e_["rho0"]),
                                     Phi_n_usd=round(pn), ROI_n=round(pn / e_["K"], 3),
                                     T_pb_yr=(lambda p: round(p, 2) if p is not None else "")(
                                         payback(e_["R0"], e_["K"], e_["E"], e_["M"])),
                                     c_tok_usd_mtok=round(e_["c_tok"] * 1e6, 4),
                                     gross_margin=round(e_["gross"], 3)))
    figure(results)
    path = os.path.join(HERE, "profit_model.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wr.writeheader()
        [wr.writerow(r) for r in all_rows]
    print(f"\nwrote {os.path.basename(path)}")


if __name__ == "__main__":
    main()
