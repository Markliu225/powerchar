"""Cumulative NET-CASH-FLOW figures for the paper economic model — capping vs TDP, mixed workload.

Two figure groups, both on the paper's economic model (经济模型 §, eq. 1-15), mixed workload:

GROUP 1  fig_profit_model.png — 2 rows (V100, H200) x 4 cols (price decay lambda = 0/10/20/30 %/yr).
  y = cumulative NET CASH FLOW = cumulative revenue - electricity - maintenance - upfront capex.
      CF(t) = pi*Q*(1-e^{-lambda t})/lambda - (E+M)*t - K   (lambda=0: pi*Q*t - (E+M)*t - K)
  Starts at -K (t=0), so the two policies start at DIFFERENT depths and CROSS; at t=n it equals
  the accrual profit Phi(n) of eq. (11) exactly, because D*n = K when S=0. Two curves per panel
  (TDP baseline, CAP), both the measured MIXED workload (per-class racks summed, blended price).
  Annotated: T_x = the two curves' crossover = payback period of the EXTRA capex; G = ratio of the
  two end values Phi_CAP(n)/Phi_TDP(n). Same y-scale per row. Reading: within a row T_x barely
  moves with lambda; decay only lowers the tail and shrinks G; both rows -> same conclusion.

GROUP 2  fig_profit_mix.png — 1 row x 2 cols (V100, H200), lambda fixed at 20 %/yr.
  y = CF_CAP(t) - CF_TDP(t), the capping-vs-baseline cash-flow DIFFERENCE. Starts at -dK (the extra
  investment; dK is mix-INDEPENDENT because every rack is 32 GPUs capped / 20 GPUs TDP regardless of
  class). Bold solid = the REAL mixed workload at its real proportions; the band = perturbing EACH
  class's demand share by +-20% one at a time and renormalizing (each perturbation re-solves the
  blended price and the per-class rack COUNTS N_j ∝ w_j/X_j). Annotated: the real-mix crossing = T_x,
  the band of crossings, the real end value and the [min,max] envelope of end values. Reading: the
  band is narrow and the crossings cluster — a +-20% error in any single class share doesn't flip it.

SHARED SETUP (实验设置): each class is solved for its baseline (TDP) and capped (OPT) rack recipe
independently ({v100,h200}/workload_rack_capping.csv); baseline power is the MEASURED draw; the
cluster is the sum over racks (1 MW IT power both policies -> 200 V100 / ~71 H200 racks, allocated
to classes by N_j ∝ w_j/X_j so capacity matches the demand mix). Per-device knobs: GPU cost c_g and
the throughput/power data. Economic knobs shared across the whole figure set: electricity e, PUE
beta, maintenance mu, life n, per-class token prices. The demand mix w_j is the TOKEN-VOLUME share
= estimated production REQUEST shares r_j across the 7 II-C classes × the classes' own trace
tokens/request (Lp_mean+Ld_mean, workload_classes.csv); both r_j and prices are replaceable knobs.

NOT MODELLED: latency/SLO; demand growth (utilization = 100%, all output sold).

python3 plot_profit_model.py -> fig_profit_model.png + fig_profit_mix.png + profit_model.csv
"""
from __future__ import annotations
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- economic parameters shared across the whole figure set (paper §5) ------------------------
C_G = {"V100": 2500.0, "RTX 5090": 3500.0, "H200": 27000.0}   # $/GPU deployed (sticker incl.
# server share; 5090: ~$2k street + chassis/network share — MOCK platform, see data_5090/)
N_YR = 5.0                 # service life n (paper: 4-6 yr)
SALVAGE = 0.0              # S, conservative baseline
ELEC = 0.10               # e, $/kWh
PUE = 1.1                 # beta
MU = 0.04                 # mu, maintenance fraction of K per year
# PER-CLASS token prices ($/Mtok input, output), eq. (8): each of the 7 II-C classes priced at the
# 2026 market rate of the model TIER that realistically serves it. Verified July 2026: flagship
# (Opus 4.8 / GPT-5.6 Sol) $5/$25-30 · Sonnet-class $3/$15 · Sonnet-5 $2/$10 · Haiku-4.5 $1/$5.
PRICE = {"推理": (5.0, 25.0),           # reasoning models = flagship tier
         "Agentic工具调用": (5.0, 25.0),  # agents run on the best models
         "长上下文对话": (3.0, 15.0),    # long-document work, Sonnet-class
         "多模态图文": (2.0, 10.0),      # vision-language, mid tier
         "对话": (2.0, 10.0),           # high-volume consumer chat, Sonnet-5 tier
         "助手API": (1.0, 5.0),         # programmatic API tasks, Haiku tier
         "代码补全": (1.0, 5.0)}        # IDE inline completion = fast/cheap models (agent coding is Agentic)
DEFAULT_PRICE = (1.0, 5.0)
# Haircut applied to every listed price. The throughput curves are MEASURED on small models
# (phi3-mini / qwen-3B / qwen-7B — what the testbed can run) while the list prices above are the
# market rate of the much larger models that really serve these classes; charging flagship rates for
# small-model token volume overstates absolute revenue. 1/3 is a deliberate conservative discount on
# that mismatch (see ECONOMICS.md §6). Set to 1.0 to recover raw list prices.
PRICE_SCALE = 1.0 / 3.0
T_YR_S = 365.0 * 86400.0  # seconds/yr (~3.15e7, eq. 6)
CLUSTER_MW = 1.0          # cluster IT power both policies are normalized to
DECAYS = [0.0, 0.10, 0.20, 0.30]   # price annual decay rates lambda (columns of group 1)
MID_DECAY = 0.20                   # lambda for group 2 (mix sensitivity)
MIX_SHIFT = 0.20                   # +-20% perturbation of ONE class's demand share (group 2)
W_OVERRIDE = None                  # {klass: token-share} to replace the derived demand mix

# ---- demand mix: production REQUEST shares x the classes' own trace tokens/request --------------
# r_j = share of total REQUESTS across the 7 II-C classes, research-grounded (ECONOMICS.md §3 for
# sources & per-class confidence): ServeGen Table 1 request counts (M-code 276M/wk = 24.6%, M-long
# 4.3%, reasoning 1.7% in Mar'25 right after R1 — understated), ChatGPT 2.5B prompts/day, Copilot
# 400M completions/day + Cursor billions/day, OpenRouter'25 reasoning-token growth, Kimi FAST'25
# toolagent:conversation ~2:1. Converted to TOKEN-VOLUME shares w_j via each class's trace
# Lp_mean+Ld_mean (workload_classes.csv), renormalized. Replace via R_SHARE / W_OVERRIDE.
R_SHARE = {"对话": 30.0,            # [med]  ChatGPT-scale consumer chat, diluted by reasoning routing
           "代码补全": 27.0,        # [high] ServeGen M-code 24.6% + Copilot/Cursor volumes
           "推理": 14.0,            # [med]  post-R1 growth; OpenRouter reasoning tokens 0 -> >50%
           "Agentic工具调用": 11.0, # [low]  Kimi toolagent 2:1; Anthropic index 77% automation
           "助手API": 10.0,         # [low]  ServeGen M-large ~4.9% batched API, grown modestly
           "长上下文对话": 4.0,     # [med]  ServeGen M-long 4.3% measured
           "多模态图文": 4.0}       # [low]  ServeGen 1.4% in early'25, omni growth uplift
# -----------------------------------------------------------------------------------------------

DEV = {"V100": dict(csv="v100", budget_kw=5.0, tdp_w=250),
       "RTX 5090": dict(csv="5090", budget_kw=11.5, tdp_w=575),   # MOCK data (data_5090/)
       "H200": dict(csv="h200", budget_kw=14.0, tdp_w=700)}
DEVS = list(DEV)
NAME = {"推理": "Reasoning", "助手API": "Assistant API", "多模态图文": "Multimodal",
        "对话": "Chat (dialogue)", "长上下文对话": "Long-context chat",
        "Agentic工具调用": "Agentic tool-use", "代码补全": "Code completion"}
GREEN, RED, GOLD, MUTE, GRID = "#2ca02c", "#d62728", "#cc6600", "#52514e", "#e1e0d9"


# ---- per-rack economics (eq. 1-7; lambda-independent) -----------------------------------------
def econ(m, X, P_w, kappa, c_g, price):
    """price = (pi_p, pi_d) for THIS class in $/Mtok (eq. 8 per-class pricing)."""
    pi_p, pi_d = price[0] * PRICE_SCALE / 1e6, price[1] * PRICE_SCALE / 1e6
    K = m * c_g
    D = (K - SALVAGE) / N_YR
    E = ELEC * PUE * (P_w / 1000.0) * 8760.0
    M = MU * K
    Q = X * T_YR_S
    pi = (kappa * pi_p + pi_d) / (1.0 + kappa)
    return dict(m=m, X=X, kappa=kappa, K=K, D=D, E=E, M=M, C=D + E + M, Q=Q, pi=pi, R0=pi * Q)


def load(dev):
    """Per-class per-policy rack economics + kappa, straight from the II-C rack solver CSV
    (already solved at each class's trace rho-bar — no per-class overrides needed)."""
    rows = list(csv.DictReader(open(os.path.join(HERE, DEV[dev]["csv"], "workload_rack_capping.csv"),
                                    encoding="utf-8")))
    out = []
    for r in rows:
        price = PRICE.get(r["klass"], DEFAULT_PRICE)
        kappa = float(r["ratio_agg"])
        m_opt = int(r["opt_N_prefill"]) + int(r["opt_N_decode"])
        m_tdp = int(r["tdp_N_prefill"]) + int(r["tdp_N_decode"])
        p_tdp = float(r["tdp_tok_s"]) / float(r["tdp_rack_tok_per_j"])   # measured draw (W)
        out.append(dict(klass=r["klass"], kappa=kappa,
                        CAP=econ(m_opt, float(r["opt_tok_s"]), float(r["opt_w_measured"]), kappa, C_G[dev], price),
                        TDP=econ(m_tdp, float(r["tdp_tok_s"]), p_tdp, kappa, C_G[dev], price)))
    return out


def load_mix():
    """Demand mix w_j = TOKEN-VOLUME share = r_j (R_SHARE request shares) × L_j (the class's own
    trace tokens/request, Lp_mean+Ld_mean from workload_classes.csv), renormalized.
    W_OVERRIDE replaces the whole mix."""
    if W_OVERRIDE is not None:
        z = sum(W_OVERRIDE.values())
        return {k: v / z for k, v in W_OVERRIDE.items()}
    rows = list(csv.DictReader(open(os.path.join(HERE, "workload_classes.csv"), encoding="utf-8")))
    L = {r["klass"]: float(r["Lp_mean"]) + float(r["Ld_mean"]) for r in rows}
    vol = {k: R_SHARE.get(k, 0.0) * L.get(k, 0.0) for k in NAME}
    z = sum(vol.values())
    return {k: v / z for k, v in vol.items()}


def perturb_mix(w, klass, factor):
    """Scale ONE class's demand share by `factor` (e.g. 1.2 / 0.8) and renormalize — the per-class
    ±20% share-estimation perturbation. Every other class rescales to keep sum(w)=1."""
    ww = dict(w)
    ww[klass] = w[klass] * factor
    z = sum(ww.values())
    return {k: v / z for k, v in ww.items()}


def cluster(dev, classes, w):
    """Demand-partitioned cluster per CLUSTER_MW, per policy. Racks per class N_j ∝ w_j/X_j
    (capacity matches demand share); rack count fixed by power budget (same racks both policies)."""
    n_racks = CLUSTER_MW * 1e6 / (DEV[dev]["budget_kw"] * 1e3)
    res = {}
    for pol in ("CAP", "TDP"):
        need = {c["klass"]: w[c["klass"]] / c[pol]["X"] for c in classes}
        z = sum(need.values())
        f = {k: v / z for k, v in need.items()}
        agg = {q: n_racks * sum(f[c["klass"]] * c[pol][q] for c in classes)
               for q in ("K", "E", "M", "C", "Q", "R0", "X", "m")}
        res[pol] = dict(agg, f=f, n_racks=n_racks, pi=agg["R0"] / agg["Q"])
    return res


# ---- cash flow & accrual (eq. 11 / 13) --------------------------------------------------------
def cashflow(cl, lam, t):
    """Cumulative net cash flow: revenue(decayed) - (E+M)t - K. At t=n equals accrual Phi(n)."""
    t = np.asarray(t, float)
    rev = cl["R0"] * t if lam == 0 else cl["R0"] * (1.0 - np.exp(-lam * t)) / lam
    return rev - (cl["E"] + cl["M"]) * t - cl["K"]


def first_cross(t, y):
    """First upward zero-crossing of y(t) (interpolated); None if it never crosses."""
    s = np.sign(y)
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    if not len(idx):
        return None
    i = idx[0]
    return float(t[i] + (t[i + 1] - t[i]) * (-y[i]) / (y[i + 1] - y[i]))


fmt_m = lambda v: (f"{'-' if v < 0 else ''}${abs(v)/1e6:.1f}M" if abs(v) >= 1e6
                   else f"{'-' if v < 0 else ''}${abs(v)/1e3:.0f}k")
fmt_pb = lambda t: "—" if t is None else (f"{t:.1f} yr" if t >= 1 else f"{t*12:.1f} mo")


# ---- GROUP 1: 2 (device) x 4 (decay) net-cash-flow panels --------------------------------------
def fig_group1(dev_cl):
    t = np.linspace(0, N_YR, 600)
    fig, axes = plt.subplots(len(DEVS), len(DECAYS), figsize=(19, 12.0), sharey="row", sharex=True)
    rows = []
    for i, dev in enumerate(DEVS):
        cl = dev_cl[dev]
        for j, lam in enumerate(DECAYS):
            ax = axes[i, j]
            yc = cashflow(cl["CAP"], lam, t) / 1e6
            yt = cashflow(cl["TDP"], lam, t) / 1e6
            ax.plot(t, yt, color=RED, lw=2.3, zorder=3, label="NO CAP (TDP)")
            ax.plot(t, yc, color=GREEN, lw=2.3, zorder=3, label="POWER CAP")
            ax.axhline(0, color="k", ls="--", lw=0.8)
            tx = first_cross(t, yc - yt)
            if tx is not None:
                yx = float(np.interp(tx, t, yc))
                ax.plot(tx, yx, "o", color=GOLD, ms=8, mec="white", mew=0.9, zorder=6)
                # curves rise left-to-right, so the upper-left corner never holds a curve —
                # both readouts live there instead of hugging the lines they describe
                ax.annotate(f"T× {fmt_pb(tx)}", (0.05, 0.83), xycoords="axes fraction",
                            fontsize=13, color=GOLD, weight="bold", va="top")
            phi_c = float(cashflow(cl["CAP"], lam, N_YR)); phi_t = float(cashflow(cl["TDP"], lam, N_YR))
            g = phi_c / phi_t
            if phi_c > 0 and phi_t > 0:                        # ratio only meaningful when both profit
                gtxt, gcol = f"G = {g:.2f}", GREEN
            else:                                              # unprofitable: show signed extra, capping worse
                gtxt = f"CAP {fmt_m(phi_c-phi_t)} vs TDP" + ("\n(both lose)" if phi_t < 0 else "")
                gcol = RED
            ax.annotate(gtxt, (0.05, 0.95), xycoords="axes fraction",
                        fontsize=13.5, color=gcol, weight="bold", va="top")
            if i == 0:
                ax.set_title(f"λ = {lam*100:.0f} %/yr" + ("  (no decay)" if lam == 0 else ""),
                             fontsize=15.5, weight="bold")
            ax.grid(alpha=.35, color=GRID, lw=0.7)
            ax.tick_params(labelsize=16, colors=MUTE)
            [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]
            if i == len(DEVS) - 1:
                ax.set_xlabel("years since deployment", fontsize=20, color=MUTE)
            rows.append(dict(view="decay", device=dev, lambda_pct=round(lam * 100),
                             cap_K_usd=round(cl["CAP"]["K"]), tdp_K_usd=round(cl["TDP"]["K"]),
                             cap_Phi_n_usd=round(float(cashflow(cl["CAP"], lam, N_YR))),
                             tdp_Phi_n_usd=round(float(cashflow(cl["TDP"], lam, N_YR))),
                             T_cross_yr=round(tx, 3) if tx is not None else "",
                             G_end_ratio=round(g, 3)))
        axes[i, 0].set_ylabel(f"{dev}", fontsize=21, weight="bold")
    axes[0, 0].legend(loc="lower right", fontsize=13, frameon=False)
    fig.suptitle(
        "Cumulative net cash flow — POWER CAP vs TDP, mixed workload, "
        f"{CLUSTER_MW:.0f} MW cluster over {N_YR:.0f} yr   (rows = device · cols = token-price decay λ)\n"
        "CF(t) = revenue(price ↓ at λ) − electricity − maintenance − upfront capex; starts at −K, "
        "ends at accrual profit Φ(n).  ● T× = crossover = extra-capex payback · G = Φ_CAP(n)/Φ_TDP(n)",
        fontsize=16)
    fig.text(0.5, 0.008,
             f"mixed workload (per-class racks summed, N_j ∝ w_j/X_j; w = est. request shares × trace tokens/req, 7 II-C classes)  ·  "
             f"n={N_YR:.0f} yr S=0 · e=\\${ELEC:.2f}/kWh × PUE {PUE} · μ={MU:.0%}\n"
             f"per-class 2026 tier price × {PRICE_SCALE:.3g} small-model haircut "
             f"(reasoning/agentic \\${5*PRICE_SCALE:.2f}/\\${25*PRICE_SCALE:.2f} · long-ctx \\${3*PRICE_SCALE:.2f}/\\${15*PRICE_SCALE:.2f} · "
             f"chat/multimodal \\${2*PRICE_SCALE:.2f}/\\${10*PRICE_SCALE:.2f} · API/completion \\${1*PRICE_SCALE:.2f}/\\${5*PRICE_SCALE:.2f} per Mtok; "
             f"blended \\${dev_cl['V100']['CAP']['pi']*1e6:.2f}) · "
             "c_g " + "/".join(f"\\${C_G[d]:,.0f} {d}" for d in DEVS)
             + "  ·  100% util, SLO not priced  ·  ⚠ RTX 5090 row = MOCK data",
             ha="center", fontsize=11.5, color=MUTE)
    fig.supylabel("cumulative net cash flow (M$)", fontsize=20)
    fig.tight_layout(rect=(0.025, 0.045, 1, 0.94))
    out = os.path.join(HERE, "fig_profit_model.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return rows


# ---- GROUP 2: 1 x 2 mix-sensitivity of the CAP−TDP cash-flow difference -------------------------
def fig_group2(classes_by_dev, w0):
    """Real mix (solid) vs perturbing EACH class's demand share by ±MIX_SHIFT one at a time
    (renormalized). Every perturbation re-solves the blended price and the per-class rack counts;
    the curves form a sensitivity band around the real mix. λ fixed at MID_DECAY."""
    t = np.linspace(0, N_YR, 600)
    lam = MID_DECAY
    INK = "#0b0b0b"
    fig, axes = plt.subplots(1, len(DEVS), figsize=(20, 7.2), sharex=True,
                             gridspec_kw=dict(wspace=0.24))
    rows = []
    # one color per class (both its +20% and −20% share the color) so it is visually obvious that
    # EVERY class is perturbed, not just one; order/colors shared across both panels, by demand share.
    # Only classes actually in the mix (w>0) are shown/perturbed.
    order = [k for k in sorted(w0, key=lambda k: -w0[k]) if w0[k] > 0]
    cmap = plt.get_cmap("tab10")
    ccol = {k: cmap(i % 10) for i, k in enumerate(order)}

    def dcf(dev, classes, w):
        cl = cluster(dev, classes, w)
        return (cashflow(cl["CAP"], lam, t) - cashflow(cl["TDP"], lam, t)) / 1e6, cl

    for ax, dev in zip(axes, DEVS):
        classes = classes_by_dev[dev]
        base_dy, base_cl = dcf(dev, classes, w0)
        dK = base_cl["CAP"]["K"] - base_cl["TDP"]["K"]

        curves, txs = [], []
        for c in [c for c in classes if w0[c["klass"]] > 0]:   # active classes × {+20%, −20%}
            for fac in (1 + MIX_SHIFT, 1 - MIX_SHIFT):
                dy, _ = dcf(dev, classes, perturb_mix(w0, c["klass"], fac))
                ax.plot(t, dy, color=ccol[c["klass"]], lw=0.85, alpha=.7, zorder=3)
                curves.append(dy)
                tc = first_cross(t, dy)
                if tc is not None:
                    txs.append(tc)
                rows.append(dict(view="mix", device=dev,
                                 mix=f"{c['klass']} {'+' if fac > 1 else '-'}{MIX_SHIFT*100:.0f}%",
                                 dK_usd=round(float(dK)),
                                 T_cross_yr=round(tc, 3) if tc is not None else "",
                                 dPhi_n_usd=round(float(dy[-1]) * 1e6)))
        band = np.array(curves)
        lo, hi = band.min(0), band.max(0)
        ax.fill_between(t, lo, hi, color="#9a9a9a", alpha=.16, zorder=1, lw=0)   # sensitivity envelope
        ax.plot(t, base_dy, color=INK, lw=1.3, zorder=6, label="real mix (measured proportions)")
        ax.axhline(0, color="k", ls="--", lw=0.8)

        tx0 = first_cross(t, base_dy)
        if tx0 is not None:
            ax.plot(tx0, 0, "o", color=INK, ms=6, mec="white", mew=0.9, zorder=7)
        # no in-panel text — every readout lives in profit_model.csv / ECONOMICS.md; the y-range is
        # clamped to the band (plus margin) so the curve fan fills the panel instead of huddling
        y_pad = (hi[-1] - min(0.0, float(-dK / 1e6))) * 0.08
        ax.set_ylim(min(float(-dK / 1e6), float(lo.min())) - y_pad, float(hi.max()) + y_pad)
        ax.set_title(f"{dev}", fontsize=16.5, weight="bold")
        ax.set_xlabel("years since deployment", fontsize=20)
        ax.set_ylabel("CAP − TDP net cash flow (M$)", fontsize=20)
        ax.grid(alpha=.3, color=GRID, lw=0.7)
        ax.tick_params(labelsize=16, colors=MUTE)
        [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]

    # shared legend: bold real-mix line + one swatch per perturbed class (share % in the label)
    shr = lambda k: f"{w0[k]*100:.0f}%" if w0[k] >= 0.01 else "<1%"
    handles = [Line2D([], [], color=INK, lw=1.6, label="base mix (request shares × trace tokens)")]
    handles += [Line2D([], [], color=ccol[k], lw=2.2,
                       label=f"{NAME[k].split(' (')[0]} {shr(k)}  (±{MIX_SHIFT*100:.0f}%)")
                for k in order]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=13.5, frameon=False,
               columnspacing=2.0, handletextpad=0.8, labelspacing=0.7,
               bbox_to_anchor=(0.5, -0.085))
    fig.suptitle(
        f"Mix sensitivity — CAP vs TDP net-cash-flow difference, λ = {MID_DECAY*100:.0f} %/yr   "
        f"(real mixed workload; starts at −ΔK, zero-crossing = T×, end value = extra net profit)\n"
        f"each of the {len(order)} classes' demand share perturbed ±{MIX_SHIFT*100:.0f}% one at a time "
        "(color = perturbed class, both directions), renormalized (price & per-class rack counts re-solved)",
        fontsize=15.5)
    fig.tight_layout(rect=(0, 0.15, 1, 0.88))
    out = os.path.join(HERE, "fig_profit_mix.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return rows


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")           # console progress uses λ/Φ/κ
    except AttributeError:
        pass
    w = load_mix()
    classes_by_dev = {dev: load(dev) for dev in DEVS}
    dev_cl = {dev: cluster(dev, classes_by_dev[dev], w) for dev in DEVS}

    for dev in DEVS:
        cl = dev_cl[dev]
        print(f"\n{dev}: {cl['CAP']['n_racks']:.0f} racks/{CLUSTER_MW:.0f} MW · "
              f"CAP {cl['CAP']['m']:.0f} GPU K={fmt_m(cl['CAP']['K'])} · TDP {cl['TDP']['m']:.0f} GPU K={fmt_m(cl['TDP']['K'])}")
        for lam in DECAYS:
            g = cashflow(cl["CAP"], lam, N_YR) / cashflow(cl["TDP"], lam, N_YR)
            tx = first_cross(np.linspace(0, N_YR, 4000),
                             cashflow(cl["CAP"], lam, np.linspace(0, N_YR, 4000))
                             - cashflow(cl["TDP"], lam, np.linspace(0, N_YR, 4000)))
            print(f"  λ={lam*100:>2.0f}%: Φ_CAP={fmt_m(float(cashflow(cl['CAP'],lam,N_YR)))} "
                  f"Φ_TDP={fmt_m(float(cashflow(cl['TDP'],lam,N_YR)))} T×={fmt_pb(tx)} G={g:.2f}")

    rows1 = fig_group1(dev_cl)
    rows2 = fig_group2(classes_by_dev, w)

    print(f"\nmix sensitivity (λ={MID_DECAY*100:.0f}%, each class ±{MIX_SHIFT*100:.0f}% one at a time):")
    for dev in DEVS:
        rr = [r for r in rows2 if r["device"] == dev]
        ends = [r["dPhi_n_usd"] for r in rr]
        txs = [r["T_cross_yr"] for r in rr if r["T_cross_yr"] != ""]
        hi = max(rr, key=lambda r: r["dPhi_n_usd"]); lo = min(rr, key=lambda r: r["dPhi_n_usd"])
        tx_str = f"{min(txs)*12:.1f}–{max(txs)*12:.1f} mo" if txs else "no crossover in window"
        print(f"  {dev}: ΔΦ(n) band [{fmt_m(min(ends))}, {fmt_m(max(ends))}] "
              f"(lo={lo['mix']}, hi={hi['mix']}) · T× {tx_str}")

    path = os.path.join(HERE, "profit_model.csv")
    keys = ["view", "device", "lambda_pct", "mix", "cap_K_usd", "tdp_K_usd", "dK_usd",
            "cap_Phi_n_usd", "tdp_Phi_n_usd", "dPhi_n_usd", "T_cross_yr", "G_end_ratio"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        [wr.writerow(r) for r in rows1 + rows2]
    print(f"wrote {os.path.basename(path)}")


if __name__ == "__main__":
    main()
