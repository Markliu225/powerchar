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
import palette                                  # single source: the paper palette
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
GREEN, RED, GOLD = palette.OK, palette.BAD, palette.HL
DEV_C = {"V100": palette.PAL["navy"], "RTX 5090": palette.PAL["orange"],
         "H200": palette.PAL["green"]}   # fig_profit_model: one panel per lambda, so
                                        # hue carries the DEVICE and the dash the policy
MUTE, GRID = palette.INK2, palette.GRID
# white halo behind any label that sits on top of a curve / the zero line (keeps text legible)
HALO = dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.85)
GUIDE = dict(boxstyle="round,pad=0.45", fc="#f7f6f1", ec="#d6d4ca", lw=0.9)   # symbol reading-guide box


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
    """The original grid: rows = device, cols = token-price decay lambda, colour = POLICY
    (green = POWER CAP, red = NO CAP/TDP), gold dot = T-cross. Kept at 3 x 4 so every device
    is shown against every decay rate; the type and the canvas are scaled to the rest of the
    paper's figures."""
    t = np.linspace(0, N_YR, 600)
    fig, axes = plt.subplots(len(DEVS), len(DECAYS), figsize=(26, 14.0), sharey="row", sharex=True)
    rows = []
    for i, dev in enumerate(DEVS):
        cl = dev_cl[dev]
        for j, lam in enumerate(DECAYS):
            ax = axes[i, j]
            yc = cashflow(cl["CAP"], lam, t) / 1e6
            yt = cashflow(cl["TDP"], lam, t) / 1e6
            ax.plot(t, yt, color=RED, lw=3.2, zorder=3, label="NO CAP (TDP)")
            ax.plot(t, yc, color=GREEN, lw=3.2, zorder=3, label="POWER CAP")
            ax.axhline(0, color="k", ls="--", lw=0.8)
            tx = first_cross(t, yc - yt)
            if tx is not None:
                yx = float(np.interp(tx, t, yc))
                ax.plot(tx, yx, "o", color=GOLD, ms=13, mec="white", mew=1.5, zorder=8)
                # the readout sits along the bottom edge, CENTRED: hung off the crossing it landed
                # on the frame, and in the left corner its halo covered the T× dot itself (every
                # crossing happens in the first months, i.e. at the very left of the panel).
                ax.text(0.46, 0.05, f"T× {fmt_pb(tx)}", transform=ax.transAxes, ha="center",
                        va="bottom", fontsize=24, color="black", weight="bold", bbox=HALO, zorder=7)
            phi_c = float(cashflow(cl["CAP"], lam, N_YR)); phi_t = float(cashflow(cl["TDP"], lam, N_YR))
            g = phi_c / phi_t
            if phi_c > 0 and phi_t > 0:                        # ratio only meaningful when both profit
                gtxt, gcol = f"G = {g:.2f}", "black"
            else:                                              # unprofitable: show signed extra, capping worse
                gtxt = f"CAP {fmt_m(phi_c-phi_t)} vs TDP" + ("\n(both lose)" if phi_t < 0 else "")
                gcol = "black"
            # bottom-right corner: guaranteed clear of both (monotone rising) curves
            ax.text(0.975, 0.05, gtxt, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=24, color=gcol, weight="bold", bbox=HALO, zorder=7)
            if i == 0:
                ax.set_title(f"λ = {lam*100:.0f} %/yr" + ("  (no decay)" if lam == 0 else ""),
                             fontsize=27, weight="bold")
            ax.legend(loc="upper left", fontsize=22, frameon=False,   # upper-left is empty in every panel
                      handlelength=1.7, borderpad=0.2, labelspacing=0.35, handletextpad=0.6)
            ax.grid(alpha=.35, color=GRID, lw=0.7)
            ax.tick_params(labelsize=24, colors="black")
            [s.set_visible(False) for s in (ax.spines["top"], ax.spines["right"])]
            if i == len(DEVS) - 1:
                ax.set_xlabel("years since deployment", fontsize=26, color="black")
            rows.append(dict(view="decay", device=dev, lambda_pct=round(lam * 100),
                             cap_K_usd=round(cl["CAP"]["K"]), tdp_K_usd=round(cl["TDP"]["K"]),
                             cap_Phi_n_usd=round(float(cashflow(cl["CAP"], lam, N_YR))),
                             tdp_Phi_n_usd=round(float(cashflow(cl["TDP"], lam, N_YR))),
                             T_cross_yr=round(tx, 3) if tx is not None else "",
                             G_end_ratio=round(g, 3)))
        # extra head-room below the curves so the T× labels never sit on the zero line / x-axis
        lo, hi = axes[i, 0].get_ylim()
        axes[i, 0].set_ylim(lo - 0.20 * (hi - lo), hi)                        # sharey='row' → whole row
        axes[i, 0].set_ylabel(dev, fontsize=28, weight="bold")   # a two-line label is now
    fig.text(0.013, 0.50, "net cash flow (M$)", rotation=90, ha="left", va="center",
             fontsize=27, weight="bold", color="black")   # taller than a row: name the quantity once here
    fig.suptitle(
        "Cumulative net cash flow — POWER CAP vs TDP, mixed workload, "
        f"{CLUSTER_MW:.0f} MW cluster over {N_YR:.0f} yr\n"
        "rows = device  ·  cols = token-price decay λ\n"
        "CF(t) = revenue(prices decaying at λ) − electricity − maintenance − upfront capex;  "
        "starts at −K (day-0 capex),  ends at Φ(n)",
        fontsize=28, y=0.995, color="black")
    fig.text(0.5, 0.858,   # symbol reading guide — what λ, T× and G mean
             "λ  =  annual token-price decay rate         ·         "
             "●  T×  =  extra-capex payback time         ·         "
             "G  =  end-of-life profit ratio",
             ha="center", va="top", fontsize=24, color="black", bbox=GUIDE)
    fig.text(0.5, 0.006,
             "mixed workload (per-class racks summed, N_j ∝ w_j/X_j; w = est. request shares × "
             "trace tokens/req, 7 II-C classes)\n"
             "100% util, SLO not priced  ·  ⚠ RTX 5090 row = MOCK data\n"
             f"n={N_YR:.0f} yr, S=0 · e=\\${ELEC:.2f}/kWh × PUE {PUE} · μ={MU:.0%}/yr · "
             "c_g " + " / ".join(f"\\${C_G[d]:,.0f} {d}" for d in DEVS) + "\n"
             f"per-class 2026 tier price × {PRICE_SCALE:.3g} small-model haircut\n"
             f"(reasoning/agentic \\${5*PRICE_SCALE:.2f}/\\${25*PRICE_SCALE:.2f} · long-ctx \\${3*PRICE_SCALE:.2f}/\\${15*PRICE_SCALE:.2f} · "
             f"chat/multimodal \\${2*PRICE_SCALE:.2f}/\\${10*PRICE_SCALE:.2f} · API/completion \\${1*PRICE_SCALE:.2f}/\\${5*PRICE_SCALE:.2f} per Mtok; "
             f"blended \\${dev_cl['V100']['CAP']['pi']*1e6:.2f})",
             ha="center", va="bottom", fontsize=21, color="black", linespacing=1.55)
    # explicit margins: tight_layout silently gives up when the reserved bands are this large
    fig.subplots_adjust(left=0.085, right=0.99, top=0.775, bottom=0.245,
                        hspace=0.30, wspace=0.10)
    out = os.path.join(HERE, "fig_profit_model.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return rows


# ---- GROUP 2: 1 x 2 mix-sensitivity of the CAP−TDP cash-flow difference -------------------------
def fig_group2(classes_by_dev, w0):
    """1 x 3 tornado, one panel per device. The ±MIX_SHIFT band is only a few percent wide, so
    drawing the fourteen per-perturbation CURVES was hopeless — every line sat inside the width of
    the base line. What those curves were meant to say is here instead: for each class, a bar
    spanning the end-of-horizon ΔΦ(n) that its demand share moves to when shifted ±MIX_SHIFT (the
    mix is renormalized and price / per-class rack counts re-solved at every point), dashed line =
    the base mix. Bars cannot overlap, they are sorted by demand share, and the class is named on
    the axis instead of by hue alone. The cash-flow curves themselves live in fig_profit_model.png."""
    t = np.linspace(0, N_YR, 600)
    lam = MID_DECAY
    fig, axes = plt.subplots(1, len(DEVS), figsize=(21.0, 9.2), gridspec_kw=dict(wspace=0.16))
    rows = []
    order = [k for k in sorted(w0, key=lambda k: -w0[k]) if w0[k] > 0]   # by demand share
    ccol = {k: palette.CLASS_OF.get(k, palette.PAL["gray"]) for k in order}
    short = lambda k: NAME[k].split(" (")[0]

    def dcf(dev, classes, w):
        cl = cluster(dev, classes, w)
        return (cashflow(cl["CAP"], lam, t) - cashflow(cl["TDP"], lam, t)) / 1e6, cl

    for j, dev in enumerate(DEVS):
        classes = classes_by_dev[dev]
        base_dy, base_cl = dcf(dev, classes, w0)
        dK = base_cl["CAP"]["K"] - base_cl["TDP"]["K"]
        base_end = float(base_dy[-1])

        span = {}
        for c in [c for c in classes if w0[c["klass"]] > 0]:   # active classes × {+20%, −20%}
            ends = []
            for fac in (1 + MIX_SHIFT, 1 - MIX_SHIFT):
                dy, _ = dcf(dev, classes, perturb_mix(w0, c["klass"], fac))
                ends.append(float(dy[-1]))
                tc = first_cross(t, dy)
                rows.append(dict(view="mix", device=dev,
                                 mix=f"{c['klass']} {'+' if fac > 1 else '-'}{MIX_SHIFT*100:.0f}%",
                                 dK_usd=round(float(dK)),
                                 T_cross_yr=round(tc, 3) if tc is not None else "",
                                 dPhi_n_usd=round(float(dy[-1]) * 1e6)))
            span[c["klass"]] = (min(ends), max(ends))

        ax = axes[j]
        for i, k in enumerate(order):
            k_lo, k_hi = span[k]
            ax.barh(i, k_hi - k_lo, left=k_lo, height=0.68, color=ccol[k], zorder=3)
        ax.axvline(base_end, color="k", ls="--", lw=1.8, zorder=4)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([f"{short(k)}  {w0[k]*100:.0f}%" for k in order] if j == 0 else [],
                           fontsize=27, color="black")
        ax.invert_yaxis()
        ax.set_title(f"({chr(97 + j)}) {dev}", fontsize=32, weight="bold", pad=12, color="black")
        ax.set_xlabel("ΔΦ(n) (M$)", fontsize=28, color="black")
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(alpha=.3, color=GRID, lw=0.7, axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=26, colors="black")
        [sp.set_visible(False) for sp in (ax.spines["top"], ax.spines["right"])]

    fig.legend(handles=[Patch(fc=palette.PAL["steel"],
                              label=f"range spanned by shifting that class's share ±{MIX_SHIFT*100:.0f}%"),
                        Line2D([], [], color="k", lw=1.8, ls="--", label="base mix (measured shares)")],
               loc="lower center", ncol=2, fontsize=27, frameon=False, bbox_to_anchor=(0.5, 0.008),
               columnspacing=1.4, handletextpad=0.6)
    fig.suptitle(
        f"Mix sensitivity — extra net profit ΔΦ(n) of CAP over TDP at n = {N_YR:.0f} yr, "
        f"λ = {MID_DECAY*100:.0f} %/yr\n"
        f"each of the {len(order)} classes' demand share shifted ±{MIX_SHIFT*100:.0f}% one at a "
        "time, renormalized (price and per-class rack counts re-solved)",
        fontsize=30, y=0.99, color="black")
    fig.subplots_adjust(left=0.155, right=0.985, top=0.775, bottom=0.235, wspace=0.16)
    out = os.path.join(HERE, "fig_profit_mix.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.basename(out)}")
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
