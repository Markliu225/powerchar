"""THE rack solver — one rack recipe per REAL workload class, under the physical GPU-slot limit.

CLASSIFY FIRST, THEN SOLVE. The portfolio experiment measured 10 REAL workloads (chat / RAG /
code / long-form / summarization / translation / classification / reasoning ...), each with ITS
OWN prefill curve and ITS OWN decode curve (context baked in). WORKLOAD_CLASSES below groups them
into 6 application classes by request shape (decode-dominant ... pure-prefill batch); the class —
not a numeric ratio — is the unit of presentation everywhere downstream. Under the router
assumption (每个机架专服一类应用, see WORKLOADS.zh.md) each workload gets its own rack recipe:
how many prefill/decode GPUs, each capped at what.

INPUTS (nothing hand-set here):
  curves   ../../pt_cap_gpu1/portfolio/data/<id>_{prefill,decode}.csv  (V100 v3 dataset)
           fitted with fitlib (THE canonical unified model): prefill single power law,
           decode additive three-stage.  x-axis = cap_w, the CONTROL variable — a rack is
           provisioned against the enforced cap, not against the fluctuating measured draw.
  classes  WORKLOAD_CLASSES below. Each class carries one canonical prefill:decode token-shape
           assumption used ONLY inside the balance equation; results are presented by class.

CONSTRAINTS: integer GPUs, >=1 per phase, Np+Nd <= N_GPU_MAX physical slots, caps confined to
the MEASURED cap range [100,250] W (no extrapolation), decode capped at its saturation point
(beyond it, watts buy no tokens), full budget spent by floating per-phase caps.

OUTPUT: printed table + workloads_results.csv (one rack recipe per workload, labeled by class).
"""
from __future__ import annotations
import csv, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PORT = os.path.join(ROOT, "pt_cap_gpu1", "portfolio")
sys.path.insert(0, PORT)
import fitlib                                   # noqa: E402
from portfolio import PORTFOLIO                 # noqa: E402

DATA = os.path.join(PORT, "data")
F_MAX = fitlib.resolve_f_max(DATA)

# ---- scenario knobs (match solve.py) ----
W_RACK = 5000.0        # rack power budget (W)
N_GPU_MAX = 32         # physical GPU slots per rack (e.g. 4 nodes x 8 V100)
P_TDP = 250.0          # nameplate cap (W)
CAP_LO, CAP_HI = 100.0, 250.0    # measured cap range — curves are never evaluated outside it

# ---- THE workload classification ----------------------------------------------------------------
# 6 application classes covering the 10 measured workloads, ordered decode-dominant -> pure-prefill.
# Each class carries ONE canonical request-shape assumption (P:D tokens) — a planning knob consistent
# with the application characters (chat decode-heavy, long-gen/CoT decode-dominant, translation
# symmetric, RAG/code prompt-heavy, summarization/classification ~prefill-only) and with the decode
# context each curve was measured at. The shape is used ONLY in the balance equation; every table
# and figure downstream is labeled by CLASS, never by the raw ratio.
WORKLOAD_CLASSES = [
    dict(key="long-gen", zh="长生成 / 推理", en="long-gen / reasoning", short="long-gen/CoT", pd=(1, 10),
         members=["longform-phi3", "qwen3think-4b"]),        # short brief/question -> long output/CoT
    dict(key="chat", zh="对话", en="chat", short="chat", pd=(1, 2),
         members=["chat-phi3", "fastchat-qwen15b", "qwen3chat-4b"]),
    dict(key="balanced", zh="翻译 / 对称", en="balanced (translate)", short="balanced", pd=(1, 1),
         members=["translate-qwen3b"]),
    dict(key="prompt-heavy", zh="RAG / 代码", en="prompt-heavy (RAG/code)", short="RAG/code", pd=(10, 1),
         members=["rag-phi3", "code-phi3"]),                 # long prompt -> short interactive answer
    dict(key="summarize-batch", zh="批量摘要", en="summarize (batch)", short="summarize", pd=(30, 1),
         members=["summarize-qwen7b"]),                      # 32k-class document -> short summary
    dict(key="classify-batch", zh="分类 / 抽取", en="classify (batch)", short="classify", pd=(100, 1),
         members=["classify-qwen7b"]),                       # vestigial decode
]
CLASS_OF = {m: c for c in WORKLOAD_CLASSES for m in c["members"]}
# presentation order: class order above, members in class order
ORDERED_IDS = [m for c in WORKLOAD_CLASSES for m in c["members"]]


# ---- per-workload curves from the portfolio data ------------------------------------------------
def _read(path):
    """Power axis = the ENFORCED cap when swept (V100, decode); when the cap is held fixed and the
    SM CLOCK is swept instead (H200 prefill, v-next), fall back to the MEASURED draw power_avg_w."""
    rows = [r for r in csv.DictReader(open(path)) if float(r["throughput_tok_s"]) > 0]
    cap = np.array([float(r["cap_w"]) for r in rows])
    pwr = np.array([float(r["power_avg_w"]) for r in rows])
    return (cap if np.ptp(cap) > 1e-6 else pwr,
            np.array([float(r["throughput_tok_s"]) for r in rows]),
            np.array([float(r["sm_clk_avg"]) for r in rows]),
            rows)


def load_workload(w):
    """Fit both phases of one workload in CAP space; return curve functions + fit metadata."""
    Pp, Tp, Fp, _ = _read(os.path.join(DATA, f"{w['id']}_prefill.csv"))
    Pd, Td, Fd, rows = _read(os.path.join(DATA, f"{w['id']}_decode.csv"))
    B = float(rows[0]["batch"])
    preT, pre = fitlib.fit_prefill_unified(Pp, Tp, Fp, F_MAX)
    decT, dec = fitlib.fit_decode_additive(Pd, Td, Fd, B, F_MAX)

    def Tpre(q):  return np.maximum(preT(np.clip(np.asarray(q, float), CAP_LO, CAP_HI)), 0.0)
    def Tdec(q):  return np.maximum(decT(np.clip(np.asarray(q, float), CAP_LO, CAP_HI)), 0.0)

    # decode saturation cap: smallest cap giving 99.5% of the throughput at CAP_HI — beyond it,
    # extra watts buy no tokens (stage III), so the optimizer never caps decode above this.
    g = np.linspace(CAP_LO, CAP_HI, 1501)
    td = Tdec(g)
    p_dec_hi = float(g[int(np.argmax(td >= 0.995 * td[-1]))])
    cls = CLASS_OF[w["id"]]
    return dict(w=w, cls=cls, Tpre=Tpre, Tdec=Tdec, pre=pre, dec=dec, p_dec_hi=p_dec_hi,
                Lp=float(cls["pd"][0]), Ld=float(cls["pd"][1]))


def sweet_spot(Tfun, hi=CAP_HI):
    """Efficiency sweet spot in cap space: argmax_P T(P)/P over the measured range."""
    g = np.linspace(CAP_LO, hi, 1501)
    eff = Tfun(g) / g
    i = int(np.argmax(eff))
    return float(g[i]), float(Tfun(g[i])), float(eff[i])


# ---- optimization core (solve.py's _alloc/solve_opt, generalized to arbitrary curves) -----------
def _alloc(c, Np, Nd, W):
    """Fixed integer split: float the per-phase caps to spend the whole budget on the balanced rate.
    Uniform caps within a phase are optimal (concave curves); decode never above its saturation cap."""
    if Np * CAP_LO + Nd * CAP_LO > W:
        return None
    pp_hi = min(CAP_HI, (W - Nd * CAP_LO) / Np)
    if pp_hi < CAP_LO:
        return None
    pp = np.linspace(CAP_LO, pp_hi, 600)
    pd = np.minimum(c["p_dec_hi"], (W - Np * pp) / Nd)
    ok = pd >= CAP_LO
    if not ok.any():
        return None
    pp, pd = pp[ok], pd[ok]
    Cp, Cd = Np * c["Tpre"](pp), Nd * c["Tdec"](pd)
    lam = np.minimum(Cp / c["Lp"], Cd / c["Ld"])          # bottleneck request rate (token balance)
    i = int(np.argmax(lam))
    ppi, pdi = float(pp[i]), float(pd[i])
    return dict(p_p=ppi, p_d=pdi, tot=float(lam[i] * (c["Lp"] + c["Ld"])), lam=float(lam[i]),
                Np=Np, Nd=Nd, pre_cap=float(Cp[i]), dec_cap=float(Cd[i]),
                w_used=Np * ppi + Nd * pdi, leftover=W - (Np * ppi + Nd * pdi))


def solve_opt(c, W=W_RACK, n_max=N_GPU_MAX):
    """Best integer (Np, Nd) with Np,Nd>=1 and Np+Nd<=n_max; ties broken toward less power."""
    def key(r):  return (round(r["tot"], 2), -round(r["w_used"], 1))
    best = None
    for Np in range(1, n_max):
        for Nd in range(1, n_max - Np + 1):
            r = _alloc(c, Np, Nd, W)
            if r and (best is None or key(r) > key(best)):
                best = r
    return best


def solve_tdp(c, W=W_RACK, n_max=N_GPU_MAX):
    """Baseline: every GPU at nameplate 250 W, integer fill within budget and slots."""
    tp, td = float(c["Tpre"](P_TDP)), float(c["Tdec"](P_TDP))
    best = None
    for Np in range(1, min(int((W - P_TDP) // P_TDP), n_max - 1) + 1):
        Nd = min(int((W - Np * P_TDP) // P_TDP), n_max - Np)
        if Nd < 1:
            break
        lam = min(Np * tp / c["Lp"], Nd * td / c["Ld"])
        tot = lam * (c["Lp"] + c["Ld"])
        if best is None or tot > best["tot"]:
            best = dict(p_p=P_TDP, p_d=P_TDP, tot=tot, lam=lam, Np=Np, Nd=Nd,
                        w_used=(Np + Nd) * P_TDP)
    return best


def cont_bound(c, W=W_RACK):
    """Continuous (fractional-GPU, no slot wall) ceiling: every watt at its phase's sweet spot."""
    ep, ed = sweet_spot(c["Tpre"])[2], sweet_spot(c["Tdec"], c["p_dec_hi"])[2]
    fp, fd = c["Lp"] / (c["Lp"] + c["Ld"]), c["Ld"] / (c["Lp"] + c["Ld"])
    return W / (fp / ep + fd / ed)


def main():
    print(f"rack {W_RACK:.0f} W, <= {N_GPU_MAX} GPU slots, caps confined to measured [{CAP_LO:.0f},{CAP_HI:.0f}] W"
          f"   (x = cap_w, the enforced knob)\n")
    print(f"{'class':<22}{'workload':<18} |{'OPT k/s':>8}{'Np':>3}{'Nd':>3}"
          f"{'pp':>5}{'pd':>5}{'used':>6}{'%Tot*':>7} |{'TDP k/s':>8}{'Np':>3}{'Nd':>3} |{'gain':>7}  binds")
    rows = []
    by_id = {w["id"]: w for w in PORTFOLIO}
    for wid in ORDERED_IDS:
        w = by_id[wid]
        c = load_workload(w)
        o = solve_opt(c)
        t = solve_tdp(c)
        ceil = cont_bound(c)
        gain = 100 * (o["tot"] / t["tot"] - 1)
        pct = 100 * o["tot"] / ceil
        binds = []
        if o["Np"] + o["Nd"] >= N_GPU_MAX:
            binds.append(f"N_max={N_GPU_MAX}")
        if o["Np"] == 1 and c["Lp"] < c["Ld"]:
            binds.append("Np=1")
        if o["Nd"] == 1 and c["Ld"] < c["Lp"]:
            binds.append("Nd=1")
        bind = "+".join(binds)
        pP, tP, eP = sweet_spot(c["Tpre"])
        pD, tD, eD = sweet_spot(c["Tdec"], c["p_dec_hi"])
        print(f"{c['cls']['en']:<22}{w['id']:<18} |{o['tot']/1e3:>8.2f}{o['Np']:>3d}{o['Nd']:>3d}"
              f"{o['p_p']:>5.0f}{o['p_d']:>5.0f}{o['w_used']:>6.0f}{pct:>6.1f}% |{t['tot']/1e3:>8.2f}"
              f"{t['Np']:>3d}{t['Nd']:>3d} |{gain:>6.1f}%  {bind}")
        rows.append({
            "id": w["id"], "class": c["cls"]["key"], "class_zh": c["cls"]["zh"],
            "model_id": w["model_id"],
            "decode_ctx": w["decode_ctx"], "decode_batch": w["decode_batch"],
            "dec_plateau_tok_s": round(c["dec"]["T_plateau"], 1),
            "pre_sweet_cap_w": round(pP), "pre_sweet_tok_per_j": round(eP, 2),
            "dec_sweet_cap_w": round(pD), "dec_sweet_tok_per_j": round(eD, 3),
            "dec_sat_cap_w": round(c["p_dec_hi"]),
            "opt_tok_s": round(o["tot"], 1), "opt_N_prefill": o["Np"], "opt_N_decode": o["Nd"],
            "opt_cap_prefill_w": round(o["p_p"]), "opt_cap_decode_w": round(o["p_d"]),
            "opt_w_used": round(o["w_used"]), "opt_rack_tok_per_j": round(o["tot"] / o["w_used"], 3),
            "opt_pct_of_bound": round(pct, 1),
            "tdp_tok_s": round(t["tot"], 1), "tdp_N_prefill": t["Np"], "tdp_N_decode": t["Nd"],
            "tdp_rack_tok_per_j": round(t["tot"] / t["w_used"], 3),
            "gain_opt_vs_tdp_pct": round(gain, 1), "constraint_binds": bind,
            "pre_fit_R2": round(c["pre"]["R2"], 3), "dec_fit_R2": round(c["dec"]["R2"], 3),
        })

    out = os.path.join(HERE, "workloads_results.csv")
    with open(out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        [wtr.writerow(r) for r in rows]
    print(f"\nwrote {os.path.basename(out)}  (one rack recipe per REAL workload; caps are cap_w settings"
          f" for nvidia-smi -pl; %Tot* = share of the continuous no-slot-wall ceiling)")


if __name__ == "__main__":
    main()
