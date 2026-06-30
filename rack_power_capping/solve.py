"""Rack-level power-capping optimizer — driven by the CURRENT measured-fitted curves.

PROBLEM. Requests arrive with a FIXED prefill:decode token ratio P:D. Given a rack power
budget W, decide: how many GPUs, each in which phase (prefill / decode), each at what
per-GPU power cap, to MAXIMISE total (balanced) token throughput.

PER-GPU CURVES come straight from ../pt_cap_gpu1/plot_theory.py — the models fitted to the latest
measured prefill.csv / decode.csv on this V100-GPU1 — so this solver ALWAYS matches that figure
(including any hand-set PRE_*/DEC_* parameters there):
    prefill (compute-bound):  T(P) = inverse of  P = P0 + kappa*T*(1+rho*T)^2     (V²f, rises to TDP)
    decode  (memory-bound) :  T(P) = min( T_{V²f}(P), T_max )                     (piecewise roofline)
Each phase's efficiency (tok/J) sweet spot = argmax_P T(P)/P, found numerically below. For decode the
sweet spot IS the roofline knee: above it throughput is flat at T_max, so power there is pure waste.

WHY 'EVERYONE AT THE SWEET SPOT' IS NOT OPTIMAL. The continuous bound at full budget is
        Tot* = W / (f_p/eta_p + f_d/eta_d),    f_p=P/(P+D), f_d=D/(P+D),
which says: spend every watt at its phase's max efficiency eta and balance prefill:decode capacity =
P:D. But you cannot buy fractional GPUs. Forcing EVERY GPU to the sweet-spot cap leaves the rack with
SURPLUS: leftover budget too small for one more sweet-spot GPU, plus a capacity imbalance whose
over-provisioned side wastes tokens. That surplus is NOT the rack's true max throughput.

OPTIMUM (this solver, solve_opt). For each integer split (Np prefill, Nd decode) GPUs we spend the
WHOLE budget by choosing per-PHASE caps that maximise the balanced rate — pushing the bottleneck phase
ABOVE its sweet spot to burn the leftover (prefill keeps rising to TDP), and pulling the
over-provisioned phase BELOW its sweet spot to free budget for one more GPU on the bottleneck side.
Uniform caps within a phase are optimal (both curves concave -> equal split maximises the phase sum),
so the only knobs are (Np, Nd) and the prefill:decode power split, all searched below. This reaches
the rack's true max (within ~0.3% of Tot*), vs the fixed sweet-spot policy and the 250 W TDP baseline.

CONSTRAINT: integer GPUs, >=1 per phase (disaggregated serving needs >=1 prefill worker AND >=1
decode worker). At extreme ratios the >=1 floor binds (e.g. decode-heavy 1:20 still spends one
whole GPU on prefill — so that GPU is throttled to the floor to free budget for decode). See solve_opt().
"""
from __future__ import annotations
import csv, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_PT = os.path.join(os.path.dirname(HERE), "pt_cap_gpu1")
sys.path.insert(0, _PT)
import plot_theory as pt                      # noqa: E402  (import-safe: its main() is guarded)

# ---- per-GPU curves: the SAME models plot_theory.py fits to the latest measured data ----
_Pp, _Tp = pt.read(pt.PREFILL_CSV, "power_avg_w", "throughput_tok_s", lambda r: r["phase"] == "prefill")
_Pf, _Tf = pt.read(pt.FRONTIER_CSV, "power_w", "throughput_tok_s")
_preT, PRE = pt.fit_prefill(_Pp, _Tp)         # T_pre(P) + params dict {P0,kappa,rho,R2,manual}
_decT, DEC = pt.fit_decode(_Pf, _Tf)          # T_dec(P) + params dict {...,T_max}


def TpreV(p):  return np.maximum(_preT(np.asarray(p, float)), 0.0)         # prefill tok/s, vectorised
def TdecV(p):  return np.maximum(_decT(np.asarray(p, float)), 0.0)         # decode  tok/s, vectorised
def Tpre(p):   return float(TpreV(p))                                      # prefill tok/s at cap p
def Tdec(p):   return float(TdecV(p))                                      # decode  tok/s at cap p


# ---- scenario knobs ----
W_RACK = 5000.0                               # rack power budget (W); results scale linearly
P_TDP = 250.0                                 # nameplate TDP per GPU (W)
CTX = 256                                     # context the decode curve was measured at (label only)
P_PRE_LO, P_DEC_LO = float(_Pp.min()), float(_Pf.min())   # measured min power each phase can run at
P_PRE_HI = P_TDP                              # prefill is compute-bound -> keeps rising to nameplate


def best_eff(Tfun, lo, hi=P_TDP):
    """Efficiency (tok/J) sweet spot of a curve: argmax_P T(P)/P over [lo, hi]."""
    ps = np.linspace(lo, hi, 4000)
    eff = np.array([Tfun(p) / p for p in ps]); i = int(np.argmax(eff))
    return float(ps[i]), float(Tfun(ps[i])), float(eff[i])


def best_eff_prefill():  return best_eff(Tpre, P_PRE_LO)
def best_eff_decode():   return best_eff(Tdec, P_DEC_LO)

P_PRE_OPT = best_eff_prefill()[0]             # prefill efficiency sweet-spot power (W)
P_DEC_OPT = best_eff_decode()[0]              # decode  efficiency sweet-spot power (W) == roofline knee
P_DEC_HI = P_DEC_OPT                           # decode is flat (T_max) above the knee -> never exceed it


def solve(P, D, p_p, p_d, W=W_RACK):
    """FIXED per-GPU caps p_p/p_d -> INTEGER allocation with >=1 GPU in EACH phase (the TDP baseline).

    Whole GPUs, Np>=1 and Nd>=1. The balanced workload is served at request rate
    lam = min(prefill_capacity/P, decode_capacity/D); useful throughput = lam*(P+D). Sweep integer
    Np>=1, fill the rest of the budget with Nd>=1 decode GPUs, keep the best. None if budget < 1+1.
    NOTE: with one cap per phase this generally leaves SURPLUS budget — see solve_opt for the real max."""
    tp, td = Tpre(p_p), Tdec(p_d)
    best = None
    npre_max = int((W - p_d) // p_p)              # leave room for >=1 decode GPU
    for npre in range(1, npre_max + 1):
        ndec = int((W - npre * p_p) // p_d)       # fill remaining budget with decode
        if ndec < 1:
            break                                 # ndec falls as npre rises -> budget exhausted
        lam = min(npre * tp / P, ndec * td / D)   # bottleneck request rate (token balance)
        tot = lam * (P + D)                       # useful tok/s (excess capacity wasted)
        if best is None or tot > best["tot"]:
            best = dict(p_p=p_p, p_d=p_d, tp=tp, td=td, tot=tot, lam=lam, Np=npre, Nd=ndec,
                        Wp=npre * p_p, Wd=ndec * p_d, pre_cap=npre * tp, dec_cap=ndec * td,
                        w_used=npre * p_p + ndec * p_d, leftover=W - (npre * p_p + ndec * p_d))
    return best


def _alloc(Np, Nd, P, D, W):
    """Fixed integer (Np prefill, Nd decode) GPUs: pick the per-PHASE caps that spend the WHOLE budget
    on the balanced rate. Uniform within a phase is optimal (concave curves -> equal split is best), so
    the only freedom is the prefill:decode power split — swept here. Prefill may go up to TDP; decode is
    capped at its roofline knee (above it is wasted). Returns None if even the power floors don't fit."""
    if Np * P_PRE_LO + Nd * P_DEC_LO > W:                 # can't even host them at minimum power
        return None
    pp_hi = min(P_PRE_HI, (W - Nd * P_DEC_LO) / Np)       # most prefill power that still leaves decode >= floor
    if pp_hi < P_PRE_LO:
        return None
    pp = np.linspace(P_PRE_LO, pp_hi, 600)                # sweep prefill cap; decode gets the remaining budget
    pd = np.minimum(P_DEC_HI, (W - Np * pp) / Nd)         # decode cap (clamped at the knee -> any excess is slack)
    Cp, Cd = Np * TpreV(pp), Nd * TdecV(pd)               # total prefill / decode token capacity (tok/s)
    lam = np.minimum(Cp / P, Cd / D)                      # bottleneck request rate (token balance)
    i = int(np.argmax(lam))
    ppi, pdi = float(pp[i]), float(pd[i])
    wp, wd = Np * ppi, Nd * pdi
    return dict(p_p=ppi, p_d=pdi, tp=float(TpreV(ppi)), td=float(TdecV(pdi)),
                tot=float(lam[i] * (P + D)), lam=float(lam[i]), Np=Np, Nd=Nd, Wp=wp, Wd=wd,
                pre_cap=float(Cp[i]), dec_cap=float(Cd[i]), w_used=wp + wd, leftover=W - (wp + wd))


def solve_opt(P, D, W=W_RACK):
    """TRUE rack maximum: search integer (Np, Nd), each split solved by _alloc to spend the full budget.

    Unlike solve() with one fixed cap per phase, here the per-phase caps float so the bottleneck phase
    is pushed above its sweet spot (burning the leftover) and the over-provisioned phase below it (freeing
    budget for one more GPU). >=1 GPU per phase. Returns the best dict, or None if the budget is too small."""
    best = None
    # A phase's capacity-per-watt is maximised at its sweet spot, so the BOTTLENECK phase never wants
    # more than ~W/sweet GPUs (more would have to run below sweet spot = strictly worse), and the
    # over-provisioned phase wants FEWER. So cap the counts at the sweet-spot packing (+2 slack).
    Np_max = min(int((W - P_DEC_LO) // P_PRE_LO), int(W / P_PRE_OPT) + 2)   # >=1 decode GPU must still fit
    # rank by throughput, then prefer LESS power for the SAME throughput (a phase can be over-provisioned
    # -> the cheaper config wins on energy with identical tokens). round() makes the tie tolerance explicit.
    def key(r):  return (round(r["tot"], 2), -round(r["w_used"], 1))
    for Np in range(1, Np_max + 1):
        Nd_max = min(int((W - Np * P_PRE_LO) // P_DEC_LO), int(W / P_DEC_OPT) + 2)
        for Nd in range(1, Nd_max + 1):
            r = _alloc(Np, Nd, P, D, W)
            if r and (best is None or key(r) > key(best)):
                best = r
    return best


def cont_bound(P, D, W=W_RACK):
    """Continuous (fractional-GPU) upper bound Tot* = W / (f_p/eta_p + f_d/eta_d) — the ceiling solve_opt
    approaches. Both phases at their efficiency sweet spot, capacities balanced P:D, full budget spent."""
    ep, ed = best_eff_prefill()[2], best_eff_decode()[2]
    fp, fd = P / (P + D), D / (P + D)
    return W / (fp / ep + fd / ed)


def policies():
    """Fixed-cap reference policies for solve(). TDP is the baseline; SWEET_SPOT is the old
    'every GPU at its efficiency sweet spot' recipe — kept only for comparison. The real maximum is
    solve_opt (floating per-phase caps), which beats SWEET_SPOT by burning the budget it left as surplus."""
    return {"SWEET_SPOT": (P_PRE_OPT, P_DEC_OPT), "OPTIMAL": (P_PRE_OPT, P_DEC_OPT), "TDP": (P_TDP, P_TDP)}


RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1), (2, 1), (5, 1), (10, 1), (20, 1)]


def main():
    pP, tP, eP = best_eff_prefill()
    pD, tD, eD = best_eff_decode()
    print(f"prefill fit:  P0={PRE['P0']:.0f} kappa={PRE['kappa']:.2e} rho={PRE['rho']:.1e}  R²={PRE['R2']:.3f}"
          + (f"  [hand-set:{','.join(PRE['manual'])}]" if PRE['manual'] else "  [auto]"))
    print(f"decode  fit:  P0={DEC['P0']:.0f} kappa={DEC['kappa']:.2e} rho={DEC['rho']:.1e} T_max={DEC['T_max']:.0f}  R²={DEC['R2']:.3f}"
          + (f"  [hand-set:{','.join(DEC['manual'])}]" if DEC['manual'] else "  [auto]"))
    print(f"sweet spots:  prefill {pP:.0f} W -> {tP:.0f} tok/s ({eP:.1f} tok/J) | "
          f"decode {pD:.0f} W -> {tD:.0f} tok/s ({eD:.1f} tok/J)   (decode knee = T_max={DEC['T_max']:.0f})")
    print(f"\nrack {W_RACK:.0f} W, context C={CTX}   integer GPUs, >=1 per phase   "
          f"prefill cap floats in [{P_PRE_LO:.0f},{P_PRE_HI:.0f}]W, decode in [{P_DEC_LO:.0f},{P_DEC_HI:.0f}]W")
    print("OPT = full-budget throughput max (per-phase caps float)   vs   TDP = every GPU @250 W")
    print(f"{'P:D':>6} |{'OPT k/s':>8}{'Np':>3}{'Nd':>3} {'pp':>4}{'pd':>4} {'used':>5} {'%Tot*':>6} |"
          f"{'TDP k/s':>8}{'Np':>3}{'Nd':>3} |{'gain':>7}  note")
    rows = []
    for (P, D) in RATIOS:
        o = solve_opt(P, D); t = solve(P, D, *policies()["TDP"])
        gain = 100 * (o["tot"] / t["tot"] - 1)
        ceil = cont_bound(P, D); pct = 100 * o["tot"] / ceil
        bind = "Np=1 floor" if o["Np"] == 1 and P < D else ("Nd=1 floor" if o["Nd"] == 1 and D < P else "")
        print(f"{P}:{D:<4} |{o['tot']/1000:>8.1f}{o['Np']:>3d}{o['Nd']:>3d} {o['p_p']:>4.0f}{o['p_d']:>4.0f}"
              f" {o['w_used']:>5.0f} {pct:>5.1f}% |{t['tot']/1000:>8.1f}{t['Np']:>3d}{t['Nd']:>3d} |"
              f" {gain:>5.1f}%  {bind}")
        rows.append({"context_C": CTX, "P": P, "D": D,
                     "opt_tok_s": round(o["tot"]), "opt_N_prefill": o["Np"], "opt_N_decode": o["Nd"],
                     "opt_p_prefill_w": round(o["p_p"]), "opt_p_decode_w": round(o["p_d"]),
                     "opt_w_used": round(o["w_used"]), "opt_pct_of_bound": round(pct, 1),
                     "tdp_tok_s": round(t["tot"]), "tdp_N_prefill": t["Np"], "tdp_N_decode": t["Nd"],
                     "gain_opt_vs_tdp_pct": round(gain, 1), "constraint_binds": bind})

    with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    print("\nwrote results.csv   (opt_p_*/opt_w_used now vary by ratio: bottleneck phase pushed up, "
          "over-provisioned phase trimmed; %Tot* = how close to the continuous ceiling)")


if __name__ == "__main__":
    main()
