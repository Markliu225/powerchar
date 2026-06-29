"""Rack-level power-capping optimizer — driven by the CURRENT measured-fitted curves.

PROBLEM. Requests arrive with a FIXED prefill:decode token ratio P:D. Given a rack power
budget W, decide: how many GPUs, each in which phase (prefill / decode), each at what
per-GPU power cap, to MAXIMISE total (balanced) token throughput.

PER-GPU CURVES come straight from ../pt_cap_gpu1/plot_theory.py — the models fitted to the latest
measured prefill.csv / decode.csv on this V100-GPU1 — so this solver ALWAYS matches that figure
(including any hand-set PRE_*/DEC_* parameters there):
    prefill (compute-bound):  T(P) = inverse of  P = P0 + kappa*T*(1+rho*T)^2     (V²f)
    decode  (memory-bound) :  T(P) = min( T_{V²f}(P), T_max )                     (piecewise roofline)
Each phase's efficiency (tok/J) sweet spot = argmax_P T(P)/P, found numerically below.

OPTIMUM. With each phase run at its efficiency eta=T/P and token fractions f_p=P/(P+D),
f_d=D/(P+D), the max balanced rack throughput at full budget is
        Tot* = W / (f_p/eta_p + f_d/eta_d),
so the optimum runs EACH PHASE AT ITS SWEET SPOT; the ratio only sets the GPU-count split. We
compare the optimum against the TDP baseline (every GPU at 250 W nameplate).

CONSTRAINT: integer GPUs, >=1 per phase (disaggregated serving needs >=1 prefill worker AND >=1
decode worker). At extreme ratios the >=1 floor binds (e.g. decode-heavy 1:20 still spends one
whole GPU on prefill). See solve().
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


def Tpre(p):  return float(np.maximum(_preT(np.asarray(p, float)), 0.0))   # prefill tok/s at cap p
def Tdec(p):  return float(np.maximum(_decT(np.asarray(p, float)), 0.0))   # decode  tok/s at cap p


# ---- scenario knobs ----
W_RACK = 5000.0                               # rack power budget (W); results scale linearly
P_TDP = 250.0                                 # nameplate TDP per GPU (W)
CTX = 256                                     # context the decode curve was measured at (label only)
P_PRE_LO, P_DEC_LO = float(_Pp.min()), float(_Pf.min())


def best_eff(Tfun, lo, hi=P_TDP):
    """Efficiency (tok/J) sweet spot of a curve: argmax_P T(P)/P over [lo, hi]."""
    ps = np.linspace(lo, hi, 4000)
    eff = np.array([Tfun(p) / p for p in ps]); i = int(np.argmax(eff))
    return float(ps[i]), float(Tfun(ps[i])), float(eff[i])


def best_eff_prefill():  return best_eff(Tpre, P_PRE_LO)
def best_eff_decode():   return best_eff(Tdec, P_DEC_LO)

P_PRE_OPT = best_eff_prefill()[0]             # prefill efficiency sweet-spot power (W)
P_DEC_OPT = best_eff_decode()[0]              # decode  efficiency sweet-spot power (W)


def solve(P, D, p_p, p_d, W=W_RACK):
    """Token ratio P:D, per-GPU caps p_p/p_d -> INTEGER allocation with >=1 GPU in EACH phase.

    Whole GPUs, Np>=1 and Nd>=1. The balanced workload is served at request rate
    lam = min(prefill_capacity/P, decode_capacity/D); useful throughput = lam*(P+D). Sweep integer
    Np>=1, fill the rest of the budget with Nd>=1 decode GPUs, keep the best. None if budget < 1+1."""
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
                        Wp=npre * p_p, Wd=ndec * p_d, pre_cap=npre * tp, dec_cap=ndec * td)
    return best


def policies():
    return {"OPTIMAL": (P_PRE_OPT, P_DEC_OPT), "TDP": (P_TDP, P_TDP)}


RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1), (2, 1), (5, 1), (10, 1), (20, 1)]


def main():
    pP, tP, eP = best_eff_prefill()
    pD, tD, eD = best_eff_decode()
    print(f"prefill fit:  P0={PRE['P0']:.0f} kappa={PRE['kappa']:.2e} rho={PRE['rho']:.1e}  R²={PRE['R2']:.3f}"
          + (f"  [hand-set:{','.join(PRE['manual'])}]" if PRE['manual'] else "  [auto]"))
    print(f"decode  fit:  P0={DEC['P0']:.0f} kappa={DEC['kappa']:.2e} rho={DEC['rho']:.1e} T_max={DEC['T_max']:.0f}  R²={DEC['R2']:.3f}"
          + (f"  [hand-set:{','.join(DEC['manual'])}]" if DEC['manual'] else "  [auto]"))
    print(f"sweet spots:  prefill {pP:.0f} W -> {tP:.0f} tok/s ({eP:.1f} tok/J) | "
          f"decode {pD:.0f} W -> {tD:.0f} tok/s ({eD:.1f} tok/J)   (decode capped at T_max={DEC['T_max']:.0f})")
    print(f"\nrack {W_RACK:.0f} W, context C={CTX}   integer GPUs, >=1 per phase   "
          f"(OPTIMAL pre@{pP:.0f}/dec@{pD:.0f}W  vs  TDP {P_TDP:.0f}/{P_TDP:.0f}W)")
    print(f"{'P:D':>7} | {'OPT ktok/s':>10}{'Np':>4}{'Nd':>4} | {'TDP ktok/s':>10}{'Np':>4}{'Nd':>4} | {'gain':>7}  note")
    rows = []
    for (P, D) in RATIOS:
        o = solve(P, D, *policies()["OPTIMAL"]); t = solve(P, D, *policies()["TDP"])
        gain = 100 * (o["tot"] / t["tot"] - 1)
        bind = "Np=1 floor" if o["Np"] == 1 and P < D else ("Nd=1 floor" if o["Nd"] == 1 and D < P else "")
        print(f"{P}:{D:<5} | {o['tot']/1000:>10.1f}{o['Np']:>4d}{o['Nd']:>4d} | "
              f"{t['tot']/1000:>10.1f}{t['Np']:>4d}{t['Nd']:>4d} | {gain:>6.1f}%  {bind}")
        rows.append({"context_C": CTX, "P": P, "D": D,
                     "opt_tok_s": round(o["tot"]), "opt_N_prefill": o["Np"], "opt_N_decode": o["Nd"],
                     "opt_p_prefill_w": round(pP), "opt_p_decode_w": round(pD),
                     "tdp_tok_s": round(t["tot"]), "tdp_N_prefill": t["Np"], "tdp_N_decode": t["Nd"],
                     "gain_opt_vs_tdp_pct": round(gain, 1), "constraint_binds": bind})

    with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    print("\nwrote results.csv")


if __name__ == "__main__":
    main()
