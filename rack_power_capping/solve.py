"""Rack-level power-capping optimizer (Phi-3-mini / Tesla V100 curves).

PROBLEM. Requests arrive with a FIXED prefill:decode token ratio P:D. Given a rack power
budget W, decide: how many GPUs, each in which phase (prefill / decode), each at what
per-GPU power cap, to MAXIMISE total token throughput.

PER-GPU CURVES
  prefill (compute-bound, measured): piecewise-linear over the clean V100 frequency-sweep
      points; efficiency tok/J peaks at ~164 W (well below the 250 W TDP). The 250 W point
      is a linear extrapolation so the TDP baseline can run prefill at nameplate (this card
      thermally caps ~236 W in practice).
  decode (memory-bound, FIRST-PRINCIPLES bandwidth roofline -- NOT the measured affine curve,
      which was launch-overhead limited and never reached the bandwidth ceiling):
          T_dec(B,C) = beta*B/(W_bytes + B*C*kv) = T_max(C) * B/(B+B*(C))
          T_max(C) = beta/(C*kv)     <- HBM-bandwidth ceiling: throughput rises with power
                                         (concurrency) then goes FLAT here.
          B*(C)    = W_bytes/(C*kv)  <- half-saturation batch.
      Power frontier:  T_dec(P) = T_max(C)*(P-P0)/((P-P0)+K),  P0=90 W, K=70 W.
      Efficiency optimum:  P_opt = P0 + sqrt(P0*K) = 169 W, INDEPENDENT of context C
          (eta=T/P peaks where (P-P0)^2 = P0*K; T_max's 1/C factor cancels in d eta/dP).
      Context C only scales achievable throughput/efficiency (~1/C), not the optimal power.

OPTIMUM. With each phase at per-GPU efficiency eta=T/P (tok/J) and token fractions
f_p=P/(P+D), f_d=D/(P+D), the max rack throughput at full budget is
      Tot* = W / (f_p/eta_p + f_d/eta_d),
so the optimum runs EACH PHASE AT ITS EFFICIENCY SWEET-SPOT (prefill 164 W, decode 169 W);
the ratio only sets the GPU-count split. Both sweet-spots sit ~165-170 W, far below the
250 W TDP -> under a power cap you deploy MORE GPUs at lower power. We compare the optimum
against the TDP baseline (every GPU at 250 W nameplate) and a uniform-170 W cap.

  beta=6.26e11 B/s effective HBM (0.8x782 GB/s peak), W_bytes=7.642e9, kv=393216 B/token.
"""
from __future__ import annotations
import csv, os
import numpy as np

# ---- prefill: measured clean points (+ 250 W nameplate extrapolation) ----
PRE_P = np.array([103.63, 119.36, 139.76, 163.70, 202.97, 235.63, 250.0])
PRE_T = np.array([3361.9, 4114.2, 5120.6, 6115.4, 6879.0, 7334.2, 7534.5])
P_PRE_LO, P_PRE_HI = float(PRE_P[0]), float(PRE_P[-1])

# ---- decode: first-principles bandwidth-roofline model ----
BETA = 6.26e11          # effective HBM bandwidth (B/s)
KV = 393216.0           # KV bytes per token = 2*L*n_kv*head_dim*b
W_BYTES = 7.642e9       # weight bytes read per decode step
P0, K = 90.0, 70.0      # static floor / power-scale (W)
P_DEC_OPT = P0 + (P0 * K) ** 0.5            # 169.4 W, context-independent

# ---- scenario knobs ----
W_RACK = 5000.0         # rack power budget (W); results scale linearly
CTX = 1024              # decode context length (KV tokens); T_max ∝ 1/CTX


def Tmax(C):     return BETA / (C * KV)                       # bandwidth ceiling (tok/s)
def Bstar(C):    return W_BYTES / (C * KV)                    # half-saturation batch
def Tpre(p):     return float(np.interp(p, PRE_P, PRE_T))     # prefill (measured interp)
def Tdec(p, C):  x = max(p - P0, 0.0); return Tmax(C) * x / (x + K)   # decode frontier T(P)


def best_eff_prefill():
    ps = np.linspace(P_PRE_LO, P_PRE_HI, 2000)
    eff = np.array([Tpre(p) / p for p in ps]); i = int(np.argmax(eff))
    return float(ps[i]), Tpre(ps[i]), float(eff[i])


def solve(P, D, p_p, p_d, C=CTX, W=W_RACK):
    """Token ratio P:D, per-GPU powers (p_p prefill, p_d decode) -> full-budget allocation."""
    tp, td = Tpre(p_p), Tdec(p_d, C)
    eta_p, eta_d = tp / p_p, td / p_d
    fp, fd = P / (P + D), D / (P + D)
    tot = W / (fp / eta_p + fd / eta_d)          # max total tok/s at full budget
    A_pre, B_dec = fp * tot, fd * tot            # phase throughputs (tok/s)
    Np, Nd = A_pre / tp, B_dec / td              # GPU counts (continuous)
    return dict(p_p=p_p, p_d=p_d, tp=tp, td=td, eta_p=eta_p, eta_d=eta_d,
                tot=tot, Np=Np, Nd=Nd, Wp=Np * p_p, Wd=Nd * p_d)


# policies: name -> (prefill per-GPU W, decode per-GPU W)
def policies():
    pP, _, _ = best_eff_prefill()
    return {"OPTIMAL": (pP, P_DEC_OPT), "TDP": (250.0, 250.0), "Uniform170": (170.0, 170.0)}


RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1), (2, 1), (5, 1), (10, 1), (20, 1)]


def main():
    pP, tP, eP = best_eff_prefill()
    pol = policies()
    print(f"per-GPU sweet spots:  prefill {pP:.0f} W -> {tP:.0f} tok/s ({eP:.1f} tok/J) | "
          f"decode {P_DEC_OPT:.0f} W (C-independent)")
    print(f"decode bandwidth ceiling T_max(C={CTX}) = {Tmax(CTX):.0f} tok/s, "
          f"@169W -> {Tdec(P_DEC_OPT, CTX):.0f} tok/s ({Tdec(P_DEC_OPT, CTX)/P_DEC_OPT:.1f} tok/J)")
    print(f"\nrack {W_RACK:.0f} W, context C={CTX}   (OPTIMAL pre@{pP:.0f}/dec@{P_DEC_OPT:.0f}W  vs  TDP 250/250W)")
    print(f"{'P:D':>7} | {'OPT ktok/s':>10}{'N_pre':>7}{'N_dec':>7} | {'TDP ktok/s':>10}{'N_pre':>7}{'N_dec':>7} | {'gain':>7}")
    rows = []
    for (P, D) in RATIOS:
        o = solve(P, D, *pol["OPTIMAL"]); t = solve(P, D, *pol["TDP"]); u = solve(P, D, *pol["Uniform170"])
        gain = 100 * (o["tot"] / t["tot"] - 1)
        print(f"{P}:{D:<5} | {o['tot']/1000:>10.1f}{o['Np']:>7.1f}{o['Nd']:>7.1f} | "
              f"{t['tot']/1000:>10.1f}{t['Np']:>7.1f}{t['Nd']:>7.1f} | {gain:>6.1f}%")
        rows.append({"context_C": CTX, "P": P, "D": D,
                     "opt_tok_s": round(o["tot"]), "opt_N_prefill": round(o["Np"], 2), "opt_N_decode": round(o["Nd"], 2),
                     "opt_p_prefill_w": round(pP), "opt_p_decode_w": round(P_DEC_OPT),
                     "tdp_tok_s": round(t["tot"]), "tdp_N_prefill": round(t["Np"], 2), "tdp_N_decode": round(t["Nd"], 2),
                     "uniform170_tok_s": round(u["tot"]),
                     "gain_opt_vs_tdp_pct": round(gain, 1)})

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    print("\nwrote results.csv")


if __name__ == "__main__":
    main()
