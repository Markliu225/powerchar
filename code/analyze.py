"""Figures + analytic model: GPU power as a SINGLE-VALUED function of throughput.

Controlled experiment (both phases): hold the sequence/context length FIXED and
sweep only the batch (concurrency). With the per-token cost fixed, throughput is
a MONOTONE function of the one swept variable, so P(T) is single-valued -- each
throughput maps to one power. (Sweeping seq_len instead makes throughput
non-monotone -> a folded, multi-valued P(T); that is the mistake this fixes.)

Model: P = P_static + (P_cap - P_static)*u  and  T rises toward a ceiling T_max
as concurrency fills the chip. Eliminating batch gives a saturating P(T). The
ceiling differs by phase: prefill is capped by the COMPUTE roof (Φ·MFU / cost),
decode by the MEMORY/overhead limit (1/β) -- ~7-14x lower, the whole story.

  python analyze.py --step 1|2|3|4|all
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

FIG = C.FIGURES_DIR


def load_csv(name):
    with open(os.path.join(C.RESULTS_DIR, name)) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (ValueError, TypeError):
                d[k] = v
        out.append(d)
    return out


def load_info():
    with open(os.path.join(C.RESULTS_DIR, "model_info.json")) as f:
        return json.load(f)


def col(rows, k):
    return np.array([r[k] for r in rows], float)


def mape(meas, pred):
    meas, pred = np.asarray(meas, float), np.asarray(pred, float)
    return float(np.mean(np.abs((pred - meas) / meas)) * 100)


def r2(meas, pred):
    meas, pred = np.asarray(meas, float), np.asarray(pred, float)
    ss_res = np.sum((meas - pred) ** 2)
    ss_tot = np.sum((meas - meas.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def fit_sat_B(B, P):
    """P(B) = P_idle + A*(1 - exp(-B/B0)); grid B0, linear-LS (P_idle, A)."""
    best = None
    for B0 in np.linspace(B.max() * 0.02, B.max() * 2.0, 400):
        basis = np.c_[np.ones_like(B), 1 - np.exp(-B / B0)]
        coef, *_ = np.linalg.lstsq(basis, P, rcond=None)
        resid = np.sum((basis @ coef - P) ** 2)
        if best is None or resid < best[0]:
            best = (resid, B0, coef[0], coef[1])
    _, B0, P_idle, A = best
    return P_idle, A, B0


def fit_affine_time(B, tokens_per_batch, T):
    """Forward/step time is affine in batch: t(B) = t_fixed + beta*B.
    t = (tokens_per_batch * B) / T.  Throughput ceiling = tokens_per_batch/beta."""
    t = tokens_per_batch * B / T
    (t_fixed, beta), *_ = np.linalg.lstsq(np.c_[np.ones_like(B), B], t, rcond=None)
    return t_fixed, beta, tokens_per_batch / beta


def _compose_pT(Bg, per_tok, t_fixed, beta, P_idle, A, B0):
    """Parametric P(T) over a batch grid from the two measured laws."""
    T = per_tok * Bg / (t_fixed + beta * Bg)
    P = P_idle + A * (1 - np.exp(-Bg / B0))
    return T, P


def decode_clean(rows):
    """Clean memory-bound regime (up to throughput peak) vs post-peak VRAM spill."""
    b = np.array([r["batch"] for r in rows]); t = np.array([r["throughput_tok_s"] for r in rows])
    bp = b[int(np.argmax(t))]
    return [r for r in rows if r["batch"] <= bp], [r for r in rows if r["batch"] > bp], bp


def _pT_panel(ax, T, P, B, cap, cmap, ceil_T=None, ceil_label=None, wallT=None, wallP=None):
    ax.plot(T, P, "-", color="gray", alpha=.4, zorder=1)
    sc = ax.scatter(T, P, c=np.log2(B), cmap=cmap, s=90, zorder=5)
    for t, p, b in zip(T, P, B):
        ax.annotate(f"b{int(b)}", (t, p), fontsize=7, xytext=(4, -3), textcoords="offset points")
    if wallT is not None and len(wallT):
        ax.scatter(wallT, wallP, facecolors="none", edgecolors="gray", s=80, zorder=4,
                   label="VRAM-spill wall (excluded)")
    if ceil_T is not None:
        ax.axvline(ceil_T, color="C2", ls=":", lw=2, label=ceil_label)
    ax.axhline(cap, color="r", ls="--", alpha=.5, label=f"cap {cap:.0f} W")
    return sc


# ---------------------------------------------------------------------------
# Step 1 -- PREFILL: power & efficiency vs throughput (controlled batch sweep)
# ---------------------------------------------------------------------------
def step1():
    rows = sorted(load_csv("prefill.csv"), key=lambda r: r["batch"])
    info = load_info()
    cap = info["power_cap_w"]
    S = rows[0]["seq_len"]
    T = col(rows, "throughput_tok_s") / 1e3; P = col(rows, "power_avg_w")
    B = col(rows, "batch"); tokJ = col(rows, "tok_per_joule")
    roof = info["peak_matmul_flops"] / (info["dense_flops_per_token"]
                                        + info["attn_flops_per_token_per_S"] * S) / 1e3

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = _pT_panel(ax, T, P, B, cap, "viridis", roof, f"compute roof {roof:.1f}k tok/s")
    fig.colorbar(sc, label="log2(batch)")
    ax.set_xlabel("token throughput (k tok/s)"); ax.set_ylabel("GPU power (W)")
    ax.set_xlim(0, max(roof, T.max()) * 1.08); ax.set_ylim(60, cap * 1.04)
    ax.set_title(f"PREFILL: GPU power vs throughput (fixed S={int(S)}, sweep batch)\n"
                 "single-valued: power rises with throughput toward the compute roof")
    ax.legend(loc="lower right"); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step1_prefill_power_vs_throughput.png", dpi=130)
    print(f"wrote {FIG}/step1_prefill_power_vs_throughput.png")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(T, tokJ, "o-", color="C2");
    for t, e, b in zip(T, tokJ, B):
        ax.annotate(f"b{int(b)}", (t, e), fontsize=7, xytext=(3, -8), textcoords="offset points")
    ax.set_xlabel("token throughput (k tok/s)"); ax.set_ylabel("energy efficiency (tok/J)")
    ax.set_title(f"PREFILL energy efficiency vs throughput (fixed S={int(S)})")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step1_prefill_efficiency_vs_throughput.png", dpi=130)
    print(f"wrote {FIG}/step1_prefill_efficiency_vs_throughput.png")
    print(f"\n[prefill] fixed S={int(S)}, batch {int(B.min())}-{int(B.max())}: "
          f"power {P.min():.0f}→{P.max():.0f} W rises WITH throughput "
          f"{T.min()*1e3:.0f}→{T.max()*1e3:.0f} tok/s (single-valued); roof {roof:.1f}k; best {tokJ.max():.0f} tok/J")


# ---------------------------------------------------------------------------
# Step 2 -- DECODE: power & efficiency vs throughput (controlled batch sweep)
# ---------------------------------------------------------------------------
def step2():
    allrows = sorted(load_csv("decode.csv"), key=lambda r: r["batch"])
    clean, wall, bp = decode_clean(allrows)
    info = load_info(); cap = info["power_cap_w"]
    T = col(clean, "throughput_tok_s") / 1e3; P = col(clean, "power_avg_w")
    B = col(clean, "batch"); tokJ = col(clean, "tok_per_joule")
    wallT = col(wall, "throughput_tok_s") / 1e3 if wall else np.array([])
    wallP = col(wall, "power_avg_w") if wall else np.array([])

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = _pT_panel(ax, T, P, B, cap, "plasma", wallT=wallT, wallP=wallP)
    fig.colorbar(sc, label="log2(batch)")
    ax.set_xlabel("token throughput (k tok/s)"); ax.set_ylabel("GPU power (W)")
    ax.set_title(f"DECODE: GPU power vs throughput (fixed ctx={int(clean[0]['ctx_len'])}, sweep batch)\n"
                 "single-valued: power rises with throughput toward the memory ceiling")
    ax.legend(loc="lower right"); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step2_decode_power_vs_throughput.png", dpi=130)
    print(f"wrote {FIG}/step2_decode_power_vs_throughput.png")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(T, tokJ, "o-", color="C2")
    for t, e, b in zip(T, tokJ, B):
        ax.annotate(f"b{int(b)}", (t, e), fontsize=7, xytext=(3, -8), textcoords="offset points")
    ax.set_xlabel("token throughput (k tok/s)"); ax.set_ylabel("energy efficiency (tok/J)")
    ax.set_title("DECODE energy efficiency vs throughput")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step2_decode_efficiency_vs_throughput.png", dpi=130)
    print(f"wrote {FIG}/step2_decode_efficiency_vs_throughput.png")
    print(f"\n[decode] fixed ctx, batch 1-{int(bp)}: power {P.min():.0f}→{P.max():.0f} W "
          f"rises WITH throughput {T.min()*1e3:.0f}→{T.max()*1e3:.0f} tok/s; best {tokJ.max():.1f} tok/J")


# ---------------------------------------------------------------------------
# Step 3 -- ANALYTIC MODEL: single-valued P(T), one per phase
# ---------------------------------------------------------------------------
def step3():
    info = load_info()
    peak, bw, cap = info["peak_matmul_flops"], info["peak_bw_bytes_s"], info["power_cap_w"]
    dense, attn = info["dense_flops_per_token"], info["attn_flops_per_token_per_S"]
    fit = {}

    # ---- PREFILL: compose P(B) saturation with affine forward-time T(B) ----
    pre = sorted(load_csv("prefill.csv"), key=lambda r: r["batch"])
    S = pre[0]["seq_len"]
    Bp = col(pre, "batch"); Tp = col(pre, "throughput_tok_s"); Pp = col(pre, "power_avg_w")
    c_tok = dense + attn * S
    roof_ideal = peak / c_tok                           # MFU=1 compute roof
    tf_p, beta_p, ceil_p = fit_affine_time(Bp, S, Tp)   # ceiling = S/beta
    Pi_p, A_p, B0_p = fit_sat_B(Bp, Pp)
    predP_p = Pi_p + A_p * (1 - np.exp(-Bp / B0_p))
    Tc_p, Pc_p = _compose_pT(Bp, S, tf_p, beta_p, Pi_p, A_p, B0_p)
    fit["prefill"] = {
        "experiment": f"fixed S={int(S)}, sweep batch -> single-valued P(T)",
        "throughput_ceiling_tok_s": ceil_p, "compute_roof_ideal_tok_s": roof_ideal,
        "achieved_mfu": float(np.max(Tp) * c_tok / peak),
        "P_measured_w": [float(Pp.min()), float(Pp.max())],
        "P_idle_w_EXTRAPOLATED": Pi_p, "P_asymptote_w": Pi_p + A_p, "B0": B0_p,
        "t_fixed_ms": tf_p * 1e3, "beta_ms_per_batch": beta_p * 1e3,
        "P_vs_T_mape_pct": mape(Pp, predP_p), "P_vs_T_r2": r2(Pp, predP_p),
        "T_vs_B_r2": r2(Tp, Tc_p),
    }
    # ---- DECODE: compose P(B) saturation with affine step-time T(B) ----
    dec, decwall, bp = decode_clean(sorted(load_csv("decode.csv"), key=lambda r: r["batch"]))
    Bd = col(dec, "batch"); Td = col(dec, "throughput_tok_s"); Pd = col(dec, "power_avg_w")
    tf_d, beta_d, ceil_d = fit_affine_time(Bd, 1.0, Td)  # ceiling = 1/beta
    Pi_d, A_d, B0_d = fit_sat_B(Bd, Pd)
    predP_d = Pi_d + A_d * (1 - np.exp(-Bd / B0_d))
    Tc_d, Pc_d = _compose_pT(Bd, 1.0, tf_d, beta_d, Pi_d, A_d, B0_d)
    fit["decode"] = {
        "experiment": f"fixed ctx={int(dec[0]['ctx_len'])}, sweep batch -> single-valued P(T)",
        "throughput_ceiling_tok_s": ceil_d,
        "P_measured_w": [float(Pd.min()), float(Pd.max())],
        "P_idle_w": Pi_d, "P_asymptote_w": Pi_d + A_d, "B0": B0_d,
        "t_fixed_ms": tf_d * 1e3, "beta_ms_per_seq": beta_d * 1e3,
        "P_vs_T_mape_pct": mape(Pd, predP_d), "P_vs_T_r2": r2(Pd, predP_d),
        "T_vs_B_r2": r2(Td, Tc_d),
    }

    for name, (per, Bm, Tm, Pm, tf, beta, Pi, A, B0, ceil, ceil_lab, color) in {
        "prefill": (S, Bp, Tp, Pp, tf_p, beta_p, Pi_p, A_p, B0_p, roof_ideal,
                    f"compute roof {roof_ideal/1e3:.1f}k tok/s", "C1"),
        "decode": (1.0, Bd, Td, Pd, tf_d, beta_d, Pi_d, A_d, B0_d, ceil_d,
                   f"memory ceiling {ceil_d/1e3:.1f}k tok/s", "C0"),
    }.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        Bg = np.linspace(Bm.min() * 0.92, Bm.max() * 1.5, 400)   # measured range only
        Tg, Pg = _compose_pT(Bg, per, tf, beta, Pi, A, B0)
        ax.plot(Tg / 1e3, Pg, "k-", lw=2,
                label="analytic P(T): compose P(B)=P₀+A(1−e^{−B/B₀}) & T(B)=B·tok/(t_f+βB)")
        ax.scatter(Tm / 1e3, Pm, color=color, s=85, zorder=5, label="measured")
        ax.axvline(ceil / 1e3, color="C2", ls=":", lw=2, label=ceil_lab)
        ax.axhline(cap, color="r", ls="--", alpha=.5, label=f"cap {cap:.0f} W")
        f = fit[name]
        ax.set_xlabel("token throughput (k tok/s)"); ax.set_ylabel("GPU power (W)")
        ax.set_ylim(min(Pm) - 12, cap * 1.04); ax.set_xlim(0, max(ceil, Tm.max()) / 1e3 * 1.05)
        ax.set_title(f"{name.upper()} power↔throughput model (single-valued, batch sweep)\n"
                     f"P {Pm.min():.0f}→{Pm.max():.0f} W (asymptote {f['P_asymptote_w']:.0f} W), "
                     f"ceiling {f['throughput_ceiling_tok_s']/1e3:.1f}k tok/s, "
                     f"MAPE {f['P_vs_T_mape_pct']:.1f}%, R²={f['P_vs_T_r2']:.3f}")
        ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(f"{FIG}/step3_{name}_model.png", dpi=130)
        print(f"wrote {FIG}/step3_{name}_model.png")

    with open(os.path.join(C.RESULTS_DIR, "fit_summary.json"), "w") as f:
        json.dump(fit, f, indent=2)
    print(f"wrote {C.RESULTS_DIR}/fit_summary.json")
    print(f"\n[fit] prefill P(T): P {Pp.min():.0f}→{Pp.max():.0f} W (asymptote {Pi_p+A_p:.0f}), "
          f"ceiling {ceil_p/1e3:.1f}k tok/s (roof {roof_ideal/1e3:.1f}k, MFU {fit['prefill']['achieved_mfu']*100:.0f}%), "
          f"MAPE {fit['prefill']['P_vs_T_mape_pct']:.1f}%, R²={fit['prefill']['P_vs_T_r2']:.3f}")
    print(f"[fit] decode  P(T): P {Pi_d:.0f}→{Pi_d+A_d:.0f} W, ceiling {ceil_d/1e3:.1f}k tok/s, "
          f"MAPE {fit['decode']['P_vs_T_mape_pct']:.1f}%, R²={fit['decode']['P_vs_T_r2']:.3f}")
    print(f"[contrast] prefill ceiling {ceil_p/1e3:.1f}k vs decode {ceil_d/1e3:.1f}k tok/s "
          f"= {ceil_p/ceil_d:.0f}× more throughput at the same ~cap power")


# ---------------------------------------------------------------------------
# Step 4 -- SYNTHESIS: both phases on one throughput axis
# ---------------------------------------------------------------------------
def step4():
    info = load_info(); cap = info["power_cap_w"]
    pre = sorted(load_csv("prefill.csv"), key=lambda r: r["throughput_tok_s"])
    dec, _, _ = decode_clean(sorted(load_csv("decode.csv"), key=lambda r: r["batch"]))
    dec = sorted(dec, key=lambda r: r["throughput_tok_s"])

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(col(dec, "throughput_tok_s") / 1e3, col(dec, "power_avg_w"), "o-",
            color="C0", label="decode (memory-bound)")
    ax.plot(col(pre, "throughput_tok_s") / 1e3, col(pre, "power_avg_w"), "s-",
            color="C1", label="prefill (compute-bound)")
    ax.axhline(cap, color="r", ls="--", alpha=.5, label=f"cap {cap:.0f} W")
    ax.set_xscale("log")
    ratio = np.max(col(pre, "throughput_tok_s")) / np.max(col(dec, "throughput_tok_s"))
    ax.set_xlabel("token throughput (k tok/s)"); ax.set_ylabel("GPU power (W)")
    ax.set_title(f"Power vs token throughput — both rise with concurrency,\n"
                 f"but prefill reaches ~{ratio:.0f}× the throughput at the same power")
    ax.legend(loc="upper left"); ax.grid(alpha=.3, which="both"); fig.tight_layout()
    fig.savefig(f"{FIG}/step4_combined_power_vs_throughput.png", dpi=130)
    print(f"wrote {FIG}/step4_combined_power_vs_throughput.png")

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(col(dec, "throughput_tok_s") / 1e3, col(dec, "tok_per_joule"), "o-", color="C0", label="decode")
    ax.plot(col(pre, "throughput_tok_s") / 1e3, col(pre, "tok_per_joule"), "s-", color="C1", label="prefill")
    ax.set_xscale("log")
    pj = np.nanmax(col(pre, "tok_per_joule")); dj = np.nanmax(col(dec, "tok_per_joule"))
    ax.set_xlabel("token throughput (k tok/s)"); ax.set_ylabel("energy efficiency (tok/J)")
    ax.set_title(f"Energy efficiency vs throughput (prefill ~{pj/dj:.0f}× decode at best)")
    ax.legend(); ax.grid(alpha=.3, which="both"); fig.tight_layout()
    fig.savefig(f"{FIG}/step4_combined_efficiency_vs_throughput.png", dpi=130)
    print(f"wrote {FIG}/step4_combined_efficiency_vs_throughput.png")
    print(f"\n[synthesis] both couple P↔T; prefill reaches {ratio:.0f}× decode's throughput at "
          f"similar power. best tok/J prefill {pj:.0f} vs decode {dj:.1f} ({pj/dj:.0f}×)")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--step", default="all")
    args = ap.parse_args()
    steps = {"1": [step1], "2": [step2], "3": [step3], "4": [step4],
             "all": [step1, step2, step3, step4]}[args.step]
    for fn in steps:
        fn()


if __name__ == "__main__":
    main()
