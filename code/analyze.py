"""Figures + analytic-model fits, framed as POWER vs TOKEN THROUGHPUT.

The deliverable question is "how do token throughput and GPU power relate?", so
the x-axis is throughput (tokens/s) everywhere and the y-axis is power (W) or
energy efficiency (tok/J). The analytic model is built around one principle:

    P = P_static + (P_cap - P_static) * u          (power tracks utilisation u)
    T = R / c                                       (throughput = rate / cost)

where R is the bottleneck-resource rate (FLOP/s when compute-bound, byte/s when
memory-bound) and c is the per-token cost. Decode raises R with batch, so power
and throughput rise TOGETHER (coupled). Prefill runs R pinned at the compute
roof, so power is fixed while throughput varies through c -- DECOUPLED.

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


def fit_saturation(x, y):
    """y = y0 + A*(1 - exp(-x/x0)); grid x0, linear-LS (y0, A). No scipy."""
    best = None
    for x0 in np.logspace(0, np.log10(max(x) * 1.5), 300):
        basis = np.c_[np.ones_like(x), 1 - np.exp(-x / x0)]
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        resid = np.sum((basis @ coef - y) ** 2)
        if best is None or resid < best[0]:
            best = (resid, x0, coef[0], coef[1])
    _, x0, y0, A = best
    return y0, A, x0


def _decode_clean(rows):
    """Clean memory-bound regime (up to the throughput peak) vs the post-peak
    VRAM-spill 'wall' points."""
    b = np.array([r["batch"] for r in rows])
    t = np.array([r["throughput_tok_s"] for r in rows])
    b_peak = b[int(np.argmax(t))]
    return ([r for r in rows if r["batch"] <= b_peak],
            [r for r in rows if r["batch"] > b_peak], b_peak)


# ---------------------------------------------------------------------------
# Step 1 -- PREFILL: power & efficiency vs throughput
# ---------------------------------------------------------------------------
def step1():
    rows = sorted(load_csv("prefill.csv"), key=lambda r: r["seq_len"])
    info = load_info()
    cap = info["power_cap_w"]
    T = col(rows, "throughput_tok_s") / 1e3        # k tok/s
    P = col(rows, "power_avg_w")
    S = col(rows, "seq_len")
    tokJ = col(rows, "tok_per_joule")
    sat = S >= 256                                 # compute-saturated points
    P_sat = float(np.median(P[sat]))

    # (a) POWER vs THROUGHPUT
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(T, P, c=np.log2(S), cmap="viridis", s=90, zorder=5)
    # trace the trajectory of increasing S (occupancy ramp -> fold back at cap)
    ax.plot(T, P, "-", color="gray", alpha=.35, zorder=1)
    for r in rows:
        ax.annotate(f"{int(r['seq_len'])}", (r["throughput_tok_s"] / 1e3, r["power_avg_w"]),
                    fontsize=7, xytext=(4, -3), textcoords="offset points")
    ax.axhline(P_sat, color="C1", ls="-", alpha=.5, label=f"compute-bound power ≈ {P_sat:.0f} W")
    ax.axhline(cap, color="r", ls="--", alpha=.5, label=f"cap {cap:.0f} W")
    ax.annotate("occupancy ramp\n(tiny prompts)", (2.2, 80), fontsize=8, color="gray")
    ax.annotate("compute-bound: power flat,\nthroughput set by attention",
                (5.2, 150), fontsize=8, color="C1")
    cb = fig.colorbar(sc, label="seq_len S"); cb.set_ticks(np.log2([64, 256, 1024, 4096]))
    cb.set_ticklabels([64, 256, 1024, 4096])
    ax.set_xlabel("token throughput (k tok/s)"); ax.set_ylabel("GPU power (W)")
    ax.set_ylim(50, cap * 1.05)
    ax.set_title("PREFILL: GPU power vs token throughput\n"
                 "power saturates at the cap; throughput is decoupled from it")
    ax.legend(loc="lower right"); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step1_prefill_power_vs_throughput.png", dpi=130)
    print(f"wrote {FIG}/step1_prefill_power_vs_throughput.png")

    # (b) EFFICIENCY vs THROUGHPUT
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(T, tokJ, c=np.log2(S), cmap="viridis", s=80, zorder=5)
    ax.plot(T, tokJ, "-", color="gray", alpha=.35)
    ax.set_xlabel("token throughput (k tok/s)"); ax.set_ylabel("energy efficiency (tok/J)")
    ax.set_title("PREFILL energy efficiency vs throughput")
    fig.colorbar(sc, label="log2(seq_len)")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step1_prefill_efficiency_vs_throughput.png", dpi=130)
    print(f"wrote {FIG}/step1_prefill_efficiency_vs_throughput.png")

    cv = float(np.std(P[sat]) / np.mean(P[sat]))
    print(f"\n[prefill] compute-bound power {P_sat:.0f} W (CV {cv*100:.1f}% across "
          f"S>=256) while throughput spans {T[sat].min():.1f}-{T[sat].max():.1f}k tok/s "
          f"-> power DECOUPLED from throughput; best {tokJ.max():.0f} tok/J")


# ---------------------------------------------------------------------------
# Step 2 -- DECODE: power & efficiency vs throughput
# ---------------------------------------------------------------------------
def step2():
    allrows = sorted(load_csv("decode.csv"), key=lambda r: r["batch"])
    clean, wall, b_peak = _decode_clean(allrows)
    info = load_info()
    cap = info["power_cap_w"]
    T = col(clean, "throughput_tok_s") / 1e3
    P = col(clean, "power_avg_w")
    B = col(clean, "batch")
    tokJ = col(clean, "tok_per_joule")

    # (a) POWER vs THROUGHPUT
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(T, P, "-", color="gray", alpha=.4, zorder=1)
    sc = ax.scatter(T, P, c=np.log2(B), cmap="plasma", s=90, zorder=5)
    if wall:
        ax.scatter(col(wall, "throughput_tok_s") / 1e3, col(wall, "power_avg_w"),
                   facecolors="none", edgecolors="gray", s=80, zorder=4,
                   label=f"VRAM-spill wall (b>{b_peak:.0f})")
    for r in clean:
        ax.annotate(f"b{int(r['batch'])}", (r["throughput_tok_s"] / 1e3, r["power_avg_w"]),
                    fontsize=7, xytext=(4, -2), textcoords="offset points")
    ax.axhline(cap, color="r", ls="--", alpha=.5, label=f"cap {cap:.0f} W")
    cb = fig.colorbar(sc, label="batch B"); cb.set_ticks(np.log2([1, 4, 16, 48]))
    cb.set_ticklabels([1, 4, 16, 48])
    ax.set_xlabel("token throughput (k tok/s)"); ax.set_ylabel("GPU power (W)")
    ax.set_title("DECODE: GPU power vs token throughput\n"
                 "power and throughput rise TOGETHER with batch (coupled)")
    ax.legend(loc="lower right"); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step2_decode_power_vs_throughput.png", dpi=130)
    print(f"wrote {FIG}/step2_decode_power_vs_throughput.png")

    # (b) EFFICIENCY vs THROUGHPUT
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(T, tokJ, c=np.log2(B), cmap="plasma", s=80, zorder=5)
    ax.plot(T, tokJ, "-", color="gray", alpha=.35)
    ax.set_xlabel("token throughput (k tok/s)"); ax.set_ylabel("energy efficiency (tok/J)")
    ax.set_title("DECODE energy efficiency vs throughput\n(batching up is the only lever)")
    fig.colorbar(sc, label="log2(batch)")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step2_decode_efficiency_vs_throughput.png", dpi=130)
    print(f"wrote {FIG}/step2_decode_efficiency_vs_throughput.png")

    print(f"\n[decode] power {P.min():.0f}->{P.max():.0f} W rises WITH throughput "
          f"{T.min()*1e3:.0f}->{T.max()*1e3:.0f} tok/s (coupled); best {tokJ.max():.1f} tok/J")


# ---------------------------------------------------------------------------
# Step 3 -- ANALYTIC MODEL: P(T) for each phase
# ---------------------------------------------------------------------------
def step3():
    info = load_info()
    peak, bw, cap = info["peak_matmul_flops"], info["peak_bw_bytes_s"], info["power_cap_w"]
    dense, attn = info["dense_flops_per_token"], info["attn_flops_per_token_per_S"]
    wbytes, kvbytes = info["weight_bytes"], info["kv_bytes_per_token"]
    fit = {}

    # ===== DECODE: coupled P(T) from composing two measured laws =====
    decall = sorted(load_csv("decode.csv"), key=lambda r: r["batch"])
    dec, decwall, b_peak = _decode_clean(decall)
    B = col(dec, "batch"); Tm = col(dec, "throughput_tok_s"); Pm = col(dec, "power_avg_w")
    # law 1: step time affine in batch  -> T(B) = B/(t_fixed + beta*B)
    t_step = B / Tm
    (t_fixed, beta), *_ = np.linalg.lstsq(np.c_[np.ones_like(B), B], t_step, rcond=None)
    # law 2: power saturates with batch  -> P(B) = P_idle + A(1 - e^{-B/B0})
    P_idle, A, B0 = fit_saturation(B, Pm)
    # compose -> P(T): vary B, both laws give (T, P). Predict P at measured B:
    P_pred = P_idle + A * (1 - np.exp(-B / B0))
    fit["decode"] = {
        "principle": "memory-bound: raising batch raises bandwidth use -> raises BOTH T and P",
        "t_fixed_ms": t_fixed * 1e3, "beta_ms_per_seq": beta * 1e3,
        "tput_asymptote_tok_s": 1.0 / beta,
        "P_idle_w": P_idle, "P_asymptote_w": P_idle + A, "B0": B0,
        "weight_stream_ideal_ms": wbytes / bw * 1e3,
        "P_vs_T_mape_pct": mape(Pm, P_pred), "P_vs_T_r2": r2(Pm, P_pred),
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    Bg = np.linspace(0.5, b_peak, 400)
    Tg = Bg / (t_fixed + beta * Bg)
    Pg = P_idle + A * (1 - np.exp(-Bg / B0))
    ax.plot(Tg / 1e3, Pg, "k-", lw=2, label="analytic P(T) = P_idle+A(1−e^{−B/B0})")
    ax.scatter(Tm / 1e3, Pm, color="C0", s=80, zorder=5, label="measured")
    ax.axhline(P_idle + A, color="gray", ls=":", alpha=.6,
               label=f"power asymptote {P_idle+A:.0f} W")
    ax.axhline(cap, color="r", ls="--", alpha=.5, label=f"cap {cap:.0f} W")
    ax.set_xlabel("token throughput (k tok/s)"); ax.set_ylabel("GPU power (W)")
    ax.set_title(f"DECODE power↔throughput: COUPLED (memory-bound)\n"
                 f"MAPE {fit['decode']['P_vs_T_mape_pct']:.1f}%, "
                 f"R²={fit['decode']['P_vs_T_r2']:.3f}")
    ax.legend(loc="lower right"); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step3_decode_model.png", dpi=130)
    print(f"wrote {FIG}/step3_decode_model.png")

    # ===== PREFILL: decoupled -- P pinned at the compute roof, T set by cost =====
    allpre = sorted([r for r in load_csv("prefill.csv")], key=lambda r: r["seq_len"])
    S = col(allpre, "seq_len"); Tp = col(allpre, "throughput_tok_s"); Pp = col(allpre, "power_avg_w")
    Up = col(allpre, "util_gpu_avg")
    s_peak = S[int(np.argmax(Tp))]
    # per-token cost law (sets T at fixed R): 1/T = (C + k_attn*S)/(peak*mfu) = a+b*S
    cb = S >= s_peak
    (a_, b_), *_ = np.linalg.lstsq(np.c_[np.ones(cb.sum()), S[cb]], 1.0 / Tp[cb], rcond=None)
    mfu = dense / (peak * a_)
    P_sat = float(np.median(Pp[S >= 256]))
    cvP = float(np.std(Pp[S >= 256]) / np.mean(Pp[S >= 256]))
    fit["prefill"] = {
        "principle": "compute-bound: R pinned at the matmul roof -> P fixed; T varies via per-token cost",
        "implied_mfu": mfu, "compute_bound_power_w": P_sat,
        "power_cv_pct_S>=256": cvP * 100,
        "throughput_span_kTps": [float(Tp[S >= 256].min() / 1e3), float(Tp[S >= 256].max() / 1e3)],
        "attn_doubles_at_S": a_ / b_,
        "cost_model": "1/T = (C + k_attn*S)/(peak*mfu)",
    }
    # power-utilisation law for the ramp: P = P_idle + (P_sat-P_idle)*(u/u_sat)
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(Tp / 1e3, Pp, c=Up, cmap="viridis", s=90, zorder=5, vmin=0, vmax=100)
    ax.axhline(P_sat, color="k", lw=2, label=f"compute roof: P ≈ {P_sat:.0f} W ⊥ throughput")
    ax.axvspan(Tp[S >= 256].min() / 1e3, Tp[S >= 256].max() / 1e3, color="C1", alpha=.08)
    ax.annotate(f"power constant ({cvP*100:.0f}% CV)\nover a {Tp[S>=256].max()/Tp[S>=256].min():.1f}× "
                f"throughput range", (Tp[S >= 256].mean()/1e3, P_sat+3), fontsize=8, ha="center")
    cb2 = fig.colorbar(sc, label="GPU utilisation (%)")
    ax.set_xlabel("token throughput (k tok/s)"); ax.set_ylabel("GPU power (W)")
    ax.set_ylim(50, cap * 1.05)
    ax.set_title("PREFILL power↔throughput: DECOUPLED (compute-bound)\n"
                 "GPU saturates → power pinned; attention O(S²) moves throughput, not power")
    ax.legend(loc="lower right"); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step3_prefill_model.png", dpi=130)
    print(f"wrote {FIG}/step3_prefill_model.png")

    with open(os.path.join(C.RESULTS_DIR, "fit_summary.json"), "w") as f:
        json.dump(fit, f, indent=2)
    print(f"wrote {C.RESULTS_DIR}/fit_summary.json")
    print(f"\n[fit] decode  P(T): P_idle {P_idle:.0f}→{P_idle+A:.0f} W, asym {1/beta:.0f} tok/s, "
          f"MAPE {fit['decode']['P_vs_T_mape_pct']:.1f}%, R²={fit['decode']['P_vs_T_r2']:.3f}")
    print(f"[fit] prefill P(T): P≈{P_sat:.0f} W (CV {cvP*100:.1f}%) ⊥ throughput; MFU {mfu*100:.0f}%")


# ---------------------------------------------------------------------------
# Step 4 -- SYNTHESIS: both phases on one throughput axis
# ---------------------------------------------------------------------------
def step4():
    info = load_info()
    cap = info["power_cap_w"]
    pre = sorted(load_csv("prefill.csv"), key=lambda r: r["throughput_tok_s"])
    decclean, _, _ = _decode_clean(sorted(load_csv("decode.csv"), key=lambda r: r["batch"]))
    dec = sorted(decclean, key=lambda r: r["throughput_tok_s"])

    # (a) POWER vs THROUGHPUT, both phases
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(col(dec, "throughput_tok_s") / 1e3, col(dec, "power_avg_w"), "o-",
            color="C0", label="decode (memory-bound, coupled)")
    ax.plot(col(pre, "throughput_tok_s") / 1e3, col(pre, "power_avg_w"), "s",
            color="C1", ms=8, label="prefill (compute-bound, decoupled)")
    ax.axhline(cap, color="r", ls="--", alpha=.5, label=f"cap {cap:.0f} W")
    ax.set_xscale("log")
    ax.set_xlabel("token throughput (k tok/s)"); ax.set_ylabel("GPU power (W)")
    ax.set_title("Power vs token throughput — prefill vs decode")
    ax.annotate("decode: more batch →\nmore tok/s AND more watts",
                (0.032, 118), fontsize=8.5, color="C0")
    ax.annotate("prefill: ~cap power\nat any throughput", (2.6, 128), fontsize=8.5, color="C1")
    ax.legend(loc="lower right"); ax.grid(alpha=.3, which="both"); fig.tight_layout()
    fig.savefig(f"{FIG}/step4_combined_power_vs_throughput.png", dpi=130)
    print(f"wrote {FIG}/step4_combined_power_vs_throughput.png")

    # (b) EFFICIENCY vs THROUGHPUT, both phases
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(col(dec, "throughput_tok_s") / 1e3, col(dec, "tok_per_joule"), "o-",
            color="C0", label="decode")
    ax.plot(col(pre, "throughput_tok_s") / 1e3, col(pre, "tok_per_joule"), "s",
            color="C1", ms=8, label="prefill")
    ax.set_xscale("log")
    pj = np.nanmax(col(pre, "tok_per_joule")); dj = np.nanmax(col(dec, "tok_per_joule"))
    ax.set_xlabel("token throughput (k tok/s)"); ax.set_ylabel("energy efficiency (tok/J)")
    ax.set_title(f"Energy efficiency vs throughput (prefill ~{pj/dj:.0f}× decode at best)")
    ax.legend(); ax.grid(alpha=.3, which="both"); fig.tight_layout()
    fig.savefig(f"{FIG}/step4_combined_efficiency_vs_throughput.png", dpi=130)
    print(f"wrote {FIG}/step4_combined_efficiency_vs_throughput.png")

    print(f"\n[synthesis] decode buys throughput with power (54→139 W); prefill draws "
          f"~cap at any throughput. best tok/J prefill {pj:.0f} vs decode {dj:.1f} ({pj/dj:.0f}×)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all")
    args = ap.parse_args()
    steps = {"1": [step1], "2": [step2], "3": [step3], "4": [step4],
             "all": [step1, step2, step3, step4]}[args.step]
    for fn in steps:
        fn()


if __name__ == "__main__":
    main()
