"""Plotting + analytic-model fitting for every step.

  python analyze.py --step 1   # prefill: throughput vs power, vs load, tok/J
  python analyze.py --step 2   # decode:  throughput vs power, vs batch, tok/J
  python analyze.py --step 3   # measured vs analytic-model overlays + fit json
  python analyze.py --step 4   # synthesis: prefill vs decode, efficiency
  python analyze.py --step all

Fits use numpy only (no scipy). Each step writes figures into figures/ and
prints the numbers that the markdown write-ups quote.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")        # Windows console is cp1252
except Exception:
    pass
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

FIG = C.FIGURES_DIR


def load_csv(name):
    path = os.path.join(C.RESULTS_DIR, name)
    with open(path) as f:
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


# ---------------------------------------------------------------------------
# Step 1 -- PREFILL
# ---------------------------------------------------------------------------
def step1():
    rows = sorted(load_csv("prefill.csv"), key=lambda r: r["load_tokens"])
    info = load_info()
    cap = info["power_cap_w"]
    load = col(rows, "load_tokens")
    tput = col(rows, "throughput_tok_s")
    power = col(rows, "power_avg_w")
    util = col(rows, "util_gpu_avg")
    tokJ = col(rows, "tok_per_joule")

    # (a) throughput vs power
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(power, tput / 1e3, c=np.log10(load), cmap="viridis", s=80, zorder=5)
    for r in rows:
        ax.annotate(f"{int(r['batch'])}x{int(r['seq_len'])}",
                    (r["power_avg_w"], r["throughput_tok_s"] / 1e3),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axvline(cap, color="r", ls="--", alpha=.5, label=f"power cap {cap:.0f} W")
    fig.colorbar(sc, label="log10(load tokens)")
    ax.set_xlabel("GPU power (W)"); ax.set_ylabel("throughput (k tok/s)")
    ax.set_title("PREFILL: token throughput vs GPU power\n(power pinned near cap; "
                 "throughput set by load & attention O(S²))")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step1_prefill_throughput_vs_power.png", dpi=130)
    print(f"wrote {FIG}/step1_prefill_throughput_vs_power.png")

    # (b) throughput & power vs load (dual axis)
    fig, ax = plt.subplots(figsize=(9, 6))
    l1 = ax.semilogx(load, tput / 1e3, "o-", color="C0", label="throughput")
    ax.set_xlabel("offered load = batch x seq_len (tokens)")
    ax.set_ylabel("throughput (k tok/s)", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    l2 = ax2.semilogx(load, power, "s--", color="C3", label="power")
    ax2.axhline(cap, color="r", ls=":", alpha=.5)
    ax2.set_ylabel("power (W)", color="C3"); ax2.tick_params(axis="y", labelcolor="C3")
    ax2.set_ylim(0, cap * 1.08)
    ax.set_title("PREFILL vs offered load")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step1_prefill_vs_load.png", dpi=130)
    print(f"wrote {FIG}/step1_prefill_vs_load.png")

    # (c) tok/J vs load
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(load, tokJ, "o-", color="C2")
    ax.set_xlabel("offered load (tokens)"); ax.set_ylabel("energy efficiency (tok/J)")
    ax.set_title("PREFILL energy efficiency"); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step1_prefill_efficiency.png", dpi=130)
    print(f"wrote {FIG}/step1_prefill_efficiency.png")

    i = int(np.argmax(tput))
    print(f"\n[prefill] power range {power.min():.0f}-{power.max():.0f} W "
          f"(cap {cap:.0f}); peak {tput[i]/1e3:.1f}k tok/s at {rows[i]['batch']:.0f}"
          f"x{rows[i]['seq_len']:.0f}; best {tokJ.max():.0f} tok/J; util {util.mean():.0f}% avg")


# ---------------------------------------------------------------------------
# Step 2 -- DECODE
# ---------------------------------------------------------------------------
def _decode_clean(rows):
    """Split decode points into the clean memory-bound regime (up to the
    throughput peak) and the post-peak VRAM-spill 'wall' points."""
    b = np.array([r["batch"] for r in rows])
    t = np.array([r["throughput_tok_s"] for r in rows])
    b_peak = b[int(np.argmax(t))]
    clean = [r for r in rows if r["batch"] <= b_peak]
    wall = [r for r in rows if r["batch"] > b_peak]
    return clean, wall, b_peak


def step2():
    rows = sorted(load_csv("decode.csv"), key=lambda r: r["batch"])
    info = load_info()
    cap = info["power_cap_w"]
    clean, wall, b_peak = _decode_clean(rows)
    cb, ct, cp = col(clean, "batch"), col(clean, "throughput_tok_s"), col(clean, "power_avg_w")
    batch = col(rows, "batch"); tput = col(rows, "throughput_tok_s")
    power = col(rows, "power_avg_w"); tokJ = col(rows, "tok_per_joule")

    # (a) throughput vs power
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cp, ct / 1e3, "-", color="gray", alpha=.4, zorder=1)
    sc = ax.scatter(cp, ct / 1e3, c=np.log2(cb), cmap="plasma", s=80, zorder=5)
    if wall:
        ax.scatter(col(wall, "power_avg_w"), col(wall, "throughput_tok_s") / 1e3,
                   facecolors="none", edgecolors="gray", s=80, zorder=4,
                   label="VRAM-spill wall (b>%d)" % b_peak)
    for r in rows:
        ax.annotate(f"b{int(r['batch'])}", (r["power_avg_w"], r["throughput_tok_s"] / 1e3),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axvline(cap, color="r", ls="--", alpha=.5, label=f"power cap {cap:.0f} W")
    fig.colorbar(sc, label="log2(batch)")
    ax.set_xlabel("GPU power (W)"); ax.set_ylabel("throughput (k tok/s)")
    ax.set_title("DECODE: token throughput vs GPU power\n(both rise with batch toward "
                 "the bandwidth/power ceiling)")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step2_decode_throughput_vs_power.png", dpi=130)
    print(f"wrote {FIG}/step2_decode_throughput_vs_power.png")

    # (b) throughput & power vs batch
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogx(batch, tput / 1e3, "o-", color="C0", label="throughput", base=2)
    ax.axvspan(b_peak, batch.max(), color="gray", alpha=.12)
    ax.set_xlabel("batch size (concurrent sequences)")
    ax.set_ylabel("throughput (k tok/s)", color="C0"); ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    ax2.semilogx(batch, power, "s--", color="C3", label="power", base=2)
    ax2.axhline(cap, color="r", ls=":", alpha=.5)
    ax2.set_ylabel("power (W)", color="C3"); ax2.tick_params(axis="y", labelcolor="C3")
    ax2.set_ylim(0, cap * 1.08)
    ax.set_title("DECODE vs batch size (shaded = VRAM-spill wall)")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step2_decode_vs_batch.png", dpi=130)
    print(f"wrote {FIG}/step2_decode_vs_batch.png")

    # (c) tok/J vs batch
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(batch, tokJ, "o-", color="C2", base=2)
    ax.axvspan(b_peak, batch.max(), color="gray", alpha=.12)
    ax.set_xlabel("batch size"); ax.set_ylabel("energy efficiency (tok/J)")
    ax.set_title("DECODE energy efficiency"); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step2_decode_efficiency.png", dpi=130)
    print(f"wrote {FIG}/step2_decode_efficiency.png")

    print(f"\n[decode] clean regime b<=%d: power {cp.min():.0f}->{cp.max():.0f} W; "
          % b_peak + f"throughput {ct.min():.0f}->{ct.max():.0f} tok/s; "
          f"best {np.nanmax(tokJ):.1f} tok/J; {len(wall)} spill points beyond peak")


# ---------------------------------------------------------------------------
# Step 3 -- ANALYTIC MODEL vs MEASURED
# ---------------------------------------------------------------------------
def fit_saturation(x, y):
    """y = y0 + A*(1 - exp(-x/x0)); grid x0, linear-LS y0,A. No scipy."""
    best = None
    for x0 in np.logspace(0, np.log10(max(x) * 1.5), 200):
        basis = np.c_[np.ones_like(x), 1 - np.exp(-x / x0)]
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        resid = np.sum((basis @ coef - y) ** 2)
        if best is None or resid < best[0]:
            best = (resid, x0, coef[0], coef[1])
    _, x0, y0, A = best
    return y0, A, x0


def step3():
    info = load_info()
    peak = info["peak_matmul_flops"]
    bw = info["peak_bw_bytes_s"]
    cap = info["power_cap_w"]
    dense = info["dense_flops_per_token"]
    attn = info["attn_flops_per_token_per_S"]
    wbytes = info["weight_bytes"]
    kvbytes = info["kv_bytes_per_token"]

    fit = {}

    # ---- PREFILL throughput model: 1/tput = (dense + attn*S)/(peak*mfu) ----
    # Prefill has two regimes: a RISING, occupancy-limited branch (small S, the
    # GPU is not yet full) and a DECLINING, compute-bound branch past the peak
    # where attention's O(S^2) cost dominates. The compute+attention model only
    # describes the declining branch, so we fit from the throughput peak onward.
    allpre = sorted([r for r in load_csv("prefill.csv") if r["batch"] == 1],
                    key=lambda r: r["seq_len"])
    Sall = col(allpre, "seq_len"); tpall = col(allpre, "throughput_tok_s")
    s_peak = Sall[int(np.argmax(tpall))]
    pre = [r for r in allpre if r["seq_len"] >= s_peak]
    S = col(pre, "seq_len")
    tp = col(pre, "throughput_tok_s")
    inv = 1.0 / tp
    A = np.c_[np.ones_like(S), S]                     # inv = a + b*S
    (a_, b_), *_ = np.linalg.lstsq(A, inv, rcond=None)
    pred_pre = 1.0 / (a_ + b_ * S)
    mfu_dense = dense / (peak * a_)                    # implied compute efficiency
    attn_eff = b_ * (peak * mfu_dense) / attn          # implied attn FLOP multiplier
    fit["prefill"] = {
        "model": "tput(S) = 1 / (a + b*S);  a=dense/(peak*mfu), b=attn/(peak*mfu)",
        "fit_regime": f"S >= {s_peak:.0f} (compute-bound, post-peak)",
        "a": a_, "b": b_, "implied_mfu": mfu_dense, "attn_term_ratio": attn_eff,
        "attn_doubles_at_S": a_ / b_,                  # S where attn cost == dense cost
        "mape_pct": mape(tp, pred_pre), "r2": r2(tp, pred_pre),
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(Sall[Sall < s_peak], tpall[Sall < s_peak] / 1e3, facecolors="none",
               edgecolors="C1", s=70, zorder=5, label="occupancy-limited (excluded)")
    ax.scatter(S, tp / 1e3, color="C1", s=70, zorder=5, label="compute-bound (fit)")
    Sg = np.logspace(np.log10(s_peak), np.log10(S.max()), 200)
    ax.plot(Sg, 1e-3 / (a_ + b_ * Sg), "k-", lw=2,
            label=f"model 1/(a+bS), MFU≈{mfu_dense*100:.0f}%")
    ax.plot(Sg, np.full_like(Sg, 1e-3 * peak * mfu_dense / dense), "b:", alpha=.6,
            label="compute roof (attn→0)")
    ax.set_xscale("log")
    ax.set_xlabel("seq_len S (batch=1)"); ax.set_ylabel("throughput (k tok/s)")
    ax.set_title(f"PREFILL measured vs analytic (compute-bound + O(S²) attention)\n"
                 f"fit S≥{s_peak:.0f}: MAPE {fit['prefill']['mape_pct']:.1f}%, "
                 f"R²={fit['prefill']['r2']:.3f}")
    ax.legend(); ax.grid(alpha=.3, which="both"); fig.tight_layout()
    fig.savefig(f"{FIG}/step3_prefill_model.png", dpi=130)
    print(f"wrote {FIG}/step3_prefill_model.png")

    # ---- DECODE step-time model: affine in batch ----
    # The measured decode step time is affine in B: a FIXED per-step cost (stream
    # all weights once + per-step kernel-launch overhead, which on WDDM has no
    # CUDA-graph amortisation) plus a per-sequence MARGINAL cost (KV read +
    # compute). t_step(B) = t_fixed + beta*B  ->  tput(B) = B/(t_fixed+beta*B),
    # a saturating curve with asymptote 1/beta. The pure memory roofline (no
    # fixed launch overhead) is the unreachable upper bound it falls short of.
    decall = sorted(load_csv("decode.csv"), key=lambda r: r["batch"])
    dec, decwall, b_peak = _decode_clean(decall)     # fit clean regime only
    B = col(dec, "batch")
    td = col(dec, "throughput_tok_s")
    mfu = fit["prefill"]["implied_mfu"]
    ctx = dec[0]["ctx_len"]
    t_step_meas = B / td                              # measured seconds per step
    (t_fixed, beta), *_ = np.linalg.lstsq(np.c_[np.ones_like(B), B], t_step_meas, rcond=None)
    pred_dec = B / (t_fixed + beta * B)
    t_weight_ideal = wbytes / bw                      # 100%-BW weight-stream time
    eff_bw_decode = wbytes / (bw * t_fixed)           # apparent BW util of fixed cost
    fit["decode"] = {
        "model": "t_step = t_fixed + beta*B ;  tput = B/(t_fixed+beta*B)",
        "t_fixed_ms": t_fixed * 1e3, "beta_ms_per_seq": beta * 1e3,
        "t_weight_ideal_ms": t_weight_ideal * 1e3,
        "launch_overhead_ms": (t_fixed - t_weight_ideal) * 1e3,
        "fixed_cost_bw_efficiency": eff_bw_decode,
        "tput_asymptote_tok_s": 1.0 / beta,
        "halfmax_batch": t_fixed / beta, "ridge_batch": peak / bw,
        "mape_pct": mape(td, pred_dec), "r2": r2(td, pred_dec),
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(B, td / 1e3, color="C0", s=70, zorder=5, label="measured (clean)")
    if decwall:
        ax.scatter(col(decwall, "batch"), col(decwall, "throughput_tok_s") / 1e3,
                   facecolors="none", edgecolors="gray", s=70, zorder=4,
                   label="VRAM-spill wall (excluded)")
    Bg = np.logspace(0, np.log10(col(decall, "batch").max()), 200)
    ax.plot(Bg, 1e-3 * Bg / (t_fixed + beta * Bg), "k-", lw=2,
            label=f"affine model (asymptote {1e-3/beta:.1f}k tok/s)")
    ax.plot(Bg, 1e-3 * Bg * bw / wbytes, "b:", alpha=.6,
            label="ideal weight-stream roofline")
    ax.axhline(1e-3 / beta, color="gray", ls=":", alpha=.5,
               label=f"asymptote 1/β = {1e-3/beta:.1f}k tok/s")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_ylim(0.02, 40)
    ax.set_xlabel("batch size B"); ax.set_ylabel("throughput (k tok/s)")
    ax.set_title(f"DECODE measured vs model (affine step time)\n"
                 f"t_fixed={t_fixed*1e3:.0f} ms (~{eff_bw_decode*100:.0f}% BW), "
                 f"β={beta*1e3:.2f} ms/seq, MAPE {fit['decode']['mape_pct']:.1f}%, "
                 f"R²={fit['decode']['r2']:.3f}")
    ax.legend(); ax.grid(alpha=.3, which="both"); fig.tight_layout()
    fig.savefig(f"{FIG}/step3_decode_model.png", dpi=130)
    print(f"wrote {FIG}/step3_decode_model.png")

    # ---- POWER model: P(util) saturating toward cap (decode batch sweep) ----
    Pd = col(dec, "power_avg_w")
    y0, Amp, b0 = fit_saturation(B, Pd)
    pred_P = y0 + Amp * (1 - np.exp(-B / b0))
    fit["power"] = {
        "model": "P(B) = P_idle + A*(1 - exp(-B/B0))",
        "P_idle": y0, "A": Amp, "B0": b0, "P_asymptote": y0 + Amp,
        "mape_pct": mape(Pd, pred_P), "r2": r2(Pd, pred_P),
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(B, Pd, color="C3", s=70, zorder=5, label="measured power")
    ax.plot(Bg, y0 + Amp * (1 - np.exp(-Bg / b0)), "k-", lw=2,
            label=f"P={y0:.0f}+{Amp:.0f}(1-e^(-B/{b0:.0f}))")
    ax.axhline(cap, color="r", ls="--", alpha=.5, label=f"cap {cap:.0f} W")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size B"); ax.set_ylabel("power (W)")
    ax.set_title(f"DECODE power vs batch -- saturation model\n"
                 f"P_idle≈{y0:.0f} W, asymptote≈{y0+Amp:.0f} W, "
                 f"R²={fit['power']['r2']:.3f}")
    ax.legend(); ax.grid(alpha=.3, which="both"); fig.tight_layout()
    fig.savefig(f"{FIG}/step3_power_model.png", dpi=130)
    print(f"wrote {FIG}/step3_power_model.png")

    with open(os.path.join(C.RESULTS_DIR, "fit_summary.json"), "w") as f:
        json.dump(fit, f, indent=2)
    print(f"wrote {C.RESULTS_DIR}/fit_summary.json")
    print("\n[fit] prefill: MFU≈{:.0f}%, MAPE {:.1f}%, R²={:.3f}".format(
        fit["prefill"]["implied_mfu"] * 100, fit["prefill"]["mape_pct"], fit["prefill"]["r2"]))
    print("[fit] decode:  t_fixed={:.0f} ms (~{:.0f}% BW), β={:.2f} ms/seq, "
          "asym {:.0f} tok/s, MAPE {:.1f}%, R²={:.3f}".format(
              fit["decode"]["t_fixed_ms"], fit["decode"]["fixed_cost_bw_efficiency"] * 100,
              fit["decode"]["beta_ms_per_seq"], fit["decode"]["tput_asymptote_tok_s"],
              fit["decode"]["mape_pct"], fit["decode"]["r2"]))
    print("[fit] power:   P_idle≈{:.0f} W, asymptote≈{:.0f} W, R²={:.3f}".format(
        fit["power"]["P_idle"], fit["power"]["P_asymptote"], fit["power"]["r2"]))


# ---------------------------------------------------------------------------
# Step 4 -- SYNTHESIS
# ---------------------------------------------------------------------------
def step4():
    info = load_info()
    cap = info["power_cap_w"]
    pre = sorted([r for r in load_csv("prefill.csv")], key=lambda r: r["power_avg_w"])
    decclean, _, _ = _decode_clean(sorted(load_csv("decode.csv"), key=lambda r: r["batch"]))
    dec = sorted(decclean, key=lambda r: r["power_avg_w"])

    # (a) combined throughput vs power -- prefill is a high-power vertical band
    # (power pinned near cap), decode a low-power rising diagonal.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(col(pre, "power_avg_w"), col(pre, "throughput_tok_s") / 1e3,
               marker="s", color="C1", s=70, label="prefill (compute-bound)")
    ax.plot(col(dec, "power_avg_w"), col(dec, "throughput_tok_s") / 1e3, "o-",
            color="C0", label="decode (memory-bound)")
    ax.axvline(cap, color="r", ls="--", alpha=.5, label=f"cap {cap:.0f} W")
    ax.annotate("power pinned ~140 W,\nthroughput set by attention",
                (140, 6.5), fontsize=8, color="C1", ha="center")
    ax.annotate("both rise with batch", (105, 0.9), fontsize=8, color="C0")
    ax.set_xlabel("GPU power (W)"); ax.set_ylabel("throughput (k tok/s)")
    ax.set_title("Prefill vs Decode: token throughput vs GPU power")
    ax.legend(loc="center left"); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{FIG}/step4_combined_throughput_vs_power.png", dpi=130)
    print(f"wrote {FIG}/step4_combined_throughput_vs_power.png")

    # (b) efficiency comparison tok/J
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(col(pre, "throughput_tok_s") / 1e3, col(pre, "tok_per_joule"),
               color="C1", s=70, label="prefill")
    ax.scatter(col(dec, "throughput_tok_s") / 1e3, col(dec, "tok_per_joule"),
               color="C0", s=70, label="decode")
    pj = np.nanmax(col(pre, "tok_per_joule"))
    dj = np.nanmax(col(dec, "tok_per_joule"))
    ax.set_xscale("log")
    ax.set_xlabel("throughput (k tok/s)"); ax.set_ylabel("energy efficiency (tok/J)")
    ax.set_title(f"Energy efficiency: prefill {np.nanmin(col(pre,'tok_per_joule')):.0f}-{pj:.0f} "
                 f"tok/J vs decode {np.nanmin(col(dec,'tok_per_joule')):.1f}-{dj:.1f} tok/J "
                 f"(~{pj/dj:.0f}x at best)")
    ax.legend(); ax.grid(alpha=.3, which="both"); fig.tight_layout()
    fig.savefig(f"{FIG}/step4_efficiency_comparison.png", dpi=130)
    print(f"wrote {FIG}/step4_efficiency_comparison.png")

    print(f"\n[synthesis] best prefill {pj:.0f} tok/J vs best decode {dj:.1f} tok/J "
          f"-> {pj/dj:.0f}x")


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
