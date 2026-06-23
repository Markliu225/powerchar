"""Per-model theory+experiment comparison: one figure per model, prefill | decode.

For each results/mm_<slug>_info.json (+ _prefill.csv + _decode.csv), draw a 1x2 figure:
  LEFT  PREFILL  P vs T (batch sweep): measured points + composed analytic curve
        P(T) from P(B)=P0+A(1-e^{-B/B0}) and T(B)=S·B/(t_f+βB); + compute roof Φ/c_pre + cap.
  RIGHT DECODE   P vs T (batch sweep): measured points + same composed curve (fit up to the
        throughput peak) + bandwidth ceiling β/(C·kv_tok) + cap.
Saves figures/mm_<slug>_pt.png for every model.

  python code/plot_per_model.py
"""
from __future__ import annotations
import csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C


def loadcsv(p):
    with open(p) as f:
        return [{k: (float(v) if _num(v) else v) for k, v in r.items()} for r in csv.DictReader(f)]


def _num(v):
    try: float(v); return True
    except (ValueError, TypeError): return False


def col(rows, k):
    return np.array([r[k] for r in rows], float)


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - p) ** 2) / ss if ss > 0 else float("nan")


def fit_compose(B, T, P, n):
    """affine time t=tf+β·B (ceiling n/β) + saturating power P=P0+A(1-e^{-B/B0})."""
    t = n * B / T
    (tf, beta), *_ = np.linalg.lstsq(np.c_[np.ones_like(B), B], t, rcond=None)
    best = None
    for B0 in np.linspace(max(B.max() * 0.05, 1e-3), B.max() * 3, 400):
        basis = np.c_[np.ones_like(B), 1 - np.exp(-B / B0)]
        c, *_ = np.linalg.lstsq(basis, P, rcond=None)
        res = np.sum((basis @ c - P) ** 2)
        if best is None or res < best[0]:
            best = (res, B0, c[0], c[1])
    _, B0, P0, A = best
    Bg = np.linspace(B.min() * 0.9, B.max() * 1.05, 240)
    Tg = n * Bg / (tf + beta * Bg)
    Pg = P0 + A * (1 - np.exp(-Bg / B0))
    predP = P0 + A * (1 - np.exp(-B / B0))
    ceil = n / beta if beta > 0 else np.nan
    return Tg, Pg, ceil, r2(P, predP)


def main():
    cap_default = 250.0
    infos = sorted(glob.glob(os.path.join(C.RESULTS_DIR, "mm_*_info.json")))
    if not infos:
        print("no mm_*_info.json — run run_multimodel.sh first"); return
    for ip in infos:
        slug = os.path.basename(ip)[3:-10]
        info = json.load(open(ip))
        Phi = info["peak_matmul_flops"]; beta = info["peak_bw_bytes_s"]
        cap = info.get("power_cap_w", cap_default); kv_tok = info["kv_bytes_per_token"]
        dense = info["dense_flops_per_token"]; attn = info["attn_flops_per_token_per_S"]
        nkv = info["num_kv_heads"]; nq = info["num_attention_heads"]
        atttype = "MHA" if nkv == nq else f"GQA{nq // nkv}x"
        pp = f"{C.RESULTS_DIR}/mm_{slug}_prefill.csv"; dp = f"{C.RESULTS_DIR}/mm_{slug}_decode.csv"
        if not (os.path.exists(pp) and os.path.exists(dp)):
            print(f"skip {slug}: missing prefill/decode csv"); continue
        pre = sorted(loadcsv(pp), key=lambda r: r["batch"])
        dec = sorted(loadcsv(dp), key=lambda r: r["batch"])

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

        # ---- PREFILL ----
        S = pre[0]["seq_len"]; Bp = col(pre, "batch"); Tp = col(pre, "throughput_tok_s"); Pp = col(pre, "power_avg_w")
        c_pre = dense + attn * S
        roof = Phi / c_pre
        Tg, Pg, ceilp, r2p = fit_compose(Bp, Tp, Pp, S)
        axL.scatter(Tp / 1e3, Pp, c="C1", s=80, edgecolor="k", lw=.4, zorder=5, label="measured (batch sweep)")
        axL.plot(Tg / 1e3, Pg, "k-", lw=2, label=f"analytic P(T) (R²={r2p:.3f})")
        axL.axvline(roof / 1e3, color="C2", ls=":", lw=2, label=f"compute roof Φ/c_pre={roof/1e3:.1f}k")
        axL.axhline(cap, color="r", ls="--", alpha=.5, label=f"cap {cap:.0f} W")
        axL.set_xlabel("throughput (k tok/s)"); axL.set_ylabel("power (W)")
        axL.set_title(f"PREFILL  (S={int(S)}, batch sweep)"); axL.legend(fontsize=8); axL.grid(alpha=.3)
        axL.set_xlim(0, max(roof, Tp.max()) / 1e3 * 1.06); axL.set_ylim(0, cap * 1.08)

        # ---- DECODE (fit up to throughput peak) ----
        Cc = dec[0]["ctx_len"]; Bd = col(dec, "batch"); Td = col(dec, "throughput_tok_s"); Pd = col(dec, "power_avg_w")
        kpk = int(np.argmax(Td)); sel = slice(0, kpk + 1)
        bw_ceil = beta / (Cc * kv_tok)
        Tg2, Pg2, ceild, r2d = fit_compose(Bd[sel], Td[sel], Pd[sel], 1.0)
        axR.scatter(Td / 1e3, Pd, c="C0", s=80, edgecolor="k", lw=.4, zorder=5, label="measured (batch sweep)")
        axR.plot(Tg2 / 1e3, Pg2, "k-", lw=2, label=f"analytic P(T) (R²={r2d:.3f})")
        axR.axhline(cap, color="r", ls="--", alpha=.5, label=f"cap {cap:.0f} W")
        axR.set_xlim(0, max(Td.max(), Tg2.max()) / 1e3 * 1.1)
        axR.annotate(f"theory bandwidth ceiling β/(C·kv) = {bw_ceil/1e3:.0f}k tok/s\n"
                     f"(≫ measured peak {Td.max()/1e3:.2f}k → decode is launch-overhead-limited, not BW)",
                     xy=(0.03, 0.96), xycoords="axes fraction", fontsize=7.5, color="C2", va="top")
        axR.set_xlabel("throughput (k tok/s)"); axR.set_ylabel("power (W)")
        axR.set_title(f"DECODE  (ctx={int(Cc)}, batch sweep)"); axR.legend(fontsize=8, loc="lower right"); axR.grid(alpha=.3)
        axR.set_ylim(0, cap * 1.08)

        fig.suptitle(f"{slug}  ·  {atttype}, L={info['num_layers']} d={info['hidden_size']} "
                     f"n_kv={nkv} h={info['head_dim']} · {info['total_params']/1e9:.1f}B · "
                     f"kv/tok={kv_tok/1e6:.3f} MB · V100", fontsize=12)
        fig.tight_layout()
        out = os.path.join(C.FIGURES_DIR, f"mm_{slug}_pt.png")
        fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}  | prefill roof {roof/1e3:.1f}k (meas {Tp.max()/1e3:.1f}k, MFU {Tp.max()/roof:.2f}); "
              f"decode bw-ceil {bw_ceil/1e3:.1f}k (meas peak {Td.max()/1e3:.2f}k)")


if __name__ == "__main__":
    main()
