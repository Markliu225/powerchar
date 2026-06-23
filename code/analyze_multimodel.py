"""Cross-model comparison: measured vs analytical-model predictions.

Loads results/mm_<slug>_info.json + mm_<slug>_prefill.csv + mm_<slug>_decode.csv
(+ mm_<slug>_dvfs.csv where present) and, per model, contrasts the architecture-grounded
predictions against measurement:

  prefill compute roof   theory  T_pre^max = Phi / c_pre(S),  c_pre = dense + attn·S
                         measured = max prefill throughput  ->  implied MFU = meas/theory
  decode bandwidth ceil  theory  T_dec^max = beta / (C·kv_tok),  kv_tok = 2·L·n_kv·h·b
                         measured = max decode throughput (often overhead-limited, < theory)
  decode power slope     measured affine fit  P = a + s·T

Key cross-model lever: GQA (small n_kv -> small kv_tok) lifts the decode ceiling far above
MHA. Produces figures/mm_compare.png and prints a summary table.

  python code/analyze_multimodel.py
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
    try:
        float(v); return True
    except (ValueError, TypeError):
        return False


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    infos = sorted(glob.glob(os.path.join(C.RESULTS_DIR, "mm_*_info.json")))
    rows = []
    for ip in infos:
        slug = os.path.basename(ip)[3:-10]            # mm_<slug>_info.json
        info = json.load(open(ip))
        L = info["num_layers"]; d = info["hidden_size"]; nkv = info["num_kv_heads"]
        h = info["head_dim"]; b = info["bytes_per_param"]
        Phi = info["peak_matmul_flops"]; beta = info["peak_bw_bytes_s"]
        kv_tok = info["kv_bytes_per_token"]; dense = info["dense_flops_per_token"]
        attn = info["attn_flops_per_token_per_S"]
        pre = sorted(loadcsv(f"{C.RESULTS_DIR}/mm_{slug}_prefill.csv"), key=lambda r: r["batch"]) \
            if os.path.exists(f"{C.RESULTS_DIR}/mm_{slug}_prefill.csv") else []
        dec = sorted(loadcsv(f"{C.RESULTS_DIR}/mm_{slug}_decode.csv"), key=lambda r: r["batch"]) \
            if os.path.exists(f"{C.RESULTS_DIR}/mm_{slug}_decode.csv") else []
        rec = {"slug": slug, "L": L, "d": d, "n_kv": nkv, "n_q": info["num_attention_heads"],
               "params_B": info["total_params"] / 1e9, "kv_tok_MB": kv_tok / 1e6,
               "attn": "MHA" if nkv == info["num_attention_heads"] else f"GQA{info['num_attention_heads']//nkv}x"}
        if pre:
            S = pre[0]["seq_len"]; c_pre = dense + attn * S
            T_meas = max(r["throughput_tok_s"] for r in pre)
            rec.update(S=S, roof_theory=Phi / c_pre, prefill_meas=T_meas, MFU=T_meas / (Phi / c_pre))
        if dec:
            Cc = dec[0]["ctx_len"]
            Td = np.array([r["throughput_tok_s"] for r in dec]); Pd = np.array([r["power_avg_w"] for r in dec])
            Bd = np.array([r["batch"] for r in dec])
            # decode bandwidth ceiling (theory) and measured peak
            rec.update(C=Cc, bw_ceiling_theory=beta / (Cc * kv_tok), decode_meas=Td.max())
            # affine step time t=B/T -> ceiling 1/slope ; and power slope
            t = Bd / Td
            sl, t0 = np.polyfit(Bd, t, 1)
            rec["decode_ceiling_fit"] = 1.0 / sl if sl > 0 else float("nan")
            ps, pa = np.polyfit(Td, Pd, 1)
            rec.update(decode_pslope=ps, decode_pintercept=pa, decode_pR2=r2(Pd, pa + ps * Td))
        rows.append(rec)

    if not rows:
        print("no mm_*_info.json found — run run_multimodel.sh first"); return

    # ---- summary table ----
    hdr = ["model", "attn", "params", "L", "d", "n_kv", "kv/tok(MB)",
           "roofTh(k)", "preMeas(k)", "MFU", "bwCeil(k)", "decMeas(k)", "P=a+sT"]
    print("  ".join(f"{x:>10}" for x in hdr))
    for r in sorted(rows, key=lambda r: r.get("params_B", 0)):
        print("  ".join(f"{v:>10}" for v in [
            r["slug"][:10], r["attn"], f"{r.get('params_B',0):.1f}B", r["L"], r["d"], r["n_kv"],
            f"{r['kv_tok_MB']:.3f}",
            f"{r.get('roof_theory',0)/1e3:.1f}", f"{r.get('prefill_meas',0)/1e3:.1f}",
            f"{r.get('MFU',0):.2f}", f"{r.get('bw_ceiling_theory',0)/1e3:.1f}",
            f"{r.get('decode_meas',0)/1e3:.2f}",
            f"{r.get('decode_pintercept',0):.0f}+{r.get('decode_pslope',0):.3f}T"]))

    # ---- figure: theory vs measured across models ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))
    col = {"MHA": "C3"}
    for r in rows:
        c = "C3" if r["attn"] == "MHA" else "C0"
        if "roof_theory" in r:
            axA.scatter(r["roof_theory"] / 1e3, r["prefill_meas"] / 1e3, c=c, s=90, edgecolor="k", lw=.4, zorder=5)
            axA.annotate(f"{r['slug'][:10]}", (r["roof_theory"] / 1e3, r["prefill_meas"] / 1e3),
                         fontsize=7, xytext=(4, 3), textcoords="offset points")
        if "bw_ceiling_theory" in r:
            axB.scatter(r["bw_ceiling_theory"] / 1e3, r["decode_meas"] / 1e3, c=c, s=90, edgecolor="k", lw=.4, zorder=5)
            axB.annotate(f"{r['slug'][:10]} ({r['attn']})", (r["bw_ceiling_theory"] / 1e3, r["decode_meas"] / 1e3),
                         fontsize=7, xytext=(4, 3), textcoords="offset points")
    # prefill: measured = MFU * theory (line through origin at median MFU)
    mfus = [r["MFU"] for r in rows if "MFU" in r]
    if mfus:
        mx = max(r["roof_theory"] for r in rows if "roof_theory" in r) / 1e3 * 1.05
        for mfu, ls in [(1.0, ":"), (np.median(mfus), "--")]:
            axA.plot([0, mx], [0, mfu * mx], ls, color="gray",
                     label=f"MFU={mfu:.2f}" + (" (ideal)" if mfu == 1.0 else " (median)"))
    axA.set_xlabel("theory compute roof  Φ/c_pre  (k tok/s)")
    axA.set_ylabel("measured prefill saturation (k tok/s)")
    axA.set_title("PREFILL compute roof: measured vs theory (per model)")
    axA.legend(fontsize=8); axA.grid(alpha=.3)
    axB.set_xlabel("theory bandwidth ceiling  β/(C·kv_tok)  (k tok/s)")
    axB.set_ylabel("measured decode peak (k tok/s)")
    axB.set_title("DECODE ceiling: measured vs theory — GQA(blue) lifts it far above MHA(red)")
    axB.legend(["MHA", "GQA"], fontsize=8); axB.grid(alpha=.3)
    fig.suptitle("Cross-model validation of the architecture-grounded P(T) model — V100", fontsize=13)
    fig.tight_layout()
    out = os.path.join(C.FIGURES_DIR, "mm_compare.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); print("\nwrote", out)


if __name__ == "__main__":
    main()
