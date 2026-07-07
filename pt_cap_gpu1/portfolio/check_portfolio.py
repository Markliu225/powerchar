"""Pre-flight sanity gate for the portfolio — run BEFORE burning GPU hours.

For every workload it computes, purely from architecture numbers:
  - decode VRAM   = weights + B*C*kv_bytes_per_token   (+ a rough activation pad)  -> must fit 32 GB
  - rough predicted decode T_max = B * BW_eff / (weights + B*C*kv/tok)
    (BW_eff calibrated so the Phi-3 chat baseline hits its measured 721 tok/s; this is only a
     constant-BW approximation — effective BW actually rises a bit with batch — so treat T_max as
     an order-of-magnitude check that the workloads span distinct regimes, not an exact prediction)
  - context <= model max_ctx
It flags anything that would OOM, exceed the context window, or duplicate another workload's regime.

  python3 check_portfolio.py
"""
from __future__ import annotations
from portfolio import PORTFOLIO, CAP_GRID

# fp16 arch facts from results/mm_*_info.json  (+ published context windows)
ARCH = {
    "facebook/opt-1.3b":                dict(weight_b=2_631_516_160,  kv_b=196_608, max_ctx=2048),
    "microsoft/Phi-3-mini-4k-instruct": dict(weight_b=7_642_159_104,  kv_b=393_216, max_ctx=4096),
    "Qwen/Qwen2.5-1.5B-Instruct":       dict(weight_b=3_087_428_608,  kv_b=28_672,  max_ctx=32768),
    "Qwen/Qwen2.5-3B-Instruct":         dict(weight_b=6_171_877_376,  kv_b=36_864,  max_ctx=32768),
    "Qwen/Qwen2.5-7B-Instruct":         dict(weight_b=15_231_233_024, kv_b=57_344,  max_ctx=32768),
}
VRAM_GB = 32.0
HEADROOM_GB = 6.0

# Calibrate BW_eff to the Phi-3 chat baseline: T_max=721 @ C=256,B=64.
_a = ARCH["microsoft/Phi-3-mini-4k-instruct"]
_Dmem_ref = _a["weight_b"] + 64 * 256 * _a["kv_b"]
BW_EFF = 721.0 * _Dmem_ref / 64          # bytes/s


def gb(x):
    return x / 1e9


def main():
    print(f"cap grid: {CAP_GRID}")
    print(f"calibrated BW_eff = {gb(BW_EFF):.0f} GB/s (from Phi-3 chat baseline 721 tok/s)\n")
    hdr = f"{'id':<16}{'model':<26}{'pre S×B':>10}{'dec C×B':>11}{'KV GB':>7}{'VRAM GB':>9}{'~T_max':>8}  flags"
    print(hdr); print("-" * len(hdr))
    tmax_seen = []
    for w in PORTFOLIO:
        a = ARCH.get(w["model_id"])
        flags = []
        if not a:
            print(f"{w['id']:<16}{w['model_id']:<26}  UNKNOWN MODEL"); continue
        kv = w["decode_batch"] * w["decode_ctx"] * a["kv_b"]
        vram = a["weight_b"] + kv
        vram_pad = vram + 0.12 * vram + 0.5e9        # rough activation/workspace pad
        tmax = w["decode_batch"] * BW_EFF / vram
        if gb(vram_pad) > VRAM_GB - HEADROOM_GB:
            flags.append(f"VRAM {gb(vram_pad):.1f}>{VRAM_GB-HEADROOM_GB:.0f}")
        if w["decode_ctx"] > a["max_ctx"]:
            flags.append(f"CTX {w['decode_ctx']}>{a['max_ctx']}")
        if w["prefill_seq_len"] > a["max_ctx"]:
            flags.append(f"preS {w['prefill_seq_len']}>{a['max_ctx']}")
        # regime-distinctness: warn if predicted T_max within 25% of an already-seen one
        for t0, id0 in tmax_seen:
            if abs(tmax - t0) / max(tmax, t0) < 0.25:
                flags.append(f"~dup regime of {id0}")
                break
        tmax_seen.append((tmax, w["id"]))
        model_short = w["model_id"].split("/")[-1]
        print(f"{w['id']:<16}{model_short:<26}"
              f"{str(w['prefill_seq_len'])+'×'+str(w['prefill_batch']):>10}"
              f"{str(w['decode_ctx'])+'×'+str(w['decode_batch']):>11}"
              f"{gb(kv):>7.1f}{gb(vram_pad):>9.1f}{tmax:>8.0f}  "
              f"{'  '.join(flags) if flags else 'OK'}")
    print()
    ts = sorted(t for t, _ in tmax_seen)
    if ts:
        print(f"predicted decode T_max spans {ts[0]:.0f} → {ts[-1]:.0f} tok/s "
              f"({ts[-1]/max(ts[0],1e-9):.0f}× range across the portfolio)")


if __name__ == "__main__":
    main()
