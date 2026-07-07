"""Isolate the chat-decode plateau discrepancy (v1: 739 tok/s on GPU0 vs v2: 820 tok/s on GPU2).

Two candidate causes, tested factorially at the uncapped 250 W point:
  A. KV seed chunk (v1 seeded the cache in chunks of 32, v2 uses 256) -> memory-layout effect?
  B. GPU silicon (different card -> different effective HBM bandwidth)?

Run this on EACH GPU of interest:  CUDA_VISIBLE_DEVICES=g PYTHONPATH=../../code python3 ab_chat_probe.py
It measures the chat decode point (Phi-3, C=256, B=64) twice per seed-chunk setting {32, 256}
at the default 250 W cap and prints a compact verdict line per setting.
"""
from __future__ import annotations
import os, time
import torch

import config as C
C.WARMUP_S = 0.5; C.SETTLE_S = 0.1; C.MEASURE_S = 2.5
C.MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

from power_sampler import PowerSampler
from measure import load_model, run_decode_point, free

GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
B, CTX = 64, 256


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, model = load_model()
    vocab = model.config.vocab_size
    sampler = PowerSampler(interval_s=C.SAMPLE_INTERVAL_S); sampler.start(); time.sleep(0.3)
    print(f"GPU{GPU} probe: chat decode C={CTX} B={B} at default cap", flush=True)
    for chunk in (32, 64, 128, 256):
        C.DECODE_SEED_CHUNK = chunk
        ts = []
        for _ in range(2):
            free()
            r = run_decode_point(model, sampler, B, CTX, vocab, target_steps=48)
            ts.append(r["throughput_tok_s"])
        print(f"  seed_chunk={chunk:>3}: {ts[0]:.0f} / {ts[1]:.0f} tok/s "
              f"(clk {r.get('sm_clk_avg',0):.0f}, P {r.get('power_avg_w',0):.0f}W, "
              f"util {r.get('util_gpu_avg',0):.0f}%)", flush=True)

    # decay diagnosis: with the WORST layout, does per-step time grow with steps (fragment
    # accumulation) or is it constant (static layout penalty)?
    from measure import _seed_kv
    C.DECODE_SEED_CHUNK = 32
    free()
    kv, nxt = _seed_kv(model, B, CTX, vocab)
    torch.cuda.synchronize()
    import time as _t
    print("  decay probe (chunk=32): per-16-step segment tok/s:", flush=True)
    seg = []
    for s in range(6):
        t0 = _t.perf_counter()
        for _ in range(16):
            out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            nxt = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()
        seg.append(16 * B / (_t.perf_counter() - t0))
    print("   ", " ".join(f"{x:.0f}" for x in seg), flush=True)
    sampler.stop(); sampler.shutdown()


if __name__ == "__main__":
    main()
