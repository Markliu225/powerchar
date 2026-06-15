"""Steps 1 & 2 -- measure token throughput and GPU power for PREFILL and DECODE.

Each sweep point runs a sustained loop (~MEASURE_S) so power reaches steady
state, then throughput is computed over the *exact* synchronized wall-clock
window the power is averaged over. CUDA events cross-check the timing.

  python measure.py --phase prefill
  python measure.py --phase decode
  python measure.py --phase both      (default)
"""
from __future__ import annotations
import argparse
import csv
import gc
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config as C
from power_sampler import PowerSampler


def load_model():
    tok = AutoTokenizer.from_pretrained(C.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        C.MODEL_ID, dtype=C.DTYPE, attn_implementation=C.ATTN_IMPL).to(C.DEVICE).eval()
    return tok, model


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@torch.no_grad()
def run_prefill_point(model, sampler, batch, seq_len, vocab):
    """Repeated full forwards over (batch x seq_len) tokens, use_cache=False."""
    ids = torch.randint(0, vocab, (batch, seq_len), device=C.DEVICE, dtype=torch.long)

    t_warm_end = time.perf_counter() + C.WARMUP_S
    while time.perf_counter() < t_warm_end:
        out = model(input_ids=ids, use_cache=False, logits_to_keep=1)
        del out
    torch.cuda.synchronize()
    time.sleep(C.SETTLE_S)

    iters = 0
    torch.cuda.synchronize()
    t0 = sampler.now()
    t_end = t0 + C.MEASURE_S
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    while sampler.now() < t_end:
        out = model(input_ids=ids, use_cache=False, logits_to_keep=1)
        del out
        iters += 1
    ev1.record()
    torch.cuda.synchronize()
    t1 = sampler.now()

    wall_s = t1 - t0
    tokens = iters * batch * seq_len
    stats = sampler.stats_between(t0, t1) or {}
    p = stats.get("power_avg_w", float("nan"))
    return {
        "phase": "prefill", "batch": batch, "seq_len": seq_len,
        "load_tokens": batch * seq_len, "iters": iters,
        "wall_s": wall_s, "cuda_s": ev0.elapsed_time(ev1) / 1000.0,
        "throughput_tok_s": tokens / wall_s,
        "tok_per_joule": (tokens / wall_s) / p if p == p and p > 0 else float("nan"),
        **stats,
    }


@torch.no_grad()
def _seed_kv(model, batch, ctx_len, vocab):
    """Build a `ctx_len`-token KV cache in small chunks (so high batch fits)."""
    kv, nxt, pos = None, None, 0
    while pos < ctx_len:
        n = min(C.DECODE_SEED_CHUNK, ctx_len - pos)
        chunk = torch.randint(0, vocab, (batch, n), device=C.DEVICE, dtype=torch.long)
        out = model(input_ids=chunk, past_key_values=kv, use_cache=True, logits_to_keep=1)
        kv = out.past_key_values
        nxt = out.logits[:, -1:].argmax(dim=-1)
        pos += n
    return kv, nxt


@torch.no_grad()
def run_decode_point(model, sampler, batch, ctx_len, vocab):
    """Single-token autoregressive steps over `batch` sequences (steady state)."""
    kv, nxt = _seed_kv(model, batch, ctx_len, vocab)
    t_warm_end = time.perf_counter() + C.WARMUP_S
    while time.perf_counter() < t_warm_end:
        out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        nxt = out.logits[:, -1:].argmax(dim=-1)
    del kv, nxt, out
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(C.SETTLE_S)

    kv, nxt = _seed_kv(model, batch, ctx_len, vocab)
    torch.cuda.synchronize()
    steps = 0
    max_cache = ctx_len + 256
    cur_len = ctx_len
    t0 = sampler.now()
    t_end = t0 + C.MEASURE_S
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    while sampler.now() < t_end:
        out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        nxt = out.logits[:, -1:].argmax(dim=-1)
        steps += 1
        cur_len += 1
        if cur_len >= max_cache:            # keep KV length bounded
            del kv, out
            torch.cuda.empty_cache()
            kv, nxt = _seed_kv(model, batch, ctx_len, vocab)
            cur_len = ctx_len
    ev1.record()
    torch.cuda.synchronize()
    t1 = sampler.now()

    wall_s = t1 - t0
    tokens = steps * batch
    stats = sampler.stats_between(t0, t1) or {}
    p = stats.get("power_avg_w", float("nan"))
    return {
        "phase": "decode", "batch": batch, "ctx_len": ctx_len,
        "load_tokens": batch, "steps": steps,
        "wall_s": wall_s, "cuda_s": ev0.elapsed_time(ev1) / 1000.0,
        "throughput_tok_s": tokens / wall_s,
        "tok_per_joule": (tokens / wall_s) / p if p == p and p > 0 else float("nan"),
        **stats,
    }


def write_csv(rows, path):
    keys = sorted({k for r in rows for k in r})
    # keep a few descriptive columns first for readability
    front = [k for k in ("phase", "batch", "seq_len", "ctx_len", "load_tokens",
                         "throughput_tok_s", "power_avg_w", "util_gpu_avg",
                         "sm_clk_avg", "tok_per_joule") if k in keys]
    keys = front + [k for k in keys if k not in front]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prefill", "decode", "both"], default="both")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("loading model...", flush=True)
    tok, model = load_model()
    vocab = model.config.vocab_size
    print(f"loaded {C.MODEL_ID}  vocab={vocab}  "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    sampler = PowerSampler(interval_s=C.SAMPLE_INTERVAL_S)
    sampler.start()
    print(f"GPU {sampler.name}  enforced cap {sampler.power_limit_w:.0f} W", flush=True)

    if args.phase in ("prefill", "both"):
        rows = []
        s = C.PREFILL_SEQ_LEN
        print(f"\n=== PREFILL SWEEP (controlled: fixed S={s}, swept batch) ===", flush=True)
        for b in C.PREFILL_BATCHES:
            free()
            try:
                r = run_prefill_point(model, sampler, b, s, vocab)
                rows.append(r)
                print(f"  b={b:>3} s={s:>6} | {r['throughput_tok_s']:>10.0f} tok/s | "
                      f"{r.get('power_avg_w',0):>6.1f} W | util {r.get('util_gpu_avg',0):>4.0f}% | "
                      f"sm {r.get('sm_clk_avg',0):>4.0f}MHz | {r['tok_per_joule']:>6.1f} tok/J", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  b={b} s={s} -> OOM (memory ceiling), skipped", flush=True)
                free()
        write_csv(rows, os.path.join(C.RESULTS_DIR, "prefill.csv"))

    if args.phase in ("decode", "both"):
        rows = []
        print("\n=== DECODE SWEEP (memory-bandwidth-bound; swept by batch) ===", flush=True)
        for b in C.DECODE_BATCHES:
            free()
            try:
                r = run_decode_point(model, sampler, b, C.DECODE_CTX, vocab)
                rows.append(r)
                print(f"  b={b:>3} ctx={C.DECODE_CTX} | {r['throughput_tok_s']:>10.0f} tok/s | "
                      f"{r.get('power_avg_w',0):>6.1f} W | util {r.get('util_gpu_avg',0):>4.0f}% | "
                      f"sm {r.get('sm_clk_avg',0):>4.0f}MHz | {r['tok_per_joule']:>6.1f} tok/J", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  b={b} -> OOM (memory ceiling), skipped", flush=True)
                free()
        write_csv(rows, os.path.join(C.RESULTS_DIR, "decode.csv"))

    sampler.stop()
    sampler.shutdown()


if __name__ == "__main__":
    main()
