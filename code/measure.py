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
    # transformers renamed torch_dtype -> dtype across versions; some mid versions silently
    # IGNORE the unknown kwarg (model lands in fp32 -> 2x memory, wrong numbers). Try both
    # spellings, then enforce the dtype unconditionally.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            C.MODEL_ID, dtype=C.DTYPE, attn_implementation=C.ATTN_IMPL)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            C.MODEL_ID, torch_dtype=C.DTYPE, attn_implementation=C.ATTN_IMPL)
    if next(model.parameters()).dtype != C.DTYPE:
        model = model.to(C.DTYPE)
    model = model.to(C.DEVICE).eval()
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
def run_decode_point(model, sampler, batch, ctx_len, vocab,
                     target_steps=None, min_s=None, max_s=45.0):
    """Single-token autoregressive steps over `batch` sequences (steady state).

    Default (target_steps=None): fixed MEASURE_S wall-clock window (legacy behaviour).
    With target_steps=N: run until BOTH N decode steps AND min_s (default MEASURE_S) have
    elapsed, capped at max_s. This kills the step-quantization noise of slow points (a
    6 tok/s @ b4 point does ~4 steps in 2.5 s -> +-25% quantization).

    NO RESEED EVER HAPPENS INSIDE THE TIMED WINDOW in target_steps mode: on a fast GPU a
    point can exceed target_steps long before min_s elapses, and a mid-window reseed would
    inject an uncredited ctx-length prefill into the measured interval (biasing throughput
    low). Instead the warmup measures the step rate, the headroom is sized to the expected
    in-window steps, and the KV is simply allowed to grow ctx -> ctx+steps; the real,
    traffic-weighted context is reported as ctx_eff = ctx + steps/2 and used downstream."""
    kv, nxt = _seed_kv(model, batch, ctx_len, vocab)
    t_warm_end = time.perf_counter() + C.WARMUP_S
    warm_steps, t_warm0 = 0, time.perf_counter()
    while time.perf_counter() < t_warm_end:
        out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        nxt = out.logits[:, -1:].argmax(dim=-1)
        warm_steps += 1
    torch.cuda.synchronize()
    warm_rate = warm_steps / max(time.perf_counter() - t_warm0, 1e-6)   # steps/s at this cap
    del kv, nxt, out
    torch.cuda.empty_cache()
    time.sleep(C.SETTLE_S)

    kv, nxt = _seed_kv(model, batch, ctx_len, vocab)
    torch.cuda.synchronize()
    steps, reseeds = 0, 0
    min_s = C.MEASURE_S if min_s is None else min_s
    if target_steps:
        # expected steps in the window = rate x min_s; headroom sized so the cache never hits
        # the bound mid-window (the +25% margin covers warmup-rate underestimation)
        expect = max(target_steps, int(warm_rate * min_s * 1.25) + 8)
        max_cache = ctx_len + expect + 16
    else:
        max_cache = ctx_len + int(os.environ.get("DECODE_KV_HEADROOM", "256"))
    cur_len = ctx_len
    t0 = sampler.now()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    while True:
        now = sampler.now()
        if target_steps is None:
            if now >= t0 + min_s:
                break
        else:
            if (steps >= target_steps and now >= t0 + min_s) or now >= t0 + max_s:
                break
        out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        nxt = out.logits[:, -1:].argmax(dim=-1)
        steps += 1
        cur_len += 1
        if cur_len >= max_cache:            # keep KV length bounded (should NOT fire in target mode)
            del kv, out
            torch.cuda.empty_cache()
            kv, nxt = _seed_kv(model, batch, ctx_len, vocab)
            cur_len = ctx_len
            reseeds += 1
    ev1.record()
    torch.cuda.synchronize()
    t1 = sampler.now()

    wall_s = t1 - t0
    tokens = steps * batch
    stats = sampler.stats_between(t0, t1) or {}
    p = stats.get("power_avg_w", float("nan"))
    return {
        "phase": "decode", "batch": batch, "ctx_len": ctx_len,
        # traffic-weighted effective context: the cache grows ctx -> ctx+steps within the
        # window (cycling back at the headroom bound if it reseeds), so the average KV read
        # during the window corresponds to ~ctx + grown/2 tokens, not the nominal ctx.
        "ctx_eff": ctx_len + min(steps, max_cache - ctx_len) / 2.0,
        # >0 in target mode means the headroom estimate was beaten and the point carries an
        # uncredited in-window prefill -> treat with suspicion / re-measure
        "reseeds": reseeds,
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

    per_pt = C.WARMUP_S + C.SETTLE_S + C.MEASURE_S
    if args.phase in ("prefill", "both"):
        rows = []
        s = C.PREFILL_SEQ_LEN
        n = len(C.PREFILL_BATCHES)
        print(f"\n=== PREFILL SWEEP (controlled: fixed S={s}, swept batch) ===", flush=True)
        t_phase = time.perf_counter()
        for i, b in enumerate(C.PREFILL_BATCHES, 1):
            free()
            print(f"  [{i:>2}/{n}] > prefill b={b} s={s}  "
                  f"(~{per_pt:.0f}s/pt, elapsed {time.perf_counter()-t_phase:>3.0f}s)", flush=True)
            try:
                r = run_prefill_point(model, sampler, b, s, vocab)
                rows.append(r)
                print(f"  [{i:>2}/{n}] = b={b:>3} s={s:>6} | {r['throughput_tok_s']:>10.0f} tok/s | "
                      f"{r.get('power_avg_w',0):>6.1f} W | util {r.get('util_gpu_avg',0):>4.0f}% | "
                      f"sm {r.get('sm_clk_avg',0):>4.0f}MHz | {r['tok_per_joule']:>6.1f} tok/J", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  [{i:>2}/{n}] b={b} s={s} -> OOM (memory ceiling), skipped", flush=True)
                free()
        write_csv(rows, os.path.join(C.RESULTS_DIR, "prefill.csv"))

    if args.phase in ("decode", "both"):
        rows = []
        n = len(C.DECODE_BATCHES)
        print("\n=== DECODE SWEEP (memory-bandwidth-bound; swept by batch) ===", flush=True)
        t_phase = time.perf_counter()
        for i, b in enumerate(C.DECODE_BATCHES, 1):
            free()
            print(f"  [{i:>2}/{n}] > decode b={b} ctx={C.DECODE_CTX}  "
                  f"(~{per_pt:.0f}s/pt, elapsed {time.perf_counter()-t_phase:>3.0f}s)", flush=True)
            try:
                r = run_decode_point(model, sampler, b, C.DECODE_CTX, vocab)
                rows.append(r)
                print(f"  [{i:>2}/{n}] = b={b:>3} ctx={C.DECODE_CTX} | {r['throughput_tok_s']:>10.0f} tok/s | "
                      f"{r.get('power_avg_w',0):>6.1f} W | util {r.get('util_gpu_avg',0):>4.0f}% | "
                      f"sm {r.get('sm_clk_avg',0):>4.0f}MHz | {r['tok_per_joule']:>6.1f} tok/J", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  [{i:>2}/{n}] b={b} -> OOM (memory ceiling), skipped", flush=True)
                free()
        write_csv(rows, os.path.join(C.RESULTS_DIR, "decode.csv"))

    sampler.stop()
    sampler.shutdown()


if __name__ == "__main__":
    main()
