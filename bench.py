"""Prefill vs Decode: token-throughput / power curves on a real LLM.

Separately measures, as a function of offered load:
  * PREFILL: one forward over (batch x seq_len) tokens, KV cache being built.
            Compute-bound -> high power. Swept by total prefill tokens.
  * DECODE : autoregressive single-token steps reusing the KV cache.
            Memory-bandwidth-bound -> lower power. Swept by batch size.

Each measurement point runs a sustained loop (~MEASURE_S seconds) so GPU power
reaches steady state and NVML can collect many samples. Throughput is computed
over the exact synchronized wall-clock window that power is averaged over.
"""
import argparse
import csv
import gc
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from power_sampler import PowerSampler

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = torch.float16
DEVICE = "cuda"
MEASURE_S = 4.0      # sustained measurement window per point
WARMUP_S = 1.0       # warmup before measuring
SETTLE_S = 0.3       # let power settle after warmup, before window starts


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE)
    model.eval()
    return tok, model


@torch.no_grad()
def run_prefill_point(model, sampler, batch, seq_len, vocab):
    """Sustained prefill: repeated full forwards over (batch x seq_len) tokens."""
    ids = torch.randint(0, vocab, (batch, seq_len), device=DEVICE, dtype=torch.long)
    # logits_to_keep=1: project only the last position to vocab. This both
    # avoids materializing a (batch,seq_len,151936) fp16 logits tensor (which
    # OOMs an 8GB card past ~12k tokens) AND matches real prefill, where only
    # the last token's logits are needed to emit the first output token.
    # warmup
    t_warm_end = time.perf_counter() + WARMUP_S
    while time.perf_counter() < t_warm_end:
        out = model(input_ids=ids, use_cache=False, logits_to_keep=1)
        del out
    torch.cuda.synchronize()
    time.sleep(SETTLE_S)

    # measure
    iters = 0
    torch.cuda.synchronize()
    t0 = sampler.now()
    t_end = t0 + MEASURE_S
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
    cuda_s = ev0.elapsed_time(ev1) / 1000.0
    tokens = iters * batch * seq_len
    stats = sampler.stats_between(t0, t1)
    return {
        "phase": "prefill", "batch": batch, "seq_len": seq_len,
        "tokens_per_iter": batch * seq_len, "iters": iters,
        "total_tokens": tokens, "wall_s": wall_s, "cuda_s": cuda_s,
        "throughput_tok_s": tokens / wall_s,
        **(stats or {}),
    }


@torch.no_grad()
def run_decode_point(model, sampler, batch, ctx_len, vocab):
    """Sustained decode: single-token autoregressive steps over `batch` seqs.

    The KV cache is seeded with `ctx_len` tokens fed in SMALL chunks. A one-shot
    prefill of (batch x ctx_len) tokens spikes activation memory and OOMs at high
    batch on 8GB; chunking keeps peak activation at (batch x CHUNK) so the decode
    batch is limited only by KV-cache size, not by setup activations. Chunking the
    setup does not affect the decode-step measurement at all (the steps are
    identical regardless of how the cache was built).
    """
    CHUNK = 32

    def seed_kv():
        kv = None
        pos = 0
        nxt = None
        while pos < ctx_len:
            n = min(CHUNK, ctx_len - pos)
            chunk = torch.randint(0, vocab, (batch, n), device=DEVICE, dtype=torch.long)
            out = model(input_ids=chunk, past_key_values=kv, use_cache=True,
                        logits_to_keep=1)
            kv = out.past_key_values
            nxt = out.logits[:, -1:].argmax(dim=-1)
            pos += n
        return kv, nxt

    # warmup (seed once + several decode steps)
    kv, nxt = seed_kv()
    t_warm_end = time.perf_counter() + WARMUP_S
    while time.perf_counter() < t_warm_end:
        out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        nxt = out.logits[:, -1:].argmax(dim=-1)
    del kv, nxt, out
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(SETTLE_S)

    # measure pure decode. Rebuild KV periodically so cache length stays bounded.
    kv, nxt = seed_kv()
    torch.cuda.synchronize()
    steps = 0
    max_cache = ctx_len + 256
    cur_len = ctx_len
    t0 = sampler.now()
    t_end = t0 + MEASURE_S
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    while sampler.now() < t_end:
        out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        nxt = out.logits[:, -1:].argmax(dim=-1)
        steps += 1
        cur_len += 1
        if cur_len >= max_cache:
            del kv, out
            torch.cuda.empty_cache()
            kv, nxt = seed_kv()
            cur_len = ctx_len
    ev1.record()
    torch.cuda.synchronize()
    t1 = sampler.now()

    wall_s = t1 - t0
    cuda_s = ev0.elapsed_time(ev1) / 1000.0
    tokens = steps * batch  # one token per sequence per step
    stats = sampler.stats_between(t0, t1)
    return {
        "phase": "decode", "batch": batch, "ctx_len": ctx_len,
        "tokens_per_step": batch, "steps": steps,
        "total_tokens": tokens, "wall_s": wall_s, "cuda_s": cuda_s,
        "throughput_tok_s": tokens / wall_s,
        **(stats or {}),
    }


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("loading model...", flush=True)
    tok, model = load_model()
    vocab = model.config.vocab_size
    print(f"model loaded. vocab={vocab} "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    sampler = PowerSampler(interval_s=0.02)
    sampler.start()

    # ---- PREFILL SWEEP: vary total tokens per forward (batch x seq_len) ----
    # Sweep offered load from a single short prompt (light) up to tens of
    # thousands of tokens per forward (compute-saturated). With logits_to_keep=1
    # the ceiling is set by activations: ~64k tokens fits in 8GB, beyond OOMs
    # (caught and skipped, which usefully marks the saturation ceiling).
    prefill_grid = [
        (1, 64), (1, 128), (1, 256), (1, 512), (1, 1024), (1, 2048),
        (1, 4096), (1, 8192), (2, 8192), (4, 8192), (4, 4096),
        (8, 4096), (2, 16384), (4, 16384), (8, 8192), (16, 4096),
        (8, 16384), (16, 8192),
    ]
    # ---- DECODE SWEEP: vary batch size (concurrent seqs) at fixed context ----
    # KV cache (28 layers x K&V x 2 kv-heads x 128) at max_cache=512 is
    # ~14.7MB/seq; b=256 (~3.8GB) fits in 8GB with the 3GB weights.
    decode_batches = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256]
    decode_ctx = 256

    prefill_rows, decode_rows = [], []

    print("\n=== PREFILL SWEEP ===", flush=True)
    for (b, s) in prefill_grid:
        free()
        try:
            r = run_prefill_point(model, sampler, b, s, vocab)
            prefill_rows.append(r)
            print(f"  prefill b={b:>3} s={s:>6} | "
                  f"{r['throughput_tok_s']:>10.0f} tok/s | "
                  f"{r.get('power_avg_w',0):>6.1f} W | "
                  f"util {r.get('util_gpu_avg',0):>4.0f}% | "
                  f"sm {r.get('sm_clk_avg',0):>4.0f}MHz | "
                  f"iters {r['iters']}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  prefill b={b} s={s} -> OOM, skipped", flush=True)
            free()

    print("\n=== DECODE SWEEP ===", flush=True)
    for b in decode_batches:
        free()
        try:
            r = run_decode_point(model, sampler, b, decode_ctx, vocab)
            decode_rows.append(r)
            print(f"  decode  b={b:>3} ctx={decode_ctx} | "
                  f"{r['throughput_tok_s']:>10.0f} tok/s | "
                  f"{r.get('power_avg_w',0):>6.1f} W | "
                  f"util {r.get('util_gpu_avg',0):>4.0f}% | "
                  f"sm {r.get('sm_clk_avg',0):>4.0f}MHz | "
                  f"steps {r['steps']}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  decode b={b} -> OOM, skipped", flush=True)
            free()

    sampler.stop()
    sampler.shutdown()

    # write CSVs
    for name, rows in [("prefill", prefill_rows), ("decode", decode_rows)]:
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        path = f"{args.out}_{name}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
