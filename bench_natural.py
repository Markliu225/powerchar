"""Natural sweep (NO clock locking): raise token throughput and let the GPU's
own DVFS respond. Phases isolated:
  * PREFILL curve: fix batch, sweep prompt length (prefill tokens). Decode fixed.
  * DECODE curve:  fix prompt length, sweep batch (kept in the memory-bound
                   regime). Prefill fixed.
We record the SM clock at every point to see whether DVFS naturally ramps with
throughput (the user's hypothesis) or pins to the boost ceiling.
"""
import csv
import gc
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from power_sampler import PowerSampler

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MEASURE_S = 3.0
WARMUP_S = 1.0
SETTLE_S = 0.3

PREFILL_BATCH = 8                      # fixed (>1); sweep prompt length
PREFILL_SEQS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
DECODE_CTX = 512                       # fixed prompt length (prefill fixed)
DECODE_BATCHES = [4, 8, 16, 24, 32, 48, 64]   # memory-bound regime (I < ridge~103)


def free():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


@torch.no_grad()
def prefill_point(model, sampler, B, S, vocab):
    ids = torch.randint(0, vocab, (B, S), device="cuda")
    tw = time.perf_counter() + WARMUP_S
    while time.perf_counter() < tw:
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
    time.sleep(SETTLE_S)
    iters = 0
    t0 = sampler.now(); t_end = t0 + MEASURE_S
    while sampler.now() < t_end:
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        iters += 1
    t1 = sampler.now()
    st = sampler.stats_between(t0, t1)
    return {"phase": "prefill", "batch": B, "seq_len": S,
            "throughput_tok_s": iters * B * S / (t1 - t0), **st}


@torch.no_grad()
def decode_point(model, sampler, B, ctx, vocab):
    def seed():
        kv = None; nxt = None; pos = 0
        while pos < ctx:
            n = min(32, ctx - pos)
            ch = torch.randint(0, vocab, (B, n), device="cuda")
            o = model(input_ids=ch, past_key_values=kv, use_cache=True, logits_to_keep=1)
            kv = o.past_key_values; nxt = o.logits[:, -1:].argmax(-1); pos += n
        return kv, nxt
    kv, nxt = seed()
    tw = time.perf_counter() + WARMUP_S
    while time.perf_counter() < tw:
        o = model(input_ids=nxt, past_key_values=kv, use_cache=True)
        kv = o.past_key_values; nxt = o.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
    time.sleep(SETTLE_S)
    steps = 0; cur = ctx; mx = ctx + 256
    t0 = sampler.now(); t_end = t0 + MEASURE_S
    while sampler.now() < t_end:
        o = model(input_ids=nxt, past_key_values=kv, use_cache=True)
        kv = o.past_key_values; nxt = o.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        steps += 1; cur += 1
        if cur >= mx:
            del kv, o; torch.cuda.empty_cache()
            kv, nxt = seed(); cur = ctx
    t1 = sampler.now()
    st = sampler.stats_between(t0, t1)
    return {"phase": "decode", "batch": B, "ctx_len": ctx,
            "throughput_tok_s": steps * B / (t1 - t0), **st}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
    vocab = model.config.vocab_size
    sampler = PowerSampler(interval_s=0.02); sampler.start()

    pre, dec = [], []
    print("=== PREFILL (batch=%d fixed, sweep prompt length) ===" % PREFILL_BATCH, flush=True)
    for S in PREFILL_SEQS:
        free()
        try:
            r = prefill_point(model, sampler, PREFILL_BATCH, S, vocab)
            pre.append(r)
            print(f"  b={PREFILL_BATCH} S={S:>5} | {r['throughput_tok_s']:>9.0f} tok/s | "
                  f"{r['power_avg_w']:>6.1f} W | clk {r['sm_clk_avg']:>4.0f} "
                  f"[{r['sm_clk_min']:.0f}-{r['sm_clk_max']:.0f}] MHz | util {r['util_gpu_avg']:.0f}%", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  b={PREFILL_BATCH} S={S} OOM"); free()

    print("=== DECODE (ctx=%d fixed, sweep batch) ===" % DECODE_CTX, flush=True)
    for B in DECODE_BATCHES:
        free()
        try:
            r = decode_point(model, sampler, B, DECODE_CTX, vocab)
            dec.append(r)
            print(f"  b={B:>3} ctx={DECODE_CTX} | {r['throughput_tok_s']:>8.0f} tok/s | "
                  f"{r['power_avg_w']:>6.1f} W | clk {r['sm_clk_avg']:>4.0f} "
                  f"[{r['sm_clk_min']:.0f}-{r['sm_clk_max']:.0f}] MHz | util {r['util_gpu_avg']:.0f}%", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  b={B} OOM"); free()

    sampler.stop(); sampler.shutdown()
    for name, rows in [("prefill", pre), ("decode", dec)]:
        keys = sorted({k for r in rows for k in r})
        with open(f"results_nat_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            [w.writerow(r) for r in rows]
        print(f"wrote results_nat_{name}.csv", flush=True)


if __name__ == "__main__":
    main()
