"""Jitter-free measurement WITHOUT clock-lock permission (cold-start method).

We cannot lock the clock (account lacks sudo), and this V100's clock is a thermal
feedback loop: under sustained load it throttles toward an ~82 C ceiling, and the
controller hunts (jitter). Two ideas remove the jitter from user space:

  1. COLD-START + SHORT WINDOW. Cool to a fixed baseline temperature before every
     point, then measure a brief window *before* the die heats past ~80 C. In that
     window the clock sits at its un-throttled boost value and barely moves, and
     because every point starts from the same temperature the result is
     reproducible (no dependence on what ran before -> no sweep-order confound).
     We record temp at window start/end and the clock spread so it is auditable.

  2. KEEP THE GPU FED IN DECODE. Single-token decode is bursty: syncing every step
     leaves idle gaps between kernels, and the clock pumps up/down across them.
     Queuing DECODE_SYNC_EVERY steps before a sync keeps the pipeline full, so the
     clock stays steady. Throughput is still exact (counted at sync boundaries over
     the measured wall-clock window).

Randomised point order + median aggregation. Writes results/prefill.csv and
results/decode.csv (schema compatible with analyze.py / plot_pt.py).
  python code/measure_stable.py --phase both
"""
from __future__ import annotations
import argparse
import os
import random
import time
import torch

import config as C
from power_sampler import PowerSampler
from measure import load_model, _seed_kv, free, write_csv

COOL_TARGET_C = 62.0      # cool to this before each point (common cold baseline)
COOL_MAX_S    = 35.0      # cap the cooldown wait
WARMUP_S      = 0.5       # brief: ramp the clock + prime caches
MEASURE_S     = 3.0       # short window -> stays un-throttled (< ~80 C)
DECODE_SYNC_EVERY = 16    # queue this many decode steps between syncs (keep GPU fed)
SEED = 0

PREFILL_BATCHES = [1, 2, 4, 8, 16, 32, 64]
DECODE_BATCHES  = [1, 2, 4, 8, 16, 32, 64]


def _temp(sampler):
    return sampler.samples[-1]["temp"] if sampler.samples else 0.0


def cooldown(sampler, tag):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < COOL_MAX_S:
        if _temp(sampler) <= COOL_TARGET_C:
            break
        time.sleep(0.5)
    return _temp(sampler), time.perf_counter() - t0


@torch.no_grad()
def prefill_steady(model, sampler, B, S, vocab):
    ids = torch.randint(0, vocab, (B, S), device=C.DEVICE, dtype=torch.long)
    t_warm = time.perf_counter() + WARMUP_S
    while time.perf_counter() < t_warm:
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
    iters = 0
    temp0 = _temp(sampler)
    t0 = sampler.now(); t_end = t0 + MEASURE_S
    while sampler.now() < t_end:
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(); iters += 1
    t1 = sampler.now()
    st = sampler.stats_between(t0, t1) or {}
    tput = iters * B * S / (t1 - t0)
    p = st.get("power_avg_w", float("nan"))
    return {"phase": "prefill", "batch": B, "seq_len": S, "load_tokens": B * S,
            "iters": iters, "wall_s": t1 - t0, "throughput_tok_s": tput,
            "temp_start": temp0, "temp_end": _temp(sampler),
            "tok_per_joule": tput / p if p == p and p > 0 else float("nan"), **st}


@torch.no_grad()
def decode_steady(model, sampler, B, ctx, vocab):
    kv, nxt = _seed_kv(model, B, ctx, vocab)          # window is short -> no re-seed needed
    cur = [kv, nxt]

    def queue_steps(k):
        for _ in range(k):
            o = model(input_ids=cur[1], past_key_values=cur[0], use_cache=True)
            cur[0] = o.past_key_values
            cur[1] = o.logits[:, -1:].argmax(dim=-1)

    t_warm = time.perf_counter() + WARMUP_S
    while time.perf_counter() < t_warm:
        queue_steps(DECODE_SYNC_EVERY); torch.cuda.synchronize()
    steps = 0
    temp0 = _temp(sampler)
    t0 = sampler.now(); t_end = t0 + MEASURE_S
    while sampler.now() < t_end:
        queue_steps(DECODE_SYNC_EVERY)
        torch.cuda.synchronize()
        steps += DECODE_SYNC_EVERY
    t1 = sampler.now()
    st = sampler.stats_between(t0, t1) or {}
    tput = steps * B / (t1 - t0)
    p = st.get("power_avg_w", float("nan"))
    return {"phase": "decode", "batch": B, "ctx_len": ctx, "load_tokens": B,
            "steps": steps, "wall_s": t1 - t0, "throughput_tok_s": tput,
            "temp_start": temp0, "temp_end": _temp(sampler),
            "tok_per_joule": tput / p if p == p and p > 0 else float("nan"), **st}


def run_phase(name, batches, point_fn, model, sampler, fixed, vocab):
    order = list(batches)
    random.Random(SEED).shuffle(order)
    n = len(order)
    print(f"\n=== {name.upper()} (cold-start, un-throttled; randomised order {order}) ===", flush=True)
    rows = []
    t_phase = time.perf_counter()
    for i, b in enumerate(order, 1):
        free()
        ct, cs = cooldown(sampler, b)
        print(f"  [{i:>2}/{n}] > {name} b={b} | cooled to {ct:.0f}C in {cs:.0f}s | "
              f"warm {WARMUP_S}s meas {MEASURE_S}s | elapsed {time.perf_counter()-t_phase:>3.0f}s", flush=True)
        try:
            r = point_fn(model, sampler, b, fixed, vocab)
            rows.append(r)
            print(f"  [{i:>2}/{n}] = b={b:>3} | {r['throughput_tok_s']:>9.0f} tok/s | "
                  f"{r.get('power_avg_w',0):>6.1f} W (p50 {r.get('power_p50_w',0):>6.1f}, "
                  f"std {r.get('power_std_w',0):>4.1f}) | clk {r.get('sm_clk_avg',0):>4.0f}"
                  f"[{r.get('sm_clk_min',0):.0f}-{r.get('sm_clk_max',0):.0f}] | "
                  f"{r['temp_start']:.0f}->{r['temp_end']:.0f}C | {r['tok_per_joule']:>6.1f} tok/J", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{i:>2}/{n}] b={b} -> OOM, skipped", flush=True); free()
    rows.sort(key=lambda r: r["batch"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prefill", "decode", "both"], default="both")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("loading model...", flush=True)
    tok, model = load_model()
    vocab = model.config.vocab_size
    sampler = PowerSampler(interval_s=C.SAMPLE_INTERVAL_S)
    sampler.start(); time.sleep(0.5)
    print(f"loaded {C.MODEL_ID} | GPU {sampler.name} cap {sampler.power_limit_w:.0f} W", flush=True)

    if args.phase in ("prefill", "both"):
        rows = run_phase("prefill", PREFILL_BATCHES, prefill_steady, model, sampler,
                         C.PREFILL_SEQ_LEN, vocab)
        write_csv(rows, os.path.join(C.RESULTS_DIR, "prefill.csv"))
    if args.phase in ("decode", "both"):
        rows = run_phase("decode", DECODE_BATCHES, decode_steady, model, sampler,
                         C.DECODE_CTX, vocab)
        write_csv(rows, os.path.join(C.RESULTS_DIR, "decode.csv"))

    sampler.stop(); sampler.shutdown()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
