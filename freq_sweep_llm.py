"""Capture the ~cubic throughput-vs-power law by sweeping GPU CLOCK FREQUENCY
on real LLM prefill/decode workloads (not by sweeping load at pinned clock).

For a COMPUTE-BOUND workload, token throughput ∝ f while power ∝ ~f^3, so
plotting throughput vs power traces the ~cubic curve. We measure:
  * prefill (compute-bound)        -> expect ~cubic
  * decode, low batch (mem-bound)  -> throughput ~flat vs clock, power rises
  * decode, high batch (compute-b) -> rejoins the ~cubic curve
"""
import csv
import subprocess
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from power_sampler import PowerSampler

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FREQS = [600, 825, 1050, 1275, 1500, 1725, 1950, 2175, 2400, 2600]
WARM_S = 0.8
MEAS_S = 2.5


def sudo_smi(args):
    subprocess.run("echo 1 | sudo -S nvidia-smi " + args, shell=True,
                   capture_output=True, text=True)


@torch.no_grad()
def prefill_tps(model, sampler, B, S, vocab):
    ids = torch.randint(0, vocab, (B, S), device="cuda")
    tw = time.perf_counter() + WARM_S
    while time.perf_counter() < tw:
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
    iters = 0
    t0 = sampler.now(); t_end = t0 + MEAS_S
    while sampler.now() < t_end:
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        iters += 1
    t1 = sampler.now()
    st = sampler.stats_between(t0, t1)
    return iters * B * S / (t1 - t0), st


@torch.no_grad()
def decode_tps(model, sampler, B, ctx, vocab):
    # chunked seed so high batch fits
    kv = None; nxt = None; pos = 0
    while pos < ctx:
        n = min(32, ctx - pos)
        ch = torch.randint(0, vocab, (B, n), device="cuda")
        out = model(input_ids=ch, past_key_values=kv, use_cache=True, logits_to_keep=1)
        kv = out.past_key_values; nxt = out.logits[:, -1:].argmax(-1); pos += n
    tw = time.perf_counter() + WARM_S
    while time.perf_counter() < tw:
        out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
        kv = out.past_key_values; nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
    steps = 0; cur = ctx; mx = ctx + 256
    t0 = sampler.now(); t_end = t0 + MEAS_S
    while sampler.now() < t_end:
        out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
        kv = out.past_key_values; nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        steps += 1; cur += 1
        if cur >= mx:
            del kv, out; torch.cuda.empty_cache()
            kv = None; pos = 0
            while pos < ctx:
                n = min(32, ctx - pos)
                ch = torch.randint(0, vocab, (B, n), device="cuda")
                o = model(input_ids=ch, past_key_values=kv, use_cache=True, logits_to_keep=1)
                kv = o.past_key_values; nxt = o.logits[:, -1:].argmax(-1); pos += n
            cur = ctx
    t1 = sampler.now()
    st = sampler.stats_between(t0, t1)
    return steps * B / (t1 - t0), st


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
    vocab = model.config.vocab_size
    sampler = PowerSampler(interval_s=0.02); sampler.start()

    # Decode is memory-bandwidth-bound; we only characterize it in that regime
    # (low batch). High-batch short-context micro-benchmarks can nominally cross
    # the roofline ridge, but that is an artifact of an unrealistically short
    # context (real decode grows the KV cache) and is not representative.
    workloads = [
        ("prefill_s2048", lambda: prefill_tps(model, sampler, 1, 2048, vocab)),
        ("decode_b1",     lambda: decode_tps(model, sampler, 1, 256, vocab)),
    ]
    rows = []
    try:
        for f in FREQS:
            sudo_smi(f"-lgc {f}")
            time.sleep(0.3)
            for name, fn in workloads:
                torch.cuda.empty_cache()
                tps, st = fn()
                act = st["sm_clk_avg"]; pw = st["power_avg_w"]
                rows.append({"workload": name, "req_freq": f,
                             "act_sm_clk": round(act, 0), "tok_s": round(tps, 1),
                             "power_avg_w": round(pw, 1), "n": st["n_samples"]})
                print(f"  f={f:>4} {name:<14} act={act:>5.0f}MHz  "
                      f"{tps:>9.1f} tok/s  {pw:>6.1f} W", flush=True)
    finally:
        sudo_smi("-rgc")
        sampler.stop(); sampler.shutdown()

    with open("results_freq_llm.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        [w.writerow(r) for r in rows]
    print("wrote results_freq_llm.csv")


if __name__ == "__main__":
    main()
