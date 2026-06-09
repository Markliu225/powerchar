"""Prefill throughput-vs-power as a function of DELIVERED tokens/s (rate-limited).

Instead of running prefills back-to-back (which pins the GPU at 100% busy and
inflates the low-throughput power floor), we issue prefills at a controlled rate
and let the GPU idle between them. Low delivered tokens/s => GPU mostly idle =>
its own DVFS down-clocks => low average power. As the rate rises toward the
GPU's peak, idle shrinks and power climbs to the cap. No clock locking.

throughput axis = delivered tokens/s (the controlled load).
"""
import csv
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from power_sampler import PowerSampler

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
B, S = 1, 256                 # one 256-token prompt per prefill event
TOKENS = B * S
WINDOW = 6.0
WARM = 2.0


@torch.no_grad()
def one_prefill(model, ids):
    model(input_ids=ids, use_cache=False, logits_to_keep=1)
    torch.cuda.synchronize()


def run_rate(model, sampler, ids, target_tps):
    """Issue prefills to deliver ~target_tps tokens/s; measure avg power."""
    period = TOKENS / target_tps          # seconds between prefill starts
    # warmup at this duty cycle so clock/power settle
    tend = time.perf_counter() + WARM
    tnext = time.perf_counter()
    while time.perf_counter() < tend:
        one_prefill(model, ids)
        tnext += period
        slp = tnext - time.perf_counter()
        if slp > 0:
            time.sleep(slp)
    # measure
    events = 0
    t0 = sampler.now(); t_end = t0 + WINDOW
    tnext = time.perf_counter()
    while sampler.now() < t_end:
        one_prefill(model, ids)
        events += 1
        tnext += period
        slp = tnext - time.perf_counter()
        if slp > 0:
            time.sleep(slp)
    t1 = sampler.now()
    st = sampler.stats_between(t0, t1)
    delivered = events * TOKENS / (t1 - t0)
    return delivered, st


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
    vocab = model.config.vocab_size
    ids = torch.randint(0, vocab, (B, S), device="cuda")
    sampler = PowerSampler(interval_s=0.02); sampler.start()

    # measure peak (back-to-back)
    for _ in range(20):
        one_prefill(model, ids)
    t0 = sampler.now(); n = 0; t_end = t0 + 2.0
    while sampler.now() < t_end:
        one_prefill(model, ids); n += 1
    peak = n * TOKENS / (sampler.now() - t0)
    print(f"peak (saturated) throughput ≈ {peak:.0f} tok/s", flush=True)

    targets = [0.05, 0.08, 0.12, 0.18, 0.27, 0.38, 0.52, 0.68, 0.85, 1.0]
    rows = []
    for fr in targets:
        tgt = fr * peak
        delivered, st = run_rate(model, sampler, ids, tgt)
        rows.append({"target_frac": fr, "delivered_tok_s": round(delivered, 1),
                     "power_avg_w": round(st["power_avg_w"], 1),
                     "sm_clk_avg": round(st["sm_clk_avg"], 0),
                     "util_gpu_avg": round(st["util_gpu_avg"], 1),
                     "n": st["n_samples"]})
        print(f"  target {fr*100:>3.0f}% | delivered {delivered:>8.0f} tok/s | "
              f"{st['power_avg_w']:>6.1f} W | clk {st['sm_clk_avg']:>4.0f} MHz | "
              f"util {st['util_gpu_avg']:>3.0f}%", flush=True)

    sampler.stop(); sampler.shutdown()
    with open("results_rate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in rows]
    print("wrote results_rate.csv", flush=True)


if __name__ == "__main__":
    main()
