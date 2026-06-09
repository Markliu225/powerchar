"""NATURAL-DVFS throughput-vs-power: sweep the GPU POWER LIMIT and let the
GPU's own DVFS choose the clock (no clock locking). For a power-limited
compute-bound workload the GPU self-clocks to fit the budget: P = k*f^3, so
f ∝ P^(1/3), throughput ∝ f ∝ P^(1/3)  ==>  P ∝ throughput^3.

We measure real PREFILL token throughput at each power cap.
"""
import csv
import subprocess
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from power_sampler import PowerSampler

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
B, S = 8, 2048                 # power-dense prefill (binds the cap)
CAPS = [123, 126, 129, 132, 135, 138, 141, 145]
WARM = 1.5
MEAS = 3.0


def sudo_smi(args):
    subprocess.run("echo 1 | sudo -S nvidia-smi " + args, shell=True,
                   capture_output=True, text=True)


@torch.no_grad()
def prefill_tps(model, sampler, ids):
    tw = time.perf_counter() + WARM
    while time.perf_counter() < tw:
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
    iters = 0
    t0 = sampler.now(); t_end = t0 + MEAS
    while sampler.now() < t_end:
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        iters += 1
    t1 = sampler.now()
    st = sampler.stats_between(t0, t1)
    return iters * B * S / (t1 - t0), st


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
    vocab = model.config.vocab_size
    ids = torch.randint(0, vocab, (B, S), device="cuda")
    sampler = PowerSampler(interval_s=0.02); sampler.start()
    sudo_smi("-pm 1")
    rows = []
    try:
        for cap in CAPS:
            sudo_smi(f"-pl {cap}")
            time.sleep(0.6)
            tps, st = prefill_tps(model, sampler, ids)
            rows.append({"cap_w": cap, "delivered_tok_s": round(tps, 1),
                         "power_avg_w": round(st["power_avg_w"], 1),
                         "sm_clk_avg": round(st["sm_clk_avg"], 0),
                         "n": st["n_samples"]})
            print(f"  cap {cap}W | {tps:>8.0f} tok/s | "
                  f"power {st['power_avg_w']:>6.1f} W | clk {st['sm_clk_avg']:>4.0f} MHz",
                  flush=True)
    finally:
        sudo_smi("-pl 145")
        sudo_smi("-pm 0")
        sampler.stop(); sampler.shutdown()
    with open("results_powercap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in rows]
    print("wrote results_powercap.csv", flush=True)


if __name__ == "__main__":
    main()
