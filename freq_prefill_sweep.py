"""Fixed large prefill input; sweep GPU CLOCK (which sets the power level);
measure real prefill tok/s at each clock; output (power, tok/s) pairs.

  fixed workload  -> occupancy constant
  clock f varies  -> power = static + k*f*V(f)^2  (V rises with f)  -> ~cubic
  tok/s ∝ f       -> so power vs tok/s traces the cubic operating curve
"""
import csv
import subprocess
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from power_sampler import PowerSampler

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
B, S = 8, 2048                 # fixed LARGE prefill input (16384 tokens/forward)
FREQS = [510, 690, 870, 1050, 1230, 1410, 1590, 1770, 1950, 2130, 2310, 2490, 2600]
WARM = 1.0
MEAS = 3.0


def sudo_smi(args):
    subprocess.run("echo 1 | sudo -S nvidia-smi " + args, shell=True,
                   capture_output=True, text=True)


@torch.no_grad()
def prefill_tps(model, sampler, ids):
    # sync each iter so the async queue stays bounded at low clocks
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
        for f in FREQS:
            sudo_smi(f"-lgc {f}")
            time.sleep(0.4)
            tps, st = prefill_tps(model, sampler, ids)
            rows.append({"req_freq": f, "act_clk": round(st["sm_clk_avg"], 0),
                         "tok_s": round(tps, 1),
                         "power_w": round(st["power_avg_w"], 1),
                         "n": st["n_samples"]})
            print(f"  clk {f:>4}MHz (act {st['sm_clk_avg']:>4.0f}) | "
                  f"{tps:>9.0f} tok/s | {st['power_avg_w']:>6.1f} W", flush=True)
    finally:
        sudo_smi("-rgc")
        sudo_smi("-pm 0")
        sampler.stop(); sampler.shutdown()
    with open("results_freq_prefill.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in rows]
    print("wrote results_freq_prefill.csv", flush=True)


if __name__ == "__main__":
    main()
