"""Prefill throughput-vs-power curve traced from LOW throughput upward, by
sweeping prompt length from very short to long. Throughput = tokens/latency, so
short prompts give low throughput (few tokens per forward); it rises as the
prompt grows and the GPU fills up. Done for batch=1 (reaches ~hundreds tok/s)
and batch=8 (higher floor) to show the parallelism effect."""
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from power_sampler import PowerSampler
import bench_natural as bn

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEQS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
        1024, 1536, 2048, 3072, 4096]
BATCHES = [1, 8]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
    vocab = model.config.vocab_size
    sampler = PowerSampler(interval_s=0.02); sampler.start()
    rows = []
    for B in BATCHES:
        print(f"=== PREFILL batch={B} ===", flush=True)
        for S in SEQS:
            bn.free()
            try:
                r = bn.prefill_point(model, sampler, B, S, vocab)
                rows.append(r)
                print(f"  b={B} S={S:>5} | {r['throughput_tok_s']:>9.0f} tok/s | "
                      f"{r['power_avg_w']:>6.1f} W | clk {r['sm_clk_avg']:>4.0f} MHz | "
                      f"util {r['util_gpu_avg']:>3.0f}%", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  b={B} S={S} OOM"); bn.free()
    sampler.stop(); sampler.shutdown()
    keys = sorted({k for r in rows for k in r})
    with open("results_prefill_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        [w.writerow(r) for r in rows]
    print("wrote results_prefill_curve.csv", flush=True)


if __name__ == "__main__":
    main()
