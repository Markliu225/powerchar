"""DVFS sweep: lock SM clock at a range of frequencies, run a FIXED full-occupancy
compute load, and measure power. With the active fraction of the chip held
constant, only frequency and voltage vary -> this exposes the dynamic-power law
P_dyn ∝ f·V², and since V rises ~linearly with f over the usable range, P ∝ ~f³.
This is the 'cubic' relationship — it lives in the FREQUENCY domain, not the
token-throughput domain.
"""
import csv
import subprocess
import time
import torch
from power_sampler import PowerSampler

FREQS = [600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]
LOAD_S = 3.0
WARM_S = 1.0


def sudo_smi(args):
    p = subprocess.run("echo 1 | sudo -S nvidia-smi " + args,
                       shell=True, capture_output=True, text=True)
    return p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr.strip()


def main():
    dev = "cuda"
    n = 8192
    a = torch.randn(n, n, device=dev, dtype=torch.float16)
    b = torch.randn(n, n, device=dev, dtype=torch.float16)
    sampler = PowerSampler(interval_s=0.02)
    sampler.start()
    rows = []
    try:
        for f in FREQS:
            print(sudo_smi(f"-lgc {f}"))
            # warmup at this clock. Sync each iter so the async launch queue
            # stays bounded (otherwise the backlog drains for minutes and the
            # GPU keeps running the OLD clock's work into the next setting).
            tw = time.perf_counter() + WARM_S
            while time.perf_counter() < tw:
                c = a @ b
                torch.cuda.synchronize()
            time.sleep(0.2)
            # measure
            iters = 0
            torch.cuda.synchronize()
            t0 = sampler.now(); t_end = t0 + LOAD_S
            while sampler.now() < t_end:
                c = a @ b
                torch.cuda.synchronize()
                iters += 1
            t1 = sampler.now()
            st = sampler.stats_between(t0, t1)
            tflops = iters * (2 * n ** 3) / (t1 - t0) / 1e12
            row = {"req_freq": f, "act_sm_clk": round(st["sm_clk_avg"], 0),
                   "power_avg_w": round(st["power_avg_w"], 1),
                   "power_max_w": round(st["power_max_w"], 1),
                   "tflops": round(tflops, 1), "n": st["n_samples"]}
            rows.append(row)
            print(f"  req {f:>4}MHz -> act {row['act_sm_clk']:.0f}MHz | "
                  f"{row['power_avg_w']:>6.1f}W | {row['tflops']:>5.1f} TFLOP/s")
    finally:
        print(sudo_smi("-rgc"))
        sampler.stop(); sampler.shutdown()

    with open("results_clock.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote results_clock.csv")


if __name__ == "__main__":
    main()
