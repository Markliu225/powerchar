"""Diagnostic: at each locked clock, print decode power as MEAN vs MEDIAN vs MAX vs nvidia-smi.

The eager decode loop has GPU-idle gaps (Python between single-token steps), so the window MEAN
under-reads vs the active/sustained power nvidia-smi shows. This tells us which estimator matches.
"""
from __future__ import annotations
import os, subprocess, time
import config as C
C.WARMUP_S = 0.4; C.SETTLE_S = 0.1; C.MEASURE_S = 2.5
import torch
import pynvml
from power_sampler import PowerSampler
from measure import load_model, run_decode_point, free

PW = os.environ.get("SUDO_PASS", "")
GPU = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")


def sudo(a):
    return subprocess.run(["sudo", "-S", "-p", ""] + a, input=PW + "\n", text=True, capture_output=True)


def smi_power():
    r = subprocess.run(["nvidia-smi", "-i", GPU, "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip().split("\n")[0])
    except Exception:
        return float("nan")


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, model = load_model(); vocab = model.config.vocab_size
    s = PowerSampler(interval_s=0.01); s.start(); time.sleep(0.3)         # 100 Hz for a sharp peak
    print(f"{'clk':>5} | {'avg':>5} {'p50':>5} {'max':>5} | {'smi(busy)':>9} | {'util':>4} {'T tok/s':>7}", flush=True)
    for clk in [255, 562, 870, 1117, 1387, 1528]:
        sudo(["nvidia-smi", "-i", GPU, "-lgc", f"{clk},{clk}"]); time.sleep(0.4); free()
        # warm a bit, then read a few live nvidia-smi values while decode is hammering
        r = run_decode_point(model, s, 96, 256, vocab)
        smi = max(smi_power() for _ in range(4))
        print(f"{clk:>5} | {r['power_avg_w']:>5.0f} {r.get('power_p50_w',0):>5.0f} {r.get('power_max_w',0):>5.0f} "
              f"| {smi:>9.0f} | {r.get('util_gpu_avg',0):>4.0f} {r['throughput_tok_s']:>7.0f}", flush=True)
    sudo(["nvidia-smi", "-i", GPU, "-rgc"]); s.stop(); s.shutdown(); pynvml.nvmlShutdown()
    print("[reset] -rgc", flush=True)


if __name__ == "__main__":
    main()
