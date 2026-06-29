"""Prefill & decode power<->throughput UNDER POWER CAPS, one model, GPU = CUDA_VISIBLE_DEVICES.

The right way to trace a P-vs-T curve is to sweep the POWER CAP (nvidia-smi -pl): at each cap the
card draws ~that power and picks the clock that fits, so power AND throughput both move. (A plain
batch sweep instead pins power at the ~250 W enforced cap almost immediately -> a degenerate
vertical/L-shaped curve.) We hold a fixed heavy workload per phase and sweep the cap.

  prefill: compute-bound -> cap throttles the clock -> throughput drops with the cap (steep curve).
  decode : memory-bound, ~clock-insensitive -> throughput barely drops as the cap falls (flat curve)
           = the key result: capping power costs prefill a lot, decode almost nothing.

Telemetry (pynvml) and the -pl target follow CUDA_VISIBLE_DEVICES. Short bursts + light cooldown so
points reflect the CAP, not thermal throttling. nvidia-smi is only used to set/reset -pl.

  SUDO_PASS=... CUDA_VISIBLE_DEVICES=1 PYTHONPATH=code python3 code/pt_cap_sweep.py
"""
from __future__ import annotations
import csv, os, subprocess, time
import torch
import pynvml
import config as C
C.WARMUP_S = 0.5; C.SETTLE_S = 0.1; C.MEASURE_S = 2.5
from power_sampler import PowerSampler                       # noqa: E402  (auto-targets the CVD GPU)
from measure import load_model, run_prefill_point, run_decode_point, free  # noqa: E402

PW = os.environ.get("SUDO_PASS", "")
GPU = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")
CAP_GRID = [100, 120, 140, 160, 180, 200, 220, 250]         # W, power-cap sweep (both phases)
PREFILL_BATCH, PREFILL_S = 8, 512                           # heavy prefill -> draws >=250W, cap always bites
DECODE_BATCH, DECODE_CTX = 48, 256                          # heavy decode -> ~225W, caps below that bite
COOL_TARGET, COOL_HOT, COOL_MAX_S = 48.0, 58.0, 60.0


def sudo_pl(w):
    return subprocess.run(["sudo", "-S", "-p", "", "nvidia-smi", "-i", GPU, "-pl", str(int(w))],
                          input=PW + "\n", text=True, capture_output=True)


def cool(sampler):
    t = sampler.samples[-1]["temp"] if sampler.samples else 0
    if not t or t <= COOL_HOT:
        return
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < COOL_MAX_S:
        tt = sampler.samples[-1]["temp"] if sampler.samples else 0
        if tt and tt <= COOL_TARGET:
            break
        time.sleep(1.0)


def main():
    if not PW:
        print("ERROR: set SUDO_PASS"); return
    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(int(GPU))
    mn, mx = [x / 1000.0 for x in pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)]
    default = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000.0
    caps = [c for c in CAP_GRID if mn <= c <= mx]
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, model = load_model(); vocab = model.config.vocab_size
    sampler = PowerSampler(interval_s=C.SAMPLE_INTERVAL_S); sampler.start(); time.sleep(0.3)
    print(f"GPU{GPU} {sampler.name} | {C.MODEL_ID} | cap range [{mn:.0f},{mx:.0f}] default {default:.0f}W | caps {caps}", flush=True)

    rows = []
    try:
        for cap in caps:
            r = sudo_pl(cap)
            if r.returncode != 0:
                print(f"  -pl {cap} FAILED: {r.stderr.strip()[:80]}"); continue
            time.sleep(0.4)
            print(f"=== CAP {cap} W ===", flush=True)
            for phase, fn, args in (("prefill", run_prefill_point, (PREFILL_BATCH, PREFILL_S)),
                                    ("decode", run_decode_point, (DECODE_BATCH, DECODE_CTX))):
                free(); cool(sampler)
                try:
                    res = fn(model, sampler, *args, vocab)
                except torch.cuda.OutOfMemoryError:
                    print(f"  {phase} OOM", flush=True); free(); continue
                rows.append({"phase": phase, "cap_w": cap,
                             "throughput_tok_s": round(res["throughput_tok_s"], 1),
                             "power_avg_w": round(res.get("power_avg_w", 0), 1),
                             "sm_clk_avg": round(res.get("sm_clk_avg", 0)),
                             "mem_clk_avg": round(res.get("mem_clk_avg", 0)),
                             "util_gpu_avg": round(res.get("util_gpu_avg", 0)),
                             "util_mem_avg": round(res.get("util_mem_avg", 0)),
                             "tok_per_joule": round(res.get("tok_per_joule", 0), 2),
                             "temp_avg": round(res.get("temp_avg", 0), 1)})
                print(f"  {phase:<7} | {res['throughput_tok_s']:>7.0f} tok/s | {res.get('power_avg_w',0):>5.0f}W "
                      f"| sm {res.get('sm_clk_avg',0):.0f} mem {res.get('mem_clk_avg',0):.0f} "
                      f"| util {res.get('util_gpu_avg',0):.0f}/{res.get('util_mem_avg',0):.0f}% "
                      f"| {res.get('tok_per_joule',0):.1f} tok/J", flush=True)
    finally:
        sudo_pl(round(default))
        print(f"\n[reset] -pl {default:.0f}W", flush=True)
        sampler.stop(); sampler.shutdown(); pynvml.nvmlShutdown()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pt_cap_gpu1")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pt_cap.csv")
    keys = ["phase", "cap_w", "throughput_tok_s", "power_avg_w", "sm_clk_avg", "mem_clk_avg",
            "util_gpu_avg", "util_mem_avg", "tok_per_joule", "temp_avg"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
