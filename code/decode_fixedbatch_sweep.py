"""Decode at FIXED (large) batch, sweeping only the POWER CAP — to test whether decode ever becomes
compute-bound.

If decode were compute-bound at some point, capping power (which throttles the SM CORE clock, NOT the
fixed 877 MHz HBM clock) would drop throughput strongly. If it is bandwidth-bound throughout, throughput
should stay nearly flat as power falls — only a weak slope from kernel-launch overhead / bandwidth
under-saturation, both of which scale with core clock, not from a compute roofline.

Holds batch fixed (default 64), sweeps the cap low->high, and records throughput, ACTUAL power, SM clock
and efficiency at each cap. Plot with plot_decode_fixedbatch.py (throughput-vs-power + efficiency-vs-power).

  SUDO_PASS=... PYTHONPATH=code python3 code/decode_fixedbatch_sweep.py     # GPU pinned to GPU1 by config
"""
from __future__ import annotations
import csv, os, subprocess, time
import torch
import pynvml
import config as C
C.WARMUP_S = 0.6; C.SETTLE_S = 0.1; C.MEASURE_S = 2.5            # short bursts -> stay cool, clean points
from power_sampler import PowerSampler                            # noqa: E402  (auto-targets the CVD GPU)
from measure import load_model, run_decode_point, free           # noqa: E402

PW = os.environ.get("SUDO_PASS", "")
GPU = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")    # physical index for nvidia-smi
BATCH = int(os.environ.get("DECODE_BATCH", "64"))                # FIXED batch (large; held constant)
CTX = 256
CAP_GRID = [100, 115, 130, 145, 160, 175, 190, 210, 230, 250]   # W, ascending
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
    print(f"GPU{GPU} {sampler.name} | {C.MODEL_ID} | FIXED batch={BATCH}, ctx={CTX} | "
          f"cap range [{mn:.0f},{mx:.0f}] | caps {caps}", flush=True)

    rows = []
    try:
        for cap in caps:
            if sudo_pl(cap).returncode != 0:
                print(f"  -pl {cap} FAILED"); continue
            time.sleep(0.4); free(); cool(sampler)
            try:
                r = run_decode_point(model, sampler, BATCH, CTX, vocab)
            except torch.cuda.OutOfMemoryError:
                print(f"  cap {cap}: OOM at batch {BATCH} -> lower DECODE_BATCH"); free(); break
            rows.append({"cap_w": cap, "batch": BATCH, "ctx": CTX,
                         "throughput_tok_s": round(r["throughput_tok_s"], 1),
                         "power_avg_w": round(r.get("power_avg_w", 0), 1),
                         "sm_clk_avg": round(r.get("sm_clk_avg", 0)),
                         "mem_clk_avg": round(r.get("mem_clk_avg", 0)),
                         "util_gpu_avg": round(r.get("util_gpu_avg", 0)),
                         "tok_per_joule": round(r.get("tok_per_joule", 0), 2),
                         "temp_avg": round(r.get("temp_avg", 0), 1)})
            print(f"  cap {cap:>3}W | {r['throughput_tok_s']:>6.0f} tok/s | act {r.get('power_avg_w',0):>5.0f}W "
                  f"| sm {r.get('sm_clk_avg',0):.0f} mem {r.get('mem_clk_avg',0):.0f} "
                  f"| {r.get('tok_per_joule',0):.1f} tok/J", flush=True)
    finally:
        sudo_pl(round(default))
        print(f"\n[reset] -pl {default:.0f}W", flush=True)
        sampler.stop(); sampler.shutdown(); pynvml.nvmlShutdown()

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pt_cap_gpu1")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "decode_fixedbatch.csv")
    keys = ["cap_w", "batch", "ctx", "throughput_tok_s", "power_avg_w", "sm_clk_avg",
            "mem_clk_avg", "util_gpu_avg", "tok_per_joule", "temp_avg"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"wrote {path} ({len(rows)} caps at fixed batch {BATCH})", flush=True)


if __name__ == "__main__":
    main()
