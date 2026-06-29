"""Decode power<->throughput: sweep the POWER CAP (ascending), and at each cap sweep BATCH.

The earlier decode sweep fixed batch=48 and only varied the cap, which traced a thin curve. Here we
map the full decode operating space: for each power cap (low -> high) we grow the batch from small to
large (concurrency knob), recording throughput + actual power + clocks. The cap bounds the SM clock;
the batch adds concurrency -> together they fill out the decode P-T cloud.

Batch starts SMALL and grows, stopping the moment a point OOMs (KV cache fills VRAM) -- so we never
blow up memory by starting too full. Context fixed at C=256.

  SUDO_PASS=... PYTHONPATH=code python3 code/decode_pt_sweep.py        # GPU pinned by config (GPU1)
"""
from __future__ import annotations
import csv, os, subprocess, time
import torch
import pynvml
import config as C
C.WARMUP_S = 0.5; C.SETTLE_S = 0.1; C.MEASURE_S = 2.0          # short bursts -> stay cool, clean points
from power_sampler import PowerSampler                          # noqa: E402  (auto-targets the CVD GPU)
from measure import load_model, run_decode_point, free          # noqa: E402

PW = os.environ.get("SUDO_PASS", "")
GPU = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")   # physical index for nvidia-smi
CAP_GRID = [120, 150, 180, 210, 250]                           # W, ascending power-cap sweep
BATCHES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96]                # ascending; start small, break on OOM
CTX = 256
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
    print(f"GPU{GPU} {sampler.name} | {C.MODEL_ID} | cap range [{mn:.0f},{mx:.0f}] | caps {caps}", flush=True)

    rows = []
    try:
        for cap in caps:
            if sudo_pl(cap).returncode != 0:
                print(f"  -pl {cap} FAILED"); continue
            time.sleep(0.4)
            print(f"=== CAP {cap} W ===", flush=True)
            for b in BATCHES:                                  # small -> large; stop at OOM
                free(); cool(sampler)
                try:
                    r = run_decode_point(model, sampler, b, CTX, vocab)
                except torch.cuda.OutOfMemoryError:
                    print(f"  b{b:>3}  OOM -> stop growing batch at this cap", flush=True); free(); break
                rows.append({"cap_w": cap, "batch": b, "ctx": CTX,
                             "throughput_tok_s": round(r["throughput_tok_s"], 1),
                             "power_avg_w": round(r.get("power_avg_w", 0), 1),
                             "sm_clk_avg": round(r.get("sm_clk_avg", 0)),
                             "mem_clk_avg": round(r.get("mem_clk_avg", 0)),
                             "util_gpu_avg": round(r.get("util_gpu_avg", 0)),
                             "tok_per_joule": round(r.get("tok_per_joule", 0), 2),
                             "temp_avg": round(r.get("temp_avg", 0), 1)})
                print(f"  b{b:>3} | {r['throughput_tok_s']:>6.0f} tok/s | {r.get('power_avg_w',0):>5.0f}W "
                      f"| sm {r.get('sm_clk_avg',0):.0f} | {r.get('tok_per_joule',0):.1f} tok/J", flush=True)
    finally:
        sudo_pl(round(default))
        print(f"\n[reset] -pl {default:.0f}W", flush=True)
        sampler.stop(); sampler.shutdown(); pynvml.nvmlShutdown()

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pt_cap_gpu1")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "decode_pt.csv")
    keys = ["cap_w", "batch", "ctx", "throughput_tok_s", "power_avg_w", "sm_clk_avg",
            "mem_clk_avg", "util_gpu_avg", "tok_per_joule", "temp_avg"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
