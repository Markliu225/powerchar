"""DVFS sweep -- the OTHER controlled experiment for P(T): the ≈cubic law.

Here the controlled variable is the **SM clock frequency**, with the workload
held FIXED. For a compute-bound load throughput scales with clock (T ∝ f) while
power follows the dynamic-power law P ≈ P_static + k·f^γ (γ ≈ 2–3, because the
core voltage rises with frequency, P_dyn = C·V²·f). Eliminating f gives the
convex, super-linear  P ≈ P_static + k'·T^γ  -- the "approximately cubic"
power–throughput curve. A memory-bound decode load is included as a contrast:
its throughput barely responds to clock, so its curve is far less steep.

REQUIRES CLOCK-LOCK PERMISSION. On Windows GeForce, nvmlDeviceSetGpuLockedClocks
needs an **Administrator** shell; on Linux, run with sudo. If locking is denied
the script prints instructions and exits without touching the GPU clocks.

Run (Windows, elevated PowerShell):   python code/measure_dvfs.py
Then analyse:                         python code/analyze.py --step dvfs
"""
from __future__ import annotations
import csv
import os
import pynvml
import torch

import config as C
from power_sampler import PowerSampler
from measure import load_model, run_prefill_point, run_decode_point, free

# SM clocks to sweep (MHz). The card may not sustain the very top under load;
# the actual achieved clock is recorded alongside the requested one.
FREQS = [600, 900, 1200, 1500, 1800, 2100, 2400, 2700]

# Fixed, compute-bound prefill workload (T should track clock closely) and a
# memory-bound decode workload (T should barely move with clock).
PREFILL_WL = dict(batch=4, seq_len=512)
DECODE_WL = dict(batch=16, ctx_len=256)


def can_lock(h) -> bool:
    try:
        pynvml.nvmlDeviceSetGpuLockedClocks(h, FREQS[0], FREQS[0])
        pynvml.nvmlDeviceResetGpuLockedClocks(h)
        return True
    except pynvml.NVMLError as e:
        print("\n[!] Cannot lock GPU clocks:", e)
        print("    The DVFS / cubic-law sweep needs clock-control permission.")
        print("    Windows: launch an *Administrator* PowerShell, then run")
        print("             python code/measure_dvfs.py")
        print("    Linux:   run with sudo (and `nvidia-smi -pm 1`).\n")
        return False


def main():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(int((os.environ.get("CUDA_VISIBLE_DEVICES","0").split(",")[0] or "0")))
    if not can_lock(h):
        pynvml.nvmlShutdown()
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    print("loading model...", flush=True)
    tok, model = load_model()
    vocab = model.config.vocab_size
    sampler = PowerSampler(interval_s=C.SAMPLE_INTERVAL_S)
    sampler.start()
    print(f"GPU {sampler.name}  cap {sampler.power_limit_w:.0f} W", flush=True)

    rows = []
    try:
        for f in FREQS:
            pynvml.nvmlDeviceSetGpuLockedClocks(h, f, f)
            free()
            rp = run_prefill_point(model, sampler, PREFILL_WL["batch"],
                                   PREFILL_WL["seq_len"], vocab)
            free()
            rd = run_decode_point(model, sampler, DECODE_WL["batch"],
                                  DECODE_WL["ctx_len"], vocab)
            for wl, r in (("prefill", rp), ("decode", rd)):
                rows.append({
                    "workload": wl, "req_clk_mhz": f,
                    "act_clk_mhz": round(r.get("sm_clk_avg", 0), 0),
                    "throughput_tok_s": round(r["throughput_tok_s"], 1),
                    "power_avg_w": round(r.get("power_avg_w", 0), 2),
                })
            print(f"  f={f:>4} MHz | prefill {rp['throughput_tok_s']:>8.0f} tok/s "
                  f"{rp.get('power_avg_w',0):>6.1f} W (clk {rp.get('sm_clk_avg',0):.0f}) | "
                  f"decode {rd['throughput_tok_s']:>6.0f} tok/s "
                  f"{rd.get('power_avg_w',0):>6.1f} W", flush=True)
    finally:
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(h)
            print("[ok] GPU clocks reset to default", flush=True)
        except pynvml.NVMLError:
            pass
        sampler.stop(); sampler.shutdown()

    path = os.path.join(C.RESULTS_DIR, "dvfs.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        [w.writerow(r) for r in rows]
    print(f"wrote {path} ({len(rows)} rows)  ->  python code/analyze.py --step dvfs")


if __name__ == "__main__":
    main()
