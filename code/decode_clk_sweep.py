"""Decode at FIXED full batch, sweeping the SM CLOCK LOCK (-lgc) — fine power resolution, from ~50 W.

For a light workload like decode the power cap (-pl) floor is 100 W and barely binds, so -pl can't go
below ~100 W. Locking the SM clock (-lgc) directly chooses the operating point: low clock -> low power
(down toward ~45-50 W), finely sampled. HBM clock stays fixed at 877 MHz, isolating how decode
throughput responds to compute (core clock) at fixed bandwidth.

Fixed batch (default 96 = full), sweeps locked SM clock from CLK_LO (default 255 MHz ~ 45 W; below this
the prefill seed gets impractically slow for little extra power range) up to max, denser at the low end.
Rows are written INCREMENTALLY to pt_cap_gpu1/decode_fixedbatch.csv so partial data survives a kill.

  SUDO_PASS=... CUDA_VISIBLE_DEVICES=0 N_CLK=14 CLK_LO=255 PYTHONPATH=code python3 code/decode_clk_sweep.py
"""
from __future__ import annotations
import csv, os, subprocess, time
import numpy as np
import torch
import pynvml
import config as C
C.WARMUP_S = 0.4; C.SETTLE_S = 0.1; C.MEASURE_S = 2.0
from power_sampler import PowerSampler                            # noqa: E402
from measure import load_model, run_decode_point, free           # noqa: E402

PW = os.environ.get("SUDO_PASS", "")
GPU = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")
BATCH = int(os.environ.get("DECODE_BATCH", "96"))
CTX = 256
N_CLK = int(os.environ.get("N_CLK", "14"))
CLK_LO = int(os.environ.get("CLK_LO", "255"))                    # lowest SM clock to sample (MHz)
COOL_TARGET, COOL_HOT, COOL_MAX_S = 50.0, 60.0, 30.0


def sudo(args):
    return subprocess.run(["sudo", "-S", "-p", ""] + args, input=PW + "\n", text=True, capture_output=True)


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


def pick_clocks(h):
    mem = sorted(pynvml.nvmlDeviceGetSupportedMemoryClocks(h))[-1]
    sup = sorted(c for c in set(pynvml.nvmlDeviceGetSupportedGraphicsClocks(h, mem)) if c >= CLK_LO)
    lo, hi = sup[0], sup[-1]
    targets = lo + (hi - lo) * np.linspace(0, 1, N_CLK) ** 1.5    # denser at low clock
    return mem, sorted({min(sup, key=lambda c: abs(c - t)) for t in targets})


def main():
    if not PW:
        print("ERROR: set SUDO_PASS", flush=True); return
    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(int(GPU))
    mem, clks = pick_clocks(h)
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, model = load_model(); vocab = model.config.vocab_size
    sampler = PowerSampler(interval_s=C.SAMPLE_INTERVAL_S); sampler.start(); time.sleep(0.3)
    print(f"GPU{GPU} {sampler.name} | {C.MODEL_ID} | batch={BATCH} | mem {mem}MHz "
          f"| SM clocks {clks[0]}..{clks[-1]} ({len(clks)} pts)", flush=True)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pt_cap_gpu1")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "decode_fixedbatch.csv")
    keys = ["cap_w", "batch", "ctx", "throughput_tok_s", "power_avg_w", "sm_clk_avg",
            "mem_clk_avg", "util_gpu_avg", "tok_per_joule", "temp_avg"]
    fcsv = open(path, "w", newline=""); wr = csv.DictWriter(fcsv, fieldnames=keys)
    wr.writeheader(); fcsv.flush()
    n = 0
    try:
        for clk in clks:
            if sudo(["nvidia-smi", "-i", GPU, "-lgc", f"{clk},{clk}"]).returncode != 0:
                print(f"  -lgc {clk} FAILED", flush=True); continue
            time.sleep(0.4); free(); cool(sampler)
            try:
                r = run_decode_point(model, sampler, BATCH, CTX, vocab)
            except torch.cuda.OutOfMemoryError:
                print(f"  clk {clk}: OOM", flush=True); free(); break
            row = {"cap_w": clk, "batch": BATCH, "ctx": CTX,
                   "throughput_tok_s": round(r["throughput_tok_s"], 1),
                   "power_avg_w": round(r.get("power_avg_w", 0), 1),
                   "sm_clk_avg": round(r.get("sm_clk_avg", 0)), "mem_clk_avg": round(r.get("mem_clk_avg", 0)),
                   "util_gpu_avg": round(r.get("util_gpu_avg", 0)),
                   "tok_per_joule": round(r.get("tok_per_joule", 0), 2), "temp_avg": round(r.get("temp_avg", 0), 1)}
            wr.writerow(row); fcsv.flush(); n += 1
            print(f"  lock {clk:>4}MHz | {r['throughput_tok_s']:>6.0f} tok/s | act {r.get('power_avg_w',0):>5.0f}W "
                  f"| sm {r.get('sm_clk_avg',0):.0f} | {r.get('tok_per_joule',0):.1f} tok/J", flush=True)
    finally:
        sudo(["nvidia-smi", "-i", GPU, "-rgc"]); fcsv.close()
        print(f"\n[reset] -rgc; wrote {path} ({n} clock points)", flush=True)
        sampler.stop(); sampler.shutdown(); pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
