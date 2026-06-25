"""Decode SATURATION sweep — push batch well past B*=W/(C*kv) to expose the bandwidth ceiling.

Why the earlier sweep missed it: at C=256 the half-saturation batch is B*=76, but weights+KV
exhaust the 32 GB V100 right around b~76 too -> you OOM AT the knee, never past it, so throughput
still looks linear in batch. Fix: use LONGER context so B* shrinks (B*∝1/C) and a modest,
memory-feasible batch already exceeds it. C=1024 -> B*=19 (reach ~2x); C=2048 -> B*=9.5 (reach ~2.5x).

Per point we log T, the per-sequence rate T/B (constant => still linear/unsaturated; falling =>
saturating), and the IMPLIED effective bandwidth beta_eff = T*(W+B*C*kv)/B. If the roofline model
holds, beta_eff is ~constant across batch and equals the real achieved HBM bandwidth; then the true
ceiling is T_max = beta_eff/(C*kv).

  SUDO_PASS=... CUDA_VISIBLE_DEVICES=0 PYTHONPATH=code python3 code/decode_sat.py
"""
from __future__ import annotations
import csv, os, subprocess, time
import torch
import pynvml
import config as C
C.WARMUP_S = 0.6; C.SETTLE_S = 0.1; C.MEASURE_S = 3.0
from power_sampler import PowerSampler                       # noqa: E402
from measure import load_model, run_decode_point, free        # noqa: E402

os.environ["DECODE_KV_HEADROOM"] = "64"   # shrink KV reserve so high batch fits (32 GB / huge MHA KV)
PW = os.environ.get("SUDO_PASS", "")
LOCK_CLOCK = 1290                       # lock SM clock (decode is clock-insensitive; removes DVFS/thermal confound)
KV = 393216.0                           # KV bytes/token
W_BYTES = 7.642e9                       # weight bytes/step
COOL_TARGET, COOL_HOT, COOL_MAX_S = 62.0, 74.0, 90.0
# context -> batch grid (pushed to batch 256 / OOM); B* = W/(C*kv): C256->76, C128->152.
# C=256 is the ORIGINAL measurement context (old sweep stopped at b=32 << B*=76 -> looked linear);
# now push past B* to expose saturation. C=128 reaches batch 256 within 32 GB.
GRID = {
    256: [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256],
    128: [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256],
}


def sudo_nv(*a):
    return subprocess.run(["sudo", "-S", "-p", "", "nvidia-smi", "-i", "0", *a],
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
    pynvml.nvmlInit()
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, model = load_model(); vocab = model.config.vocab_size
    sampler = PowerSampler(interval_s=C.SAMPLE_INTERVAL_S); sampler.start(); time.sleep(0.3)
    sudo_nv("-lgc", str(LOCK_CLOCK))
    print(f"loaded {C.MODEL_ID} | clock locked {LOCK_CLOCK} MHz\n", flush=True)

    rows = []
    try:
        for ctx, batches in GRID.items():
            bstar = W_BYTES / (ctx * KV)
            print(f"===== context C={ctx}  (B*={bstar:.1f})  =====", flush=True)
            print(f"  {'batch':>5}{'tok/s':>9}{'T/B':>7}{'beta_eff_GB/s':>14}{'%ceil':>7}{'power':>7}{'clk':>6}{'util':>6}", flush=True)
            for b in batches:
                free(); cool(sampler)
                try:
                    r = run_decode_point(model, sampler, b, ctx, vocab)
                except torch.cuda.OutOfMemoryError:
                    print(f"  {b:>5}  OOM -> memory wall at C={ctx}", flush=True); free(); break
                T = r["throughput_tok_s"]
                beta_eff = T * (W_BYTES + b * ctx * KV) / b          # implied achieved bandwidth (B/s)
                frac = T / (beta_eff / (ctx * KV))                  # T / T_max(beta_eff)  == b/(b+B*)
                p = r.get("power_avg_w", 0); clk = r.get("sm_clk_avg", 0); util = r.get("util_gpu_avg", 0)
                rows.append({"context": ctx, "batch": b, "throughput_tok_s": round(T, 1),
                             "tok_per_seq_s": round(T / b, 2), "beta_eff_gbs": round(beta_eff / 1e9, 1),
                             "frac_of_ceiling": round(frac, 3), "power_avg_w": round(p, 1),
                             "sm_clk_avg": round(clk), "util_gpu_avg": round(util), "tok_per_joule": round(T / p, 2) if p else 0})
                print(f"  {b:>5}{T:>9.0f}{T/b:>7.1f}{beta_eff/1e9:>14.1f}{100*frac:>6.0f}%{p:>7.0f}{clk:>6.0f}{util:>6.0f}", flush=True)
            print(flush=True)
    finally:
        sudo_nv("-rgc")
        print("[reset] clock unlocked", flush=True)
        sampler.stop(); sampler.shutdown(); pynvml.nvmlShutdown()

    path = os.path.join(C.RESULTS_DIR, "decode_saturation.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
