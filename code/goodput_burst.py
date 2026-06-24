"""Cold-BURST goodput — push the clock HIGH and keep it there.

Why the previous attempts failed: a sustained measurement window lets the die heat to its
~82C thermal limit, so the governor trades clock for occupancy at constant power -> 'bigger
batch => lower clock', and power stalls at the thermal ceiling (~140-200W), never reaching
high power caps.

Fix (= "push frequency, wait to cool when it overheats", taken to the limit):
  1. LOCK the SM clock high (sweep a set of locked clocks to span the power range).
  2. Before EVERY point, deep-cool the die to <= COOL_TARGET.
  3. Measure a VERY SHORT burst (warmup 0.4 s + window 1.0 s) so the burst ENDS before the
     die heats into thermal throttle -> the locked clock HOLDS even at large batch.
We record the ACTUAL clock per point to prove the lock held (act ~ req == success).
Caveat: this is a TRANSIENT peak, not a sustainable operating point; at the very top clock +
large batch the die may still heat past the limit within 1 s (then act < req -> shorten window).

  SUDO_PASS=... CUDA_VISIBLE_DEVICES=0 PYTHONPATH=code python3 code/goodput_burst.py
"""
from __future__ import annotations
import csv, os, subprocess, time
import torch
import config as C
# SHORT burst: stay cold during the measurement so the locked clock does not droop.
C.WARMUP_S = 0.4; C.SETTLE_S = 0.05; C.MEASURE_S = 1.0
from power_sampler import PowerSampler                       # noqa: E402
from measure import load_model, run_prefill_point, run_decode_point, free  # noqa: E402

PW = os.environ.get("SUDO_PASS", "")
CLOCKS = [600, 810, 1020, 1230, 1380, 1530]   # MHz, the "push frequency" knob (spans power)
PREFILL_BATCHES = [1, 4, 16, 48]
DECODE_BATCHES = [1, 8, 32, 64]
S, CTX = 512, 256
COOL_TARGET, COOL_MAX_S = 52.0, 200.0


def sudo_nv(*a):
    return subprocess.run(["sudo", "-S", "-p", "", "nvidia-smi", "-i", "0", *a],
                          input=PW + "\n", text=True, capture_output=True)


def deep_cool(sampler):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < COOL_MAX_S:
        t = sampler.samples[-1]["temp"] if sampler.samples else 0
        if t and t <= COOL_TARGET:
            break
        time.sleep(1.0)
    return sampler.samples[-1]["temp"] if sampler.samples else 0


def main():
    if not PW:
        print("ERROR: set SUDO_PASS"); return
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, model = load_model(); vocab = model.config.vocab_size
    sampler = PowerSampler(interval_s=C.SAMPLE_INTERVAL_S); sampler.start(); time.sleep(0.3)
    print(f"loaded {C.MODEL_ID} | burst W{C.WARMUP_S}/M{C.MEASURE_S}s | cool<= {COOL_TARGET}C | clocks {CLOCKS}", flush=True)
    rows = []
    try:
        for f in CLOCKS:
            if sudo_nv("-lgc", str(f)).returncode != 0:
                print(f"lock {f} FAILED"); continue
            print(f"=== locked clock {f} MHz ===", flush=True)
            for b in PREFILL_BATCHES:
                free(); ct = deep_cool(sampler)
                try:
                    pr = run_prefill_point(model, sampler, b, S, vocab)
                except torch.cuda.OutOfMemoryError:
                    print(f"  prefill b{b} OOM", flush=True); free(); continue
                ttft = pr["wall_s"] / pr["iters"] * 1e3
                act = round(pr.get("sm_clk_avg", 0))
                rows.append({"req_clk": f, "act_clk": act, "phase": "prefill", "batch": b,
                             "throughput_tok_s": round(pr["throughput_tok_s"], 1), "latency_ms": round(ttft, 2),
                             "power_avg_w": round(pr.get("power_avg_w", 0), 1), "temp_end": round(pr.get("temp_avg", 0))})
                held = "OK" if act >= f - 40 else "DROOPED"
                print(f"  prefill b{b:>3} | act {act:>4} [{held}] | {pr['throughput_tok_s']:>7.0f} tok/s | "
                      f"TTFT {ttft:>5.0f} ms | {pr.get('power_avg_w',0):>5.0f} W | cold {ct:.0f}->{pr.get('temp_avg',0):.0f}C", flush=True)
            for b in DECODE_BATCHES:
                free(); ct = deep_cool(sampler)
                try:
                    dr = run_decode_point(model, sampler, b, CTX, vocab)
                except torch.cuda.OutOfMemoryError:
                    print(f"  decode b{b} OOM", flush=True); free(); continue
                tpot = dr["wall_s"] / dr["steps"] * 1e3
                act = round(dr.get("sm_clk_avg", 0))
                rows.append({"req_clk": f, "act_clk": act, "phase": "decode", "batch": b,
                             "throughput_tok_s": round(dr["throughput_tok_s"], 1), "latency_ms": round(tpot, 2),
                             "power_avg_w": round(dr.get("power_avg_w", 0), 1), "temp_end": round(dr.get("temp_avg", 0))})
                print(f"  decode  b{b:>3} | act {act:>4} | {dr['throughput_tok_s']:>7.0f} tok/s | "
                      f"TPOT {tpot:>5.1f} ms | {dr.get('power_avg_w',0):>5.0f} W", flush=True)
    finally:
        sudo_nv("-rgc"); print("[reset] clock unlocked", flush=True)
        sampler.stop(); sampler.shutdown()
    keys = ["req_clk", "act_clk", "phase", "batch", "throughput_tok_s", "latency_ms", "power_avg_w", "temp_end"]
    slug = C.MODEL_ID.split("/")[-1]
    path = os.path.join(C.RESULTS_DIR, f"goodput_burst_{slug}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
