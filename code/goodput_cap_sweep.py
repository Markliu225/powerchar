"""Goodput vs power cap — LOCKED CLOCK + POWER CAP together, cold, with latency early-stop.

Design:
  * LOCK the SM clock to max (push frequency up so the GPU uses the whole power budget) AND
    set a POWER CAP (the budget). With both set, at a low cap the cap pulls the clock below
    the lock (= the highest clock that budget allows); at a high cap it runs at the locked max.
  * Reactive cooldown + SHORT burst so the die never reaches the thermal limit -> only the
    *cap* sets the operating point (deterministic), not temperature.
  * Sweep batch but EARLY-STOP as soon as latency exceeds the SLO (prefill TTFT / decode TPOT)
    -- larger batches only raise latency further, so there is no point measuring them.
  * Sweep the power cap. goodput(cap, SLO) = best throughput meeting the SLO at that cap
    (derived in plot_goodput.py; STOP_* here are loose guards so tighter SLOs are still covered).

  SUDO_PASS=... CUDA_VISIBLE_DEVICES=0 PYTHONPATH=code python3 code/goodput_cap_sweep.py
"""
from __future__ import annotations
import csv, os, subprocess, time
import torch
import pynvml
import config as C
C.WARMUP_S = 0.4; C.SETTLE_S = 0.05; C.MEASURE_S = 1.2     # short burst: clock holds while cold
from power_sampler import PowerSampler                       # noqa: E402
from measure import load_model, run_prefill_point, run_decode_point, free  # noqa: E402

PW = os.environ.get("SUDO_PASS", "")
LOCK_CLOCK = 1530                       # push frequency to max; the cap limits power below it
CAP_GRID = list(range(100, 301, 20))   # W, 20-W steps over the card's range
PREFILL_BATCHES = [1, 2, 4, 8, 16, 32, 48, 64]
DECODE_BATCHES  = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
S, CTX = 512, 256
COOL_TARGET, COOL_HOT, COOL_MAX_S = 55.0, 72.0, 180.0
STOP_TTFT_MS = 1500.0                   # prefill: stop growing batch once a prefill call exceeds this
STOP_TPOT_MS = 150.0                    # decode:  stop once per-token latency exceeds this


def sudo_nv(*a):
    return subprocess.run(["sudo", "-S", "-p", "", "nvidia-smi", "-i", "0", *a],
                          input=PW + "\n", text=True, capture_output=True)


def reactive_cool(sampler):
    t = sampler.samples[-1]["temp"] if sampler.samples else 0
    if not t or t <= COOL_HOT:
        return
    print(f"    [cool] {t:.0f}C -> <= {COOL_TARGET:.0f}C ...", flush=True)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < COOL_MAX_S:
        tt = sampler.samples[-1]["temp"] if sampler.samples else 0
        if tt and tt <= COOL_TARGET:
            break
        time.sleep(1.0)


def main():
    if not PW:
        print("ERROR: set SUDO_PASS"); return
    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
    mn, mx = [x / 1000.0 for x in pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)]
    default = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000.0
    caps = [c for c in CAP_GRID if mn <= c <= mx]
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, model = load_model(); vocab = model.config.vocab_size
    sampler = PowerSampler(interval_s=C.SAMPLE_INTERVAL_S); sampler.start(); time.sleep(0.3)
    sudo_nv("-lgc", str(LOCK_CLOCK))                          # push frequency high
    print(f"loaded {C.MODEL_ID} | locked clock {LOCK_CLOCK} MHz | cap range [{mn:.0f},{mx:.0f}] | caps {caps}", flush=True)

    rows = []
    try:
        for cap in caps:
            if sudo_nv("-pl", str(cap)).returncode != 0:
                print(f"  -pl {cap} FAILED"); continue
            time.sleep(0.4)
            print(f"=== CAP {cap} W  (clock locked {LOCK_CLOCK}) ===", flush=True)
            for b in PREFILL_BATCHES:
                free(); reactive_cool(sampler)
                try:
                    pr = run_prefill_point(model, sampler, b, S, vocab)
                except torch.cuda.OutOfMemoryError:
                    print(f"  prefill b{b} OOM", flush=True); free(); break
                ttft = pr["wall_s"] / pr["iters"] * 1e3
                rows.append({"cap_w": cap, "phase": "prefill", "batch": b,
                             "throughput_tok_s": round(pr["throughput_tok_s"], 1), "latency_ms": round(ttft, 2),
                             "power_avg_w": round(pr.get("power_avg_w", 0), 1), "sm_clk_avg": round(pr.get("sm_clk_avg", 0))})
                print(f"  prefill b{b:>3} | {pr['throughput_tok_s']:>7.0f} tok/s | TTFT {ttft:>6.0f} ms | "
                      f"{pr.get('power_avg_w',0):>5.0f}W clk{pr.get('sm_clk_avg',0):.0f}", flush=True)
                if ttft > STOP_TTFT_MS:
                    print(f"    TTFT>{STOP_TTFT_MS:.0f} ms -> stop growing prefill batch", flush=True); break
            for b in DECODE_BATCHES:
                free(); reactive_cool(sampler)
                try:
                    dr = run_decode_point(model, sampler, b, CTX, vocab)
                except torch.cuda.OutOfMemoryError:
                    print(f"  decode b{b} OOM", flush=True); free(); break
                tpot = dr["wall_s"] / dr["steps"] * 1e3
                rows.append({"cap_w": cap, "phase": "decode", "batch": b,
                             "throughput_tok_s": round(dr["throughput_tok_s"], 1), "latency_ms": round(tpot, 2),
                             "power_avg_w": round(dr.get("power_avg_w", 0), 1), "sm_clk_avg": round(dr.get("sm_clk_avg", 0))})
                print(f"  decode  b{b:>3} | {dr['throughput_tok_s']:>7.0f} tok/s | TPOT {tpot:>6.1f} ms | "
                      f"{dr.get('power_avg_w',0):>5.0f}W clk{dr.get('sm_clk_avg',0):.0f}", flush=True)
                if tpot > STOP_TPOT_MS:
                    print(f"    TPOT>{STOP_TPOT_MS:.0f} ms -> stop growing decode batch", flush=True); break
    finally:
        sudo_nv("-rgc"); sudo_nv("-pl", str(round(default)))
        print(f"[reset] clock unlocked, -pl {default:.0f} W", flush=True)
        sampler.stop(); sampler.shutdown(); pynvml.nvmlShutdown()

    keys = ["cap_w", "phase", "batch", "throughput_tok_s", "latency_ms", "power_avg_w", "sm_clk_avg"]
    slug = C.MODEL_ID.split("/")[-1]
    path = os.path.join(C.RESULTS_DIR, f"goodput_{slug}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
