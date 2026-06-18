"""DVFS / frequency-knob sweep — measure the prefill cubic and decode-flat laws.

Locks the SM clock across a range (via `sudo nvidia-smi -lgc`, password from the
SUDO_PASS env var) and, at each clock, measures a FIXED compute-bound prefill
workload and a FIXED memory-bound decode workload. With the workload fixed and
only the clock varying:
  - prefill (compute-bound):  T ∝ f  and  P ∝ f^~3  =>  P ∝ T^3  (the cubic)
  - decode  (memory-bound) :  T barely responds to f; P rises modestly

The prefill load is kept light (B=4, S=256) so that even at the top clock the
power stays UNDER the 250 W cap — otherwise the cap clips the cubic. Each point
is cooled to a common baseline first, then measured short, so the LOCKED clock
actually holds (no thermal override). Writes results/dvfs.csv.

  SUDO_PASS=... CUDA_VISIBLE_DEVICES=0 PYTHONPATH=code python3 code/dvfs_sweep.py
"""
from __future__ import annotations
import csv, os, subprocess, time
import torch
import config as C
from power_sampler import PowerSampler
from measure import load_model, run_prefill_point, run_decode_point, free

FREQS = [510, 660, 810, 960, 1110, 1260, 1410, 1530]   # MHz, requested SM clocks
PREFILL_WL = dict(batch=4, seq_len=256)                # compute-bound, light enough to stay < cap
DECODE_WL  = dict(batch=16, ctx_len=256)               # memory-bound contrast
COOL_TARGET_C, COOL_MAX_S = 60.0, 30.0
PW = os.environ.get("SUDO_PASS", "")


def _sudo(args):
    return subprocess.run(["sudo", "-S", "-p", "", *args], input=PW + "\n",
                          text=True, capture_output=True)


def lock(f):    return _sudo(["nvidia-smi", "-i", "0", "-lgc", str(f)])
def reset():    return _sudo(["nvidia-smi", "-i", "0", "-rgc"])


def cooldown(sampler):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < COOL_MAX_S:
        t = sampler.samples[-1]["temp"] if sampler.samples else 0
        if t and t <= COOL_TARGET_C:
            break
        time.sleep(0.5)


def main():
    if not PW:
        print("ERROR: set SUDO_PASS env var"); return
    torch.backends.cuda.matmul.allow_tf32 = True
    print("loading model...", flush=True)
    tok, model = load_model()
    vocab = model.config.vocab_size
    sampler = PowerSampler(interval_s=C.SAMPLE_INTERVAL_S); sampler.start(); time.sleep(0.3)
    print(f"GPU {sampler.name} cap {sampler.power_limit_w:.0f} W | sweeping clocks {FREQS}", flush=True)

    rows = []
    try:
        for i, f in enumerate(FREQS, 1):
            r = lock(f)
            if r.returncode != 0:
                print(f"  lock {f} FAILED: {r.stderr.strip()[:80]}"); continue
            cooldown(sampler); free()
            rp = run_prefill_point(model, sampler, PREFILL_WL["batch"], PREFILL_WL["seq_len"], vocab)
            free()
            rd = run_decode_point(model, sampler, DECODE_WL["batch"], DECODE_WL["ctx_len"], vocab)
            for wl, rr in (("prefill", rp), ("decode", rd)):
                rows.append({"workload": wl, "req_clk_mhz": f,
                             "act_clk_mhz": round(rr.get("sm_clk_avg", 0)),
                             "throughput_tok_s": round(rr["throughput_tok_s"], 1),
                             "power_avg_w": round(rr.get("power_avg_w", 0), 2),
                             "power_std_w": round(rr.get("power_std_w", 0), 2),
                             "temp_avg": round(rr.get("temp_avg", 0), 1),
                             "n_samples": rr.get("n_samples", 0)})
            print(f"  [{i}/{len(FREQS)}] req {f:>4} | "
                  f"prefill {rp['throughput_tok_s']:>7.0f} tok/s {rp.get('power_avg_w',0):>6.1f} W "
                  f"(act {rp.get('sm_clk_avg',0):>4.0f} MHz, {rp.get('temp_avg',0):.0f}C) | "
                  f"decode {rd['throughput_tok_s']:>6.0f} tok/s {rd.get('power_avg_w',0):>6.1f} W "
                  f"(act {rd.get('sm_clk_avg',0):>4.0f})", flush=True)
    finally:
        rr = reset()
        print(f"[reset] {'ok' if rr.returncode == 0 else rr.stderr.strip()[:80]}", flush=True)
        sampler.stop(); sampler.shutdown()

    keys = ["workload", "req_clk_mhz", "act_clk_mhz", "throughput_tok_s",
            "power_avg_w", "power_std_w", "temp_avg", "n_samples"]
    with open(os.path.join(C.RESULTS_DIR, "dvfs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"wrote {C.RESULTS_DIR}/dvfs.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
