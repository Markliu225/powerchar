"""DVFS / frequency-knob sweep with REPEATS + averaging.

Locks the SM clock (via `sudo nvidia-smi -lgc`, password from SUDO_PASS) and, at each
clock, measures a fixed compute-bound prefill workload and a fixed memory-bound decode
workload N_REPEAT times, then records the MEAN and STD across repeats (to average out
run-to-run noise). Light prefill load so the top clocks stay under the 250 W cap.

  SUDO_PASS=... CUDA_VISIBLE_DEVICES=1 PYTHONPATH=code python3 code/dvfs_sweep.py
"""
from __future__ import annotations
import csv, os, statistics, subprocess, time
import torch
import config as C
from power_sampler import PowerSampler
from measure import load_model, run_prefill_point, run_decode_point, free
import os

FREQS = [510, 660, 810, 960, 1110, 1260, 1410, 1530]   # MHz, requested SM clocks
PREFILL_WL = dict(batch=4, seq_len=256)
DECODE_WL  = dict(batch=16, ctx_len=256)
N_REPEAT = 3                                            # repeats per clock, averaged
COOL_TARGET_C, COOL_MAX_S = 60.0, 30.0
PW = os.environ.get("SUDO_PASS", "")


def _sudo(args):
    return subprocess.run(["sudo", "-S", "-p", "", *args], input=PW + "\n",
                          text=True, capture_output=True)
def lock(f):  return _sudo(["nvidia-smi", "-i", (os.environ.get("CUDA_VISIBLE_DEVICES","0").split(",")[0] or "0"), "-lgc", str(f)])
def reset():  return _sudo(["nvidia-smi", "-i", (os.environ.get("CUDA_VISIBLE_DEVICES","0").split(",")[0] or "0"), "-rgc"])


def cooldown(sampler):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < COOL_MAX_S:
        t = sampler.samples[-1]["temp"] if sampler.samples else 0
        if t and t <= COOL_TARGET_C:
            break
        time.sleep(0.5)


def agg(vals):
    return round(statistics.mean(vals), 2), round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 2)


def main():
    if not PW:
        print("ERROR: set SUDO_PASS env var"); return
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"loading model... (N_REPEAT={N_REPEAT} per clock)", flush=True)
    tok, model = load_model()
    vocab = model.config.vocab_size
    sampler = PowerSampler(interval_s=C.SAMPLE_INTERVAL_S); sampler.start(); time.sleep(0.3)
    print(f"GPU {sampler.name} cap {sampler.power_limit_w:.0f} W | clocks {FREQS}", flush=True)

    rows = []
    try:
        for i, f in enumerate(FREQS, 1):
            r = lock(f)
            if r.returncode != 0:
                print(f"  lock {f} FAILED: {r.stderr.strip()[:80]}"); continue
            acc = {"prefill": {"T": [], "P": [], "clk": [], "tmp": []},
                   "decode":  {"T": [], "P": [], "clk": [], "tmp": []}}
            for _ in range(N_REPEAT):
                cooldown(sampler); free()
                rp = run_prefill_point(model, sampler, PREFILL_WL["batch"], PREFILL_WL["seq_len"], vocab)
                free()
                rd = run_decode_point(model, sampler, DECODE_WL["batch"], DECODE_WL["ctx_len"], vocab)
                for wl, rr in (("prefill", rp), ("decode", rd)):
                    acc[wl]["T"].append(rr["throughput_tok_s"]); acc[wl]["P"].append(rr.get("power_avg_w", 0))
                    acc[wl]["clk"].append(rr.get("sm_clk_avg", 0)); acc[wl]["tmp"].append(rr.get("temp_avg", 0))
            for wl in ("prefill", "decode"):
                a = acc[wl]
                Tm, Ts = agg(a["T"]); Pm, Ps = agg(a["P"]); cm, _ = agg(a["clk"]); tm, _ = agg(a["tmp"])
                rows.append({"workload": wl, "req_clk_mhz": f, "act_clk_mhz": round(cm),
                             "throughput_tok_s": round(Tm, 1), "throughput_std": round(Ts, 1),
                             "power_avg_w": round(Pm, 2), "power_std_w": round(Ps, 2),
                             "temp_avg": round(tm, 1), "n_rep": N_REPEAT})
            rp_, rd_ = rows[-2], rows[-1]
            print(f"  [{i}/{len(FREQS)}] req {f:>4} | prefill {rp_['throughput_tok_s']:>7.0f}±{rp_['throughput_std']:>4.0f} tok/s "
                  f"{rp_['power_avg_w']:>6.1f}±{rp_['power_std_w']:>4.1f} W (act {rp_['act_clk_mhz']:>4}) | "
                  f"decode {rd_['throughput_tok_s']:>6.0f}±{rd_['throughput_std']:>4.0f} {rd_['power_avg_w']:>6.1f}±{rd_['power_std_w']:>4.1f} W", flush=True)
    finally:
        rr = reset()
        print(f"[reset] {'ok' if rr.returncode == 0 else rr.stderr.strip()[:80]}", flush=True)
        sampler.stop(); sampler.shutdown()

    keys = ["workload", "req_clk_mhz", "act_clk_mhz", "throughput_tok_s", "throughput_std",
            "power_avg_w", "power_std_w", "temp_avg", "n_rep"]
    with open(os.path.join(C.RESULTS_DIR, "dvfs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"wrote {C.RESULTS_DIR}/dvfs.csv ({len(rows)} rows, {N_REPEAT}x averaged)", flush=True)


if __name__ == "__main__":
    main()
