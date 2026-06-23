"""Top-up the high-clock prefill points from COLD with strong cooldown.

The dense 3x DVFS run let the card heat-saturate, so the top clocks (1260/1530)
thermal-throttled. Here we re-measure ONLY those clocks, each repeat starting from
a hard cooldown to <=50 C, so the short measurement window holds the locked clock.
Same fixed workload as dvfs_sweep (B=4, S=256) so the point is comparable.
Caveat: at the very top clock the 250 W POWER cap (not temperature) may still force
the clock down — cooling cannot beat a power cap; we record the achieved clock.

Updates the prefill rows for these clocks in results/dvfs.csv.

  SUDO_PASS=... CUDA_VISIBLE_DEVICES=0 PYTHONPATH=code python3 code/dvfs_topup.py
"""
from __future__ import annotations
import csv, os, statistics, subprocess, time
import torch
import config as C
from power_sampler import PowerSampler
from measure import load_model, run_prefill_point, free

FREQS_TOP = [1260, 1530]
PREFILL_WL = dict(batch=4, seq_len=256)          # SAME fixed workload as dvfs_sweep
N_REPEAT = 3
COOL_TARGET_C, COOL_MAX_S = 50.0, 150.0          # hard cold start
PW = os.environ.get("SUDO_PASS", "")


def _sudo(a): return subprocess.run(["sudo", "-S", "-p", "", *a], input=PW + "\n", text=True, capture_output=True)
def lock(f):  return _sudo(["nvidia-smi", "-i", "0", "-lgc", str(f)])
def reset():  return _sudo(["nvidia-smi", "-i", "0", "-rgc"])


def cool(sampler, target, cap_s):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < cap_s:
        t = sampler.samples[-1]["temp"] if sampler.samples else 0
        if t and t <= target:
            return t, time.perf_counter() - t0
        time.sleep(0.5)
    return (sampler.samples[-1]["temp"] if sampler.samples else 0), time.perf_counter() - t0


def main():
    if not PW:
        print("ERROR: set SUDO_PASS"); return
    torch.backends.cuda.matmul.allow_tf32 = True
    print("loading model...", flush=True)
    tok, model = load_model(); vocab = model.config.vocab_size
    sampler = PowerSampler(interval_s=C.SAMPLE_INTERVAL_S); sampler.start(); time.sleep(0.3)
    newrows = {}
    try:
        for f in FREQS_TOP:
            if lock(f).returncode != 0:
                print(f"lock {f} failed"); continue
            T, P, A, Tmp = [], [], [], []
            for rep in range(N_REPEAT):
                free()
                ct, cs = cool(sampler, COOL_TARGET_C, COOL_MAX_S)
                r = run_prefill_point(model, sampler, PREFILL_WL["batch"], PREFILL_WL["seq_len"], vocab)
                T.append(r["throughput_tok_s"]); P.append(r.get("power_avg_w", 0))
                A.append(r.get("sm_clk_avg", 0)); Tmp.append(r.get("temp_avg", 0))
                print(f"  f={f} rep{rep+1}: cold {ct:.0f}C/{cs:.0f}s -> T={r['throughput_tok_s']:.0f} "
                      f"P={r.get('power_avg_w',0):.1f}W act={r.get('sm_clk_avg',0):.0f} "
                      f"temp {r.get('temp_avg',0):.0f}C", flush=True)
            sd = lambda v: statistics.pstdev(v) if len(v) > 1 else 0.0
            newrows[f] = {"workload": "prefill", "req_clk_mhz": f, "act_clk_mhz": round(statistics.mean(A)),
                          "throughput_tok_s": round(statistics.mean(T), 1), "throughput_std": round(sd(T), 1),
                          "power_avg_w": round(statistics.mean(P), 2), "power_std_w": round(sd(P), 2),
                          "temp_avg": round(statistics.mean(Tmp), 1), "n_rep": N_REPEAT}
            held = "HELD lock" if newrows[f]["act_clk_mhz"] >= f - 30 else "still throttled (power cap)"
            print(f"  => f={f}: act {newrows[f]['act_clk_mhz']} ({held}), "
                  f"T={newrows[f]['throughput_tok_s']}±{newrows[f]['throughput_std']} "
                  f"P={newrows[f]['power_avg_w']}±{newrows[f]['power_std_w']}W", flush=True)
    finally:
        print(f"[reset] {'ok' if reset().returncode == 0 else 'FAILED'}", flush=True)
        sampler.stop(); sampler.shutdown()

    # merge into dvfs.csv (replace prefill rows for these clocks)
    path = os.path.join(C.RESULTS_DIR, "dvfs.csv")
    rows = list(csv.DictReader(open(path)))
    fieldnames = rows[0].keys()
    for r in rows:
        if r["workload"] == "prefill" and int(float(r["req_clk_mhz"])) in newrows:
            r.update({k: str(v) for k, v in newrows[int(float(r["req_clk_mhz"]))].items()})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fieldnames)); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"updated {path} (top-up {list(newrows)})", flush=True)


if __name__ == "__main__":
    main()
