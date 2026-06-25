"""Burst vs paced scheduling for a FIXED total workload -> compare job completion time (JCT).

Same total work = N_TOTAL identical fp16 GEMMs, two strategies:
  A "BURST"  : dump it all, run flat-out, IGNORE cooling. The die heats, thermal-throttles, the
               SM clock collapses -> each GEMM gets slower, but 100% duty (never idle).
  B "PACED"  : feed it in chunks; whenever temp reaches a soft cap (well below the 83C throttle),
               PAUSE to cool back down, then resume. Clock stays high (no throttle), but pays
               idle cooling time.
Both start from the same baseline temperature (pre-cooled). We log temp/clock/power/work-done
over time and report JCT_A vs JCT_B. The question: does staying cool (high clock, much idle) beat
powering through the throttle (low clock, zero idle)?

  CUDA_VISIBLE_DEVICES=0 python3 burst_vs_paced/run_experiment.py
Writes timeline.csv + meta.json (no sudo needed; we do NOT lock the clock).
"""
from __future__ import annotations
import csv, json, os, threading, time
import pynvml

HERE = os.path.dirname(os.path.abspath(__file__))
GEMM_N = 8192            # fp16 matmul size
N_TOTAL = 4000           # FIXED total work (number of matmuls) — long enough to expose the steady state
INNER = 4               # matmuls per small chunk (PACED feeds in small bites; also BURST sync grain)
BASE_C = 72.0           # pre-cool to this before each run (reachable even when heat-soaked -> no wasted wait)
T_TARGET = 81.0         # PACED: hold temperature here (just below the 83C throttle) by trickle-feeding
SLEEP_STEP = 0.02       # PACED: integral-controller step for the inter-chunk pause (s)
SLEEP_MAX = 1.0         # PACED: cap on the per-chunk pause
COOL_MAX_S = 90.0       # safety cap on any single pre-cool wait

state = {"strategy": "-", "phase": "init", "work": 0}    # shared with the sampler


class Sampler(threading.Thread):
    def __init__(self, h):
        super().__init__(daemon=True)
        self.h = h; self.stop_flag = False; self.samples = []; self.t0 = time.perf_counter()

    def temp(self):
        return pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU)

    def run(self):
        i = 0
        while not self.stop_flag:
            t = time.perf_counter() - self.t0
            tp = self.temp()
            clk = pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM)
            try:
                pw = pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
            except pynvml.NVMLError:
                pw = float("nan")
            self.samples.append((t, state["strategy"], state["phase"], tp, clk, pw, state["work"]))
            if i % 10 == 0:
                print(f"  [{state['strategy']}/{state['phase']:<7}] t={t:6.1f}s temp={tp:3d}C clk={clk:4d}MHz "
                      f"pw={pw:5.0f}W work={state['work']}/{N_TOTAL}", flush=True)
            i += 1
            time.sleep(0.1)


def main():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.randn(GEMM_N, GEMM_N, device="cuda", dtype=torch.float16)
    b = torch.randn(GEMM_N, GEMM_N, device="cuda", dtype=torch.float16)
    c = torch.empty_like(a)

    s = Sampler(h); s.start()

    def cool_to(target, tag):
        state["strategy"], state["phase"] = tag, "precool"
        print(f"== [{tag}] pre-cool to <= {target:.0f}C ==", flush=True)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < COOL_MAX_S:
            if s.temp() <= target:
                break
            time.sleep(1.0)

    def burn(n):
        for _ in range(n):
            import torch
            torch.matmul(a, b, out=c)
        import torch
        torch.cuda.synchronize()

    results = {}

    # ---------- Strategy B: PACED — trickle small chunks, hold temp at T_TARGET -- run FIRST ----------
    cool_to(BASE_C, "B")
    state["strategy"], state["phase"], state["work"] = "B", "run", 0
    print(f"== [B] PACED: trickle {INNER}-GEMM chunks, integral-control pause to hold {T_TARGET:.0f}C ==", flush=True)
    t0 = time.perf_counter()
    done = 0
    idle_s = 0.0
    pause = 0.0                                   # adaptive inter-chunk pause (s)
    while done < N_TOTAL:
        n = min(INNER, N_TOTAL - done)
        burn(n); done += n; state["work"] = done
        t = s.temp()
        if t > T_TARGET:                          # too hot -> feed slower (longer pause)
            pause = min(pause + SLEEP_STEP, SLEEP_MAX)
        elif t < T_TARGET - 1:                     # cool enough -> feed faster (shorter pause)
            pause = max(pause - SLEEP_STEP, 0.0)
        state["phase"] = "cool" if pause > 0 else "run"
        if pause > 0:
            time.sleep(pause); idle_s += pause
    results["B"] = time.perf_counter() - t0
    print(f"== [B] DONE  JCT_B = {results['B']:.1f}s  (of which trickle-pause {idle_s:.1f}s) ==", flush=True)
    cool_s = idle_s

    # ---------- Strategy A: BURST (ignore cooling) ----------
    cool_to(BASE_C, "A")
    state["strategy"], state["phase"], state["work"] = "A", "run", 0
    print("== [A] BURST: full workload, flat out, ignore cooling ==", flush=True)
    t0 = time.perf_counter()
    done = 0
    while done < N_TOTAL:
        n = min(INNER, N_TOTAL - done)
        burn(n); done += n; state["work"] = done
    results["A"] = time.perf_counter() - t0
    print(f"== [A] DONE  JCT_A = {results['A']:.1f}s ==", flush=True)

    s.stop_flag = True; s.join(); pynvml.nvmlShutdown()

    with open(os.path.join(HERE, "timeline.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "strategy", "phase", "temp_c", "sm_clk_mhz", "power_w", "work_done"])
        for r in s.samples:
            w.writerow([round(r[0], 2), r[1], r[2], r[3], r[4], round(r[5], 1) if r[5] == r[5] else "", r[6]])
    meta = dict(n_total=N_TOTAL, gemm_n=GEMM_N, base_c=BASE_C, t_target_c=T_TARGET,
                jct_burst_s=round(results["A"], 1), jct_paced_s=round(results["B"], 1),
                paced_cooling_s=round(cool_s, 1),
                speedup_burst_over_paced=round(results["B"] / results["A"], 2))
    json.dump(meta, open(os.path.join(HERE, "meta.json"), "w"), indent=2)
    print(f"\nJCT  BURST={results['A']:.1f}s   PACED={results['B']:.1f}s   "
          f"-> burst is {meta['speedup_burst_over_paced']}x {'FASTER' if results['B']>results['A'] else 'SLOWER'}", flush=True)
    print("wrote timeline.csv + meta.json", flush=True)


if __name__ == "__main__":
    main()
