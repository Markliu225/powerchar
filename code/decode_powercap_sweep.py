"""Decode at FULL batch, SUSTAINED power as the control variable — even power grid, frequency actuator.

Power is reported as the SUSTAINED (active) power = the MEDIAN of the high-rate NVML samples while the
GPU is decoding — this is what nvidia-smi shows and what the power wall actually limits. (The window
MEAN under-reads it because the eager loop has GPU-idle gaps between single-token steps; mean is kept
only for the energy column.) For each EVEN power target we tune the SM clock (closed loop) until the
sustained power hits the target, then record throughput + efficiency. (-pl floors at 100 W, so we
actuate via -lgc.) Because power is the control variable, the points come out EVENLY spaced in power.

Writes pt_cap_gpu1/decode_fixedbatch.csv incrementally (power_w = sustained; power_mean_w / power_max_w
also recorded) -> plot with plot_decode_fixedbatch.py.

  SUDO_PASS=... CUDA_VISIBLE_DEVICES=0 PYTHONPATH=code python3 code/decode_powercap_sweep.py
"""
from __future__ import annotations
import csv, os, subprocess, time
import numpy as np
import torch
import pynvml
import config as C
from power_sampler import PowerSampler                            # noqa: E402
from measure import load_model, run_decode_point, free           # noqa: E402

PW = os.environ.get("SUDO_PASS", "")
GPU = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")
BATCH = int(os.environ.get("DECODE_BATCH", "96"))
CTX = 256
TARGETS = [float(x) for x in os.environ.get(
    "TARGETS", "50,70,90,110,130,150,170,190,210").split(",")]      # W, sustained-power grid
TOL = 4.0
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "pt_cap_gpu1", "decode_fixedbatch.csv")
SEED = os.path.join(HERE, "pt_cap_gpu1", "decode_clk_map.csv")    # clk<->sustained-power seed (read-only)


def sudo(args):
    return subprocess.run(["sudo", "-S", "-p", ""] + args, input=PW + "\n", text=True, capture_output=True)


def psus(r):
    """Sustained (active) power = median of the in-window NVML samples (fallback to mean)."""
    return r.get("power_p50_w", r.get("power_avg_w", 0.0))


def seed_map():
    if os.path.exists(SEED):
        rs = list(csv.DictReader(open(SEED)))
        cp = sorted((float(r["power_w"]), float(r["sm_clk_avg"])) for r in rs if float(r["sm_clk_avg"]) > 0)
        pw, ck = [], []
        for p, c in cp:
            if not pw or p > pw[-1] + 0.3:
                pw.append(p); ck.append(c)
        if len(pw) >= 2:
            return np.array(pw), np.array(ck)
    return None, None


def main():
    if not PW:
        print("ERROR: set SUDO_PASS", flush=True); return
    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(int(GPU))
    mem = sorted(pynvml.nvmlDeviceGetSupportedMemoryClocks(h))[-1]
    sup = sorted(set(pynvml.nvmlDeviceGetSupportedGraphicsClocks(h, mem)))
    pows0, clks0 = seed_map()
    torch.backends.cuda.matmul.allow_tf32 = True
    tok, model = load_model(); vocab = model.config.vocab_size
    sampler = PowerSampler(interval_s=0.01); sampler.start(); time.sleep(0.3)        # 100 Hz -> clean p50/max
    print(f"GPU{GPU} {sampler.name} | batch={BATCH} ctx={CTX} | sustained-power targets {TARGETS} W", flush=True)

    def probe(clk, warm, meas):
        sudo(["nvidia-smi", "-i", GPU, "-lgc", f"{clk},{clk}"]); time.sleep(0.3); free()
        C.WARMUP_S, C.SETTLE_S, C.MEASURE_S = warm, 0.1, meas
        return run_decode_point(model, sampler, BATCH, CTX, vocab)

    keys = ["target_w", "batch", "ctx", "throughput_tok_s", "power_w", "power_mean_w", "power_max_w",
            "sm_clk_avg", "mem_clk_avg", "util_gpu_avg", "tok_per_joule", "temp_avg"]
    fcsv = open(CSV, "w", newline=""); wr = csv.DictWriter(fcsv, fieldnames=keys); wr.writeheader(); fcsv.flush()
    done = 0
    try:
        for tgt in TARGETS:
            clk = float(np.interp(tgt, pows0, clks0)) if pows0 is not None else (sup[0] + sup[-1]) / 2
            i = int(np.argmin([abs(c - clk) for c in sup]))
            r = probe(sup[i], 0.2, 0.7); p = psus(r)
            for _ in range(5):
                if abs(p - tgt) <= TOL:
                    break
                step = 1 if abs(p - tgt) < 12 else 2
                j = min(i + step, len(sup) - 1) if p < tgt else max(i - step, 0)
                if j == i:
                    break
                i = j; r = probe(sup[i], 0.2, 0.7); p = psus(r)
            rf = probe(sup[i], 0.4, 2.0)
            ps = psus(rf); T = rf["throughput_tok_s"]
            row = {"target_w": tgt, "batch": BATCH, "ctx": CTX, "throughput_tok_s": round(T, 1),
                   "power_w": round(ps, 1), "power_mean_w": round(rf.get("power_avg_w", 0), 1),
                   "power_max_w": round(rf.get("power_max_w", 0), 1), "sm_clk_avg": round(rf.get("sm_clk_avg", 0)),
                   "mem_clk_avg": round(rf.get("mem_clk_avg", 0)), "util_gpu_avg": round(rf.get("util_gpu_avg", 0)),
                   "tok_per_joule": round(T / ps, 2) if ps > 0 else 0, "temp_avg": round(rf.get("temp_avg", 0), 1)}
            wr.writerow(row); fcsv.flush(); done += 1
            print(f"  target {tgt:>3.0f}W -> sustained {ps:>5.1f}W (mean {rf.get('power_avg_w',0):.0f}, "
                  f"max {rf.get('power_max_w',0):.0f}) @ {rf.get('sm_clk_avg',0):.0f}MHz | {T:>6.0f} tok/s "
                  f"| {row['tok_per_joule']:.1f} tok/J", flush=True)
    finally:
        sudo(["nvidia-smi", "-i", GPU, "-rgc"]); fcsv.close()
        print(f"\n[reset] -rgc; wrote {CSV} ({done} power targets)", flush=True)
        sampler.stop(); sampler.shutdown(); pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
