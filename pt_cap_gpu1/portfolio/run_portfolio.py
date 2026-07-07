"""Run the POWER-CAP SWEEP for every workload in the portfolio -> per-workload CSVs.

For each workload we hold a FIXED heavy config (prefill: fixed seq_len+batch; decode: fixed
ctx+batch) and sweep ONLY the power cap (nvidia-smi -pl). At each cap the card draws ~that power
and picks the SM clock that fits, so power AND throughput both move -> one single-valued P<->T
curve per (workload, phase). This is exactly the method validated in ../plot_theory.py, generalised
across the portfolio so we can re-run the theory-vs-measured fit on each workload type.

Writes  data/<id>_prefill.csv  and  data/<id>_decode.csv  (one row per cap).

  # one GPU, whole portfolio:
  SUDO_PASS=... CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../code python3 run_portfolio.py

  # parallelise across the 4 V100s: pin a slice of the portfolio to each GPU
  SUDO_PASS=... CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../code python3 run_portfolio.py --ids chat-phi3,code-phi3
  SUDO_PASS=... CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../code python3 run_portfolio.py --ids rag-phi3,longrag-qwen7b
  ...

Telemetry (pynvml) and the -pl target both follow CUDA_VISIBLE_DEVICES (power_sampler picks the
physical NVML index from it), so each GPU is measured and capped independently -> safe to run 4 at once.
"""
from __future__ import annotations
import argparse, csv, os, subprocess, time
import torch
import pynvml

import config as C                                   # noqa: E402  (from ../../code via PYTHONPATH)
C.WARMUP_S = 0.5; C.SETTLE_S = 0.1; C.MEASURE_S = 2.5   # short bursts -> stay cool, clean cap-limited points

from portfolio import PORTFOLIO, CAP_GRID            # noqa: E402
from power_sampler import PowerSampler                # noqa: E402  (auto-targets the CVD GPU)

PW = os.environ.get("SUDO_PASS", "")
GPU = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")   # physical index for nvidia-smi
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
COOL_TARGET, COOL_HOT, COOL_MAX_S = 48.0, 60.0, 60.0


def sudo_pl(w):
    return subprocess.run(["sudo", "-S", "-p", "", "nvidia-smi", "-i", GPU, "-pl", str(int(w))],
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


def load_model_for(model_id):
    """Load an arbitrary model (portfolio overrides the config default)."""
    C.MODEL_ID = model_id
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=C.DTYPE, attn_implementation=C.ATTN_IMPL).to(C.DEVICE).eval()
    return tok, model


def run_workload(w, sampler, caps):
    """Cap sweep for one workload: prefill (fixed S,B) then decode (fixed C,B), both single-valued in cap."""
    from measure import run_prefill_point, run_decode_point, free   # imported after config is set
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"\n########## {w['id']}  ({w['name_en']})  model={w['model_id']} ##########", flush=True)
    tok, model = load_model_for(w["model_id"])
    vocab = model.config.vocab_size
    free()
    print(f"  loaded  VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    pre_rows, dec_rows = [], []
    ps, pb = w["prefill_seq_len"], w["prefill_batch"]
    dc, db = w["decode_ctx"], w["decode_batch"]
    try:
        for cap in caps:
            if sudo_pl(cap).returncode != 0:
                print(f"  -pl {cap} FAILED"); continue
            time.sleep(0.4)
            print(f"  === CAP {cap} W ===", flush=True)
            # ---- prefill: fixed (S,B), compute-bound ----
            free(); cool(sampler)
            try:
                r = run_prefill_point(model, sampler, pb, ps, vocab)
                pre_rows.append(_row("prefill", cap, r, seq_len=ps, batch=pb))
                print(f"    prefill b={pb} S={ps:<5} | {r['throughput_tok_s']:>8.0f} tok/s | "
                      f"{r.get('power_avg_w',0):>5.0f}W | sm {r.get('sm_clk_avg',0):.0f} "
                      f"| {r.get('tok_per_joule',0):.1f} tok/J", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"    prefill OOM at b={pb} S={ps}"); free()
            # ---- decode: fixed (C,B), memory-bound ----
            free(); cool(sampler)
            try:
                r = run_decode_point(model, sampler, db, dc, vocab)
                dec_rows.append(_row("decode", cap, r, ctx=dc, batch=db))
                print(f"    decode  b={db} C={dc:<5} | {r['throughput_tok_s']:>8.0f} tok/s | "
                      f"{r.get('power_avg_w',0):>5.0f}W | sm {r.get('sm_clk_avg',0):.0f} "
                      f"| {r.get('tok_per_joule',0):.1f} tok/J", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"    decode OOM at b={db} C={dc} -> lower decode_batch"); free()
    finally:
        del model
        free()
    _write(os.path.join(DATA, f"{w['id']}_prefill.csv"), pre_rows)
    _write(os.path.join(DATA, f"{w['id']}_decode.csv"), dec_rows)


def _row(phase, cap, r, **extra):
    return {"phase": phase, "cap_w": cap,
            "throughput_tok_s": round(r["throughput_tok_s"], 1),
            "power_avg_w": round(r.get("power_avg_w", 0), 1),
            "sm_clk_avg": round(r.get("sm_clk_avg", 0)),
            "mem_clk_avg": round(r.get("mem_clk_avg", 0)),
            "util_gpu_avg": round(r.get("util_gpu_avg", 0)),
            "util_mem_avg": round(r.get("util_mem_avg", 0)),
            "tok_per_joule": round(r.get("tok_per_joule", 0), 2),
            "temp_avg": round(r.get("temp_avg", 0), 1), **extra}


def _write(path, rows):
    if not rows:
        print(f"  (no rows for {os.path.basename(path)})"); return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"  wrote {path} ({len(rows)} caps)", flush=True)


def _done(wid):
    """A workload is done when its decode CSV exists with >=1 data row (resume/idempotent retries)."""
    p = os.path.join(DATA, f"{wid}_decode.csv")
    try:
        return os.path.exists(p) and sum(1 for _ in open(p)) >= 2
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="", help="comma-separated workload ids to run (default: all)")
    ap.add_argument("--force", action="store_true", help="re-run even if output CSVs already exist")
    args = ap.parse_args()
    if not PW:
        print("ERROR: set SUDO_PASS (needed for nvidia-smi -pl)"); return
    os.makedirs(DATA, exist_ok=True)
    wanted = [x.strip() for x in args.ids.split(",") if x.strip()]
    todo = [w for w in PORTFOLIO if (not wanted or w["id"] in wanted)]
    if not args.force:
        skip = [w["id"] for w in todo if _done(w["id"])]
        if skip:
            print(f"[resume] skipping already-done: {skip}", flush=True)
        todo = [w for w in todo if not _done(w["id"])]
    if not todo:
        print(f"nothing to run (all done or no match for --ids {args.ids}); "
              f"known: {[w['id'] for w in PORTFOLIO]}"); return

    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(int(GPU))
    mn, mx = [x / 1000.0 for x in pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)]
    # Restore the cap that was in force BEFORE we started (not the factory default) — on this shared
    # box the GPUs sit at an admin-set 250 W while the factory default is 300 W, so resetting to the
    # default would leave our GPU inconsistent with the others.
    start_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
    caps = [c for c in CAP_GRID if mn <= c <= mx]
    sampler = PowerSampler(interval_s=C.SAMPLE_INTERVAL_S); sampler.start(); time.sleep(0.3)
    print(f"GPU{GPU} {sampler.name} | cap range [{mn:.0f},{mx:.0f}] restore-to {start_limit:.0f}W | caps {caps}\n"
          f"workloads: {[w['id'] for w in todo]}", flush=True)
    try:
        for w in todo:
            run_workload(w, sampler, caps)
    finally:
        sudo_pl(round(start_limit))
        print(f"\n[reset] -pl {start_limit:.0f}W", flush=True)
        sampler.stop(); sampler.shutdown(); pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
