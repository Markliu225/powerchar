"""Run the POWER-CAP SWEEP for every workload in the portfolio -> per-workload CSVs.

For each workload we hold a FIXED heavy config (prefill: fixed seq_len+batch; decode: fixed
ctx+batch) and sweep ONLY the power cap (nvidia-smi -pl). At each cap the card draws ~that power
and picks the SM clock that fits, so power AND throughput both move -> one single-valued P<->T
curve per (workload, phase). Measurement methodology v3 (see DATA_QUALITY.zh.md) plus the H200
manual's telemetry rules (energy-accumulator window power, throttle-reason gating, meta.json).

GPU-agnostic: nothing V100-specific is hardcoded. The cap grid derives from the device's own
power-limit constraints (--caps auto), cooling thresholds derive from the device's slowdown
threshold, and the max SM clock is written to <outdir>/meta.json for the fitting scripts.

Writes  <outdir>/<id>_prefill.csv  and  <outdir>/<id>_decode.csv  (one row per cap) + meta.json.

  # one GPU, whole portfolio (resume-aware):
  SUDO_PASS=... CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../code python3 run_portfolio.py --outdir data_h200

  # slice the portfolio across GPUs (each writes to the same outdir, distinct files):
  SUDO_PASS=... CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../code python3 run_portfolio.py --ids rag-phi3,code-phi3

Privilege for `nvidia-smi -pl`: run as root, OR passwordless sudo, OR SUDO_PASS=<password>.
Telemetry (pynvml) and the -pl target both follow CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, time

# MUST precede the torch import. The HF DynamicCache torch.cat's the whole KV every decode step;
# with the default block allocator that realloc pattern is a lottery (rag-shaped workloads showed
# bimodal 225-267 tok/s, +-9%). expandable_segments grows one VA segment smoothly: 272-281 tok/s,
# +-1.6% AND ~15% faster -- the artifact-free measurement. Verified on V100, 8 repeats each.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import pynvml

import config as C                                   # noqa: E402  (from ../../code via PYTHONPATH)
C.WARMUP_S = 0.5; C.SETTLE_S = 0.1; C.MEASURE_S = 2.5   # short bursts -> stay cool, clean cap-limited points
C.DECODE_SEED_CHUNK = 256                            # seed the KV in ONE prefill-sized chunk stream
                                                     # (small-chunk seeding fragments the allocator
                                                     #  and starves the GPU -- DATA_QUALITY.zh.md #1)

from portfolio import PORTFOLIO                       # noqa: E402
from power_sampler import PowerSampler, THERMAL_BITS  # noqa: E402  (auto-targets the CVD GPU)

# ---- decode data-quality knobs (v3) --------------------------------------------------------------
# Step-targeted windows kill the quantization noise of slow points (a 6 tok/s b4 point does ~4
# steps in a 2.5 s window -> +-25% noise); bounding the steps also bounds the KV-growth drift.
DEC_TARGET_STEPS = 32      # every decode point measures >=32 steps (and >=MEASURE_S seconds)
DEC_MAX_S = 45.0           # hard cap per window (slowest point: ~1.2 steps/s -> ~38 steps)
DEC_REPEATS = 2            # measure each cap twice; if spread >5% take a 3rd and use the median
THERMAL_REMEASURE_FRAC = 0.05   # re-measure a point once if >5% of its samples show thermal bits

PW = os.environ.get("SUDO_PASS", "")
GPU = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")   # physical index for nvidia-smi
HERE = os.path.dirname(os.path.abspath(__file__))

# cooling thresholds: derived from the device's slowdown threshold in main(); these are fallbacks
COOL_TARGET, COOL_HOT, COOL_MAX_S = 48.0, 60.0, 90.0


def _pl_cmd(w):
    return ["nvidia-smi", "-i", GPU, "-pl", str(int(w))]


def sudo_pl(w):
    """Set the power cap: direct when root, else passwordless sudo, else sudo with SUDO_PASS."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return subprocess.run(_pl_cmd(w), text=True, capture_output=True)
    if not PW:
        return subprocess.run(["sudo", "-n"] + _pl_cmd(w), text=True, capture_output=True)
    return subprocess.run(["sudo", "-S", "-p", ""] + _pl_cmd(w),
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


def _thermally_clean(r):
    """Manual section 7.4: a point is accepted when thermal/protective throttle bits stay absent
    (power-cap / clocks-setting bits are the EXPECTED reasons during a cap sweep)."""
    return r.get("thermal_frac", 0.0) <= THERMAL_REMEASURE_FRAC


def _decode_median(model, sampler, db, dc, vocab):
    """Measure the decode point DEC_REPEATS times; if the throughput spread exceeds 5% take a
    third run and use the median-by-throughput row. Returns (chosen_result, spread_pct, n_runs)."""
    from measure import run_decode_point, free
    runs = []
    for _ in range(DEC_REPEATS):
        free()
        runs.append(run_decode_point(model, sampler, db, dc, vocab,
                                     target_steps=DEC_TARGET_STEPS, max_s=DEC_MAX_S))
    ts = [r["throughput_tok_s"] for r in runs]
    spread = (max(ts) - min(ts)) / (sum(ts) / len(ts)) * 100
    if spread > 5.0:
        free()
        runs.append(run_decode_point(model, sampler, db, dc, vocab,
                                     target_steps=DEC_TARGET_STEPS, max_s=DEC_MAX_S))
        ts = [r["throughput_tok_s"] for r in runs]
        spread = (max(ts) - min(ts)) / (sum(ts) / len(ts)) * 100
    chosen = sorted(runs, key=lambda r: r["throughput_tok_s"])[len(runs) // 2]
    return chosen, spread, len(runs)


def run_workload(w, sampler, caps, outdir, phase="both"):
    """Cap sweep for one workload: prefill (fixed S,B) then decode (fixed C,B), both single-valued in cap."""
    from measure import run_prefill_point, free   # imported after config is set
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
            if phase in ("prefill", "both"):
                free(); cool(sampler)
                try:
                    r = run_prefill_point(model, sampler, pb, ps, vocab)
                    if not _thermally_clean(r):            # manual 7.4: thermal bits -> cool + retry once
                        print("    prefill point thermally contaminated -> cooling + re-measuring", flush=True)
                        free(); cool(sampler)
                        r = run_prefill_point(model, sampler, pb, ps, vocab)
                    pre_rows.append(_row("prefill", cap, r, seq_len=ps, batch=pb))
                    print(f"    prefill b={pb} S={ps:<5} | {r['throughput_tok_s']:>8.0f} tok/s | "
                          f"{r.get('power_avg_w',0):>5.0f}W | sm {r.get('sm_clk_avg',0):.0f} "
                          f"| {r.get('tok_per_joule',0):.1f} tok/J", flush=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"    prefill OOM at b={pb} S={ps}"); free()
            # ---- decode: fixed (C,B), memory-bound; step-targeted window + repeats ----
            if phase in ("decode", "both"):
                free(); cool(sampler)
                try:
                    r, spread, nrun = _decode_median(model, sampler, db, dc, vocab)
                    if not _thermally_clean(r):
                        print("    decode point thermally contaminated -> cooling + re-measuring", flush=True)
                        free(); cool(sampler)
                        r, spread, nrun = _decode_median(model, sampler, db, dc, vocab)
                    dec_rows.append(_row("decode", cap, r, ctx=dc, batch=db,
                                         ctx_eff=round(r.get("ctx_eff", dc), 1),
                                         steps=r.get("steps", 0),
                                         window_s=round(r.get("wall_s", 0), 2),
                                         spread_pct=round(spread, 2), n_runs=nrun))
                    print(f"    decode  b={db} C={dc:<5} | {r['throughput_tok_s']:>8.0f} tok/s | "
                          f"{r.get('power_avg_w',0):>5.0f}W | sm {r.get('sm_clk_avg',0):.0f} "
                          f"| {r.get('steps',0)} steps/{r.get('wall_s',0):.1f}s "
                          f"| spread {spread:.1f}% ({nrun} runs)", flush=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"    decode OOM at b={db} C={dc} -> lower decode_batch"); free()
    finally:
        del model
        free()
    if pre_rows:
        _write(os.path.join(outdir, f"{w['id']}_prefill.csv"), pre_rows)
    if dec_rows:
        _write(os.path.join(outdir, f"{w['id']}_decode.csv"), dec_rows)


def _row(phase, cap, r, **extra):
    out = {"phase": phase, "cap_w": cap,
           "throughput_tok_s": round(r["throughput_tok_s"], 1),
           "power_avg_w": round(r.get("power_avg_w", 0), 1),           # energy-method when supported
           "power_sample_avg_w": round(r.get("power_sample_avg_w", 0), 1),
           "sm_clk_avg": round(r.get("sm_clk_avg", 0)),
           "mem_clk_avg": round(r.get("mem_clk_avg", 0)),
           "util_gpu_avg": round(r.get("util_gpu_avg", 0)),
           "util_mem_avg": round(r.get("util_mem_avg", 0)),
           "tok_per_joule": round(r.get("tok_per_joule", 0), 2),
           "temp_avg": round(r.get("temp_avg", 0), 1)}
    if "throttle_mask" in r:
        out["throttle_mask"] = hex(r["throttle_mask"])
        out["thermal_frac"] = round(r.get("thermal_frac", 0.0), 3)
    if "mem_temp_avg" in r:
        out["mem_temp_avg"] = round(r["mem_temp_avg"], 1)
    out.update(extra)
    return out


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


def _done(wid, outdir, phase):
    """A workload is done when the CSVs its phase implies exist with >=1 data row."""
    def ok(p):
        try:
            return os.path.exists(p) and sum(1 for _ in open(p)) >= 2
        except OSError:
            return False
    need = {"prefill": ["prefill"], "decode": ["decode"], "both": ["prefill", "decode"]}[phase]
    return all(ok(os.path.join(outdir, f"{wid}_{ph}.csv")) for ph in need)


def device_meta(h):
    """Constraints + identity of the device under test -> meta.json (manual sections 2.3/2.4)."""
    mn, mx = [x / 1000.0 for x in pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)]
    name = pynvml.nvmlDeviceGetName(h)
    drv = pynvml.nvmlSystemGetDriverVersion()
    meta = {
        "gpu_name": name.decode() if isinstance(name, bytes) else name,
        "driver": drv.decode() if isinstance(drv, bytes) else drv,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "pl_min_w": mn, "pl_max_w": mx,
        "pl_default_w": pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000.0,
        "pl_enforced_at_start_w": pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methodology": "v3: seed_chunk=ctx, expandable_segments, step-targeted windows, "
                       "repeats+median, ctx_eff, energy-window power, throttle gating",
    }
    try:
        mems = sorted(pynvml.nvmlDeviceGetSupportedMemoryClocks(h))
        meta["supported_mem_clocks_mhz"] = mems
        sms = sorted(set(pynvml.nvmlDeviceGetSupportedGraphicsClocks(h, mems[-1])))
        meta["sm_clk_min_mhz"], meta["f_max_mhz"] = sms[0], sms[-1]   # f_max feeds the fits (x=f/f_max)
    except pynvml.NVMLError:
        try:
            meta["f_max_mhz"] = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)
        except pynvml.NVMLError:
            pass
    for key, attr in (("temp_slowdown_c", "NVML_TEMPERATURE_THRESHOLD_SLOWDOWN"),
                      ("temp_shutdown_c", "NVML_TEMPERATURE_THRESHOLD_SHUTDOWN"),
                      ("temp_gpu_max_c", "NVML_TEMPERATURE_THRESHOLD_GPU_MAX")):
        try:
            meta[key] = pynvml.nvmlDeviceGetTemperatureThreshold(h, getattr(pynvml, attr))
        except (pynvml.NVMLError, AttributeError):
            pass
    return meta


def main():
    global COOL_TARGET, COOL_HOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="", help="comma-separated workload ids to run (default: all)")
    ap.add_argument("--force", action="store_true", help="re-run even if output CSVs already exist")
    ap.add_argument("--phase", choices=["prefill", "decode", "both"], default="both")
    ap.add_argument("--outdir", default="data", help="output dir (relative to this file), e.g. data_h200")
    ap.add_argument("--caps", default="auto",
                    help="'auto' = N points spanning the device's own [min,max] constraint, "
                         "or an explicit comma list like '200,300,400,500,600,700'")
    ap.add_argument("--grid-n", type=int, default=10, help="number of points for --caps auto")
    args = ap.parse_args()
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # privilege check up-front (root / sudo -n / SUDO_PASS)
    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(int(GPU))
    start_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
    if sudo_pl(round(start_limit)).returncode != 0:
        print("ERROR: cannot run `nvidia-smi -pl` (need root, passwordless sudo, or SUDO_PASS)"); return

    wanted = [x.strip() for x in args.ids.split(",") if x.strip()]
    todo = [w for w in PORTFOLIO if (not wanted or w["id"] in wanted)]
    if not args.force:
        skip = [w["id"] for w in todo if _done(w["id"], outdir, args.phase)]
        if skip:
            print(f"[resume] skipping already-done: {skip}", flush=True)
        todo = [w for w in todo if not _done(w["id"], outdir, args.phase)]
    if not todo:
        print(f"nothing to run (all done or no match for --ids {args.ids}); "
              f"known: {[w['id'] for w in PORTFOLIO]}"); return

    meta = device_meta(h)
    mn, mx = meta["pl_min_w"], meta["pl_max_w"]
    if args.caps == "auto":
        n = max(args.grid_n, 4)
        caps = sorted({int(round(mn + (mx - mn) * i / (n - 1))) for i in range(n)})
    else:
        caps = [c for c in (int(x) for x in args.caps.split(",")) if mn <= c <= mx]
    meta["cap_grid_w"] = caps
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    # cooling thresholds relative to the device's own slowdown threshold (fallback: V100 constants)
    slow = meta.get("temp_slowdown_c")
    if slow:
        COOL_HOT = min(slow - 20, 75)
        COOL_TARGET = COOL_HOT - 12

    sampler = PowerSampler(interval_s=C.SAMPLE_INTERVAL_S); sampler.start(); time.sleep(0.3)
    print(f"GPU{GPU} {sampler.name} | cap range [{mn:.0f},{mx:.0f}] restore-to {start_limit:.0f}W\n"
          f"caps {caps}\n"
          f"energy-window power: {sampler.has_energy} | throttle telemetry: {sampler.has_throttle} "
          f"| cool at >{COOL_HOT:.0f}C -> {COOL_TARGET:.0f}C\n"
          f"workloads: {[w['id'] for w in todo]}", flush=True)
    try:
        for w in todo:
            run_workload(w, sampler, caps, outdir, phase=args.phase)
    finally:
        # restore the cap that was in force BEFORE we started (not the factory default -- on shared
        # boxes an admin may have set a lower fleet-wide cap that we must not overwrite)
        sudo_pl(round(start_limit))
        print(f"\n[reset] -pl {start_limit:.0f}W", flush=True)
        sampler.stop(); sampler.shutdown(); pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
