"""Run the portfolio sweep -> per-workload CSVs. Methodology v4:

  PREFILL = LOCKED-CLOCK sweep (nvidia-smi -lgc), cap pinned at the device max (-pl pl_max).
    Rationale: under a cap sweep heavy prefill only reaches ~50-65% of f_max before hitting the
    power wall, so P(f) stays in the near-linear voltage-floor region and tok/J rises monotonically
    -- the efficiency sweet spot is invisible. Locking the clock walks f itself toward f_max: each
    point records the SET clock and the EFFECTIVE clock; when the card can no longer hold the lock
    (power wall), the point is flagged power_limited and the ascent stops after the effective clock
    saturates. If the sweet spot exists inside the card's envelope, this finds it; if the sweep
    saturates first, that PROVES the peak lies outside the envelope (max cap is then optimal).
  DECODE  = POWER-CAP sweep (nvidia-smi -pl), clocks unlocked -- unchanged from v3. The cap range
    is the device's own [pl_min, pl_max] constraint (H200: [200, 700] W; 700 is a hard NVML
    ceiling, -pl above it is rejected).

Measurement per point is methodology v3 (see DATA_QUALITY.zh.md) plus the H200 manual's telemetry
rules (energy-accumulator window power, throttle-reason gating, meta.json).

GPU-agnostic: the cap grid derives from the device's power-limit constraints (--caps auto), the
clock grid from the device's supported graphics clocks (--clocks auto), cooling thresholds from
the device's slowdown threshold; f_max is written to <outdir>/meta.json for the fitting scripts.

Writes  <outdir>/<id>_prefill.csv  (one row per locked clock)  and  <outdir>/<id>_decode.csv
(one row per cap) + meta.json.  NOTE: v4 prefill CSVs are NOT comparable to v3 cap-sweep prefill
CSVs -- use a fresh outdir (run_all.sh defaults to data_<gpu>_v4).

  # one GPU, whole portfolio (resume-aware):
  SUDO_PASS=... CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../code python3 run_portfolio.py --outdir data_h200_v4

  # slice the portfolio across GPUs (each writes to the same outdir, distinct files):
  SUDO_PASS=... CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../code python3 run_portfolio.py \
      --outdir data_h200_v4 --ids rag-phi3,code-phi3

Privilege for `nvidia-smi -pl/-lgc/-rgc`: run as root, OR passwordless sudo, OR SUDO_PASS=<password>.
Telemetry (pynvml) and the -pl/-lgc targets both follow CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations
import argparse, csv, json, os, signal, subprocess, sys, time

# MUST precede the torch import. The HF DynamicCache torch.cat's the whole KV every decode step;
# with the default block allocator that realloc pattern is a lottery (rag-shaped workloads showed
# bimodal 225-267 tok/s, +-9%). expandable_segments grows one VA segment smoothly: 272-281 tok/s,
# +-1.6% AND ~15% faster -- the artifact-free measurement. Verified on V100, 8 repeats each.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# measurement hosts are often air-gapped: never let transformers/hub touch the network
# (models must already be in the local cache -- preflight.py verifies)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
import pynvml

import config as C                                   # noqa: E402  (from ../../code via PYTHONPATH)
C.WARMUP_S = 0.8; C.SETTLE_S = 0.2; C.MEASURE_S = 2.5   # short bursts; warmup also estimates the step
                                                        # rate (a 700 W part settles slower than a 250 W one)
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


def _sudo_smi(args_):
    """Run `nvidia-smi -i GPU <args>`: direct when root, else passwordless sudo, else SUDO_PASS."""
    cmd = ["nvidia-smi", "-i", GPU] + [str(a) for a in args_]
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return subprocess.run(cmd, text=True, capture_output=True)
    if not PW:
        return subprocess.run(["sudo", "-n"] + cmd, text=True, capture_output=True)
    return subprocess.run(["sudo", "-S", "-p", ""] + cmd,
                          input=PW + "\n", text=True, capture_output=True)


def sudo_pl(w):
    return _sudo_smi(["-pl", int(w)])


def sudo_lgc(f):
    return _sudo_smi(["-lgc", f"{int(f)},{int(f)}"])


def sudo_rgc():
    return _sudo_smi(["-rgc"])


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
    """Load an arbitrary model (portfolio overrides the config default). Delegates to
    measure.load_model, which carries the dtype-kwarg compatibility shim (transformers
    renamed torch_dtype->dtype; some versions silently ignore the unknown spelling)."""
    C.MODEL_ID = model_id
    from measure import load_model
    return load_model()


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
    if len(runs) == 2:   # median of 2 = the run closer to the mean (picking [1] would bias fast)
        m = sum(ts) / 2
        chosen = min(runs, key=lambda r: abs(r["throughput_tok_s"] - m))
    else:
        chosen = sorted(runs, key=lambda r: r["throughput_tok_s"])[len(runs) // 2]
    return chosen, spread, len(runs)


def run_workload(w, sampler, caps, clks, pre_cap_w, outdir, phase="both"):
    """One workload, two sweeps: prefill walks a LOCKED-CLOCK grid at the max cap (fixed S,B);
    decode walks the CAP grid with clocks unlocked (fixed C,B). Each is single-valued in its knob."""
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
        # ---- prefill: LOCKED-CLOCK sweep, cap pinned at pre_cap_w (compute-bound, fixed S,B) ----
        if phase in ("prefill", "both"):
            rpl = sudo_pl(pre_cap_w)              # free the power axis; the clock is the knob
            if rpl.returncode != 0:
                print(f"  -pl {pre_cap_w} FAILED: {(rpl.stderr or rpl.stdout).strip()[:90]}")
            # record the cap that is ACTUALLY in force (truthful provenance even if the pin failed)
            try:
                hh = pynvml.nvmlDeviceGetHandleByIndex(int(GPU))
                cap_now = int(round(pynvml.nvmlDeviceGetEnforcedPowerLimit(hh) / 1000.0))
            except pynvml.NVMLError:
                cap_now = pre_cap_w
            if cap_now != pre_cap_w:
                print(f"  [warn] enforced cap is {cap_now}W (wanted {pre_cap_w}W) -- rows record {cap_now}")
            last_limited_eff = None
            try:
                for f in clks:
                    sudo_rgc()                    # cool/idle at unlocked clocks between points
                    free(); cool(sampler)
                    rlk = sudo_lgc(f)
                    if rlk.returncode != 0:
                        print(f"  -lgc {f} FAILED: {(rlk.stderr or rlk.stdout).strip()[:90]}"); continue
                    time.sleep(0.4)
                    print(f"  === CLK {f} MHz (cap {pre_cap_w} W) ===", flush=True)
                    try:
                        r = run_prefill_point(model, sampler, pb, ps, vocab)
                        if not _thermally_clean(r):    # manual 7.4: thermal bits -> cool + retry once
                            print("    prefill point thermally contaminated -> cooling + re-measuring", flush=True)
                            free(); cool(sampler)
                            r = run_prefill_point(model, sampler, pb, ps, vocab)
                        eff = r.get("sm_clk_avg", 0.0)
                        # power wall: the card cannot hold the locked clock inside the cap.
                        # Heuristic (DVFS dithers right AT the wall) -- sm_clk_avg is ground truth;
                        # a mislabel costs at most one redundant extra point before the stop fires.
                        limited = bool(eff and eff < 0.97 * f)
                        pre_rows.append(_row("prefill", cap_now, r, seq_len=ps, batch=pb,
                                             clk_set_mhz=f, power_limited=limited))
                        print(f"    prefill b={pb} S={ps:<5} | {r['throughput_tok_s']:>8.0f} tok/s | "
                              f"{r.get('power_avg_w',0):>5.0f}W | sm {eff:.0f}/{f} "
                              f"| {r.get('tok_per_joule',0):.1f} tok/J"
                              + ("  [power-limited]" if limited else ""), flush=True)
                        # two consecutive power-limited points with the same effective clock ->
                        # higher locks cannot change the operating point; stop the ascent
                        if limited and last_limited_eff and abs(eff - last_limited_eff) / last_limited_eff < 0.015:
                            print(f"    [stop] effective clock saturated ~{eff:.0f} MHz at the "
                                  f"{pre_cap_w} W wall -- the efficiency peak (if any) lies outside "
                                  f"this card's power envelope", flush=True)
                            break
                        last_limited_eff = eff if limited else None
                    except torch.cuda.OutOfMemoryError:
                        print(f"    prefill OOM at b={pb} S={ps}"); free()
            finally:
                rr = sudo_rgc()                   # NEVER leave the box with locked clocks
                if rr.returncode != 0:
                    print(f"  [warn] -rgc FAILED: {(rr.stderr or rr.stdout).strip()[:90]}", flush=True)
            # persist prefill rows NOW -- a crash mid-decode must not discard a finished sweep
            if pre_rows:
                _write(os.path.join(outdir, f"{w['id']}_prefill.csv"), pre_rows)
        # ---- decode: CAP sweep, clocks unlocked (memory-bound, fixed C,B; v3 windows+repeats) ----
        if phase in ("decode", "both"):
            for cap in caps:
                rpl = sudo_pl(cap)
                if rpl.returncode != 0:
                    print(f"  -pl {cap} FAILED: {(rpl.stderr or rpl.stdout).strip()[:90]}"); continue
                time.sleep(0.4)
                print(f"  === CAP {cap} W ===", flush=True)
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
    print(f"  wrote {path} ({len(rows)} points)", flush=True)


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
        "methodology": "v4: prefill locked-clock sweep at max cap (-lgc, power_limited flag, "
                       "saturation stop) + decode cap sweep; per-point measurement v3 "
                       "(seed_chunk=ctx, expandable_segments, step-targeted windows, "
                       "repeats+median, ctx_eff, energy-window power, throttle gating)",
    }
    try:
        mems = sorted(pynvml.nvmlDeviceGetSupportedMemoryClocks(h))
        meta["supported_mem_clocks_mhz"] = mems
        sms = sorted(set(pynvml.nvmlDeviceGetSupportedGraphicsClocks(h, mems[-1])))
        meta["sm_clk_min_mhz"], meta["f_max_mhz"] = sms[0], sms[-1]   # f_max feeds the fits (x=f/f_max)
        meta["supported_sm_clocks_mhz"] = sms      # the -lgc grid snaps to these (prefill sweep)
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
    ap.add_argument("--outdir", default="data_v4",
                    help="output dir (relative to this file), e.g. data_h200_v4. Do NOT point at a "
                         "v3 cap-sweep dir: v4 prefill CSVs are clock-swept (not comparable) and "
                         "resume would skip workloads whose v3 CSVs already exist")
    ap.add_argument("--caps", default="auto",
                    help="decode sweep: 'auto' = N points spanning the device's own [min,max] "
                         "constraint, or an explicit comma list like '200,300,400,500,600,700'")
    ap.add_argument("--grid-n", type=int, default=10, help="number of points for --caps auto")
    ap.add_argument("--clocks", default="auto",
                    help="prefill sweep: 'auto' = N locked-clock points spanning the device's own "
                         "supported SM clocks up to f_max, or a comma list like '345,700,1100,1500,1980' "
                         "(each snapped to the nearest supported clock)")
    ap.add_argument("--clock-grid-n", type=int, default=12, help="number of points for --clocks auto")
    args = ap.parse_args()
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(HERE, args.outdir)
    # methodology guard: never mix v4 clock-swept prefill rows into a pre-v4 (cap-swept) data dir --
    # resume would skip silently and --force would clobber the reference data with incomparable rows
    mpath = os.path.join(outdir, "meta.json")
    if os.path.exists(mpath):
        try:
            with open(mpath) as f:
                _m = json.load(f).get("methodology", "")
        except (OSError, ValueError):
            _m = ""
        if not _m.startswith("v4"):
            print(f"ERROR: {outdir} holds pre-v4 data (methodology '{_m[:40] or 'unknown'}'); "
                  f"v4 prefill CSVs are clock-swept and NOT comparable -- pass a fresh --outdir "
                  f"(e.g. data_h200_v4)")
            raise SystemExit(1)
    os.makedirs(outdir, exist_ok=True)

    # privilege checks up-front (root / sudo -n / SUDO_PASS)
    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(int(GPU))
    start_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
    r0 = sudo_pl(round(start_limit))
    if r0.returncode != 0:
        print("ERROR: cannot run `nvidia-smi -pl` (need root, passwordless sudo, or SUDO_PASS)")
        print((r0.stderr or r0.stdout).strip()[:200])
        raise SystemExit(1)
    # ALWAYS clear stale clock locks up-front (a previous run killed mid-prefill would otherwise
    # silently poison a decode-only run); benign no-op when nothing is locked
    r1 = sudo_rgc()
    if r1.returncode != 0:
        if args.phase in ("prefill", "both"):
            print("ERROR: cannot run `nvidia-smi -rgc` -- the prefill locked-clock sweep needs "
                  "clock-lock privilege (same as -pl). Run --phase decode, or fix the privilege.")
            print((r1.stderr or r1.stdout).strip()[:200])
            raise SystemExit(1)
        print("[warn] -rgc failed; continuing decode-only, but verify no stale clock lock is active")
    # convert SIGTERM/SIGHUP into SystemExit so the `finally` restore (-rgc, -pl) runs even when an
    # unattended run is killed. Respect an inherited SIG_IGN (a nohup'd run must NOT start dying on
    # ssh drop), and reset to SIG_DFL on first delivery so a second signal cannot interrupt the
    # restore in progress. SIGKILL is uncoverable here; run_all.sh's trap is the backstop.
    def _term(signum, _frame):
        signal.signal(signum, signal.SIG_DFL)
        raise SystemExit(128 + signum)
    for _sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            if signal.getsignal(_sig) != signal.SIG_IGN:
                signal.signal(_sig, _term)
        except (ValueError, OSError):
            pass

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
        # exponent 1.35 packs points toward the LOW end, where all the decode structure lives
        # (on a 700 W part the plateau starts ~400 W; a uniform grid would waste half its points there)
        caps = sorted({int(round(mn + (mx - mn) * (i / (n - 1)) ** 1.35)) for i in range(n)})
    else:
        caps = [c for c in (int(x) for x in args.caps.split(",")) if mn <= c <= mx]
    if not caps and args.phase in ("decode", "both"):
        print(f"ERROR: no caps to sweep (requested '{args.caps}', device range [{mn:.0f},{mx:.0f}] W)")
        raise SystemExit(1)
    if len(caps) < 3 and args.phase in ("decode", "both"):
        print(f"[warn] only {len(caps)} cap point(s) -- decode P(T) curve will be degenerate")
    meta["cap_grid_w"] = caps

    # prefill locked-clock grid: snap to the device's own supported SM clocks
    sms = meta.get("supported_sm_clocks_mhz") or []
    clks = []
    if args.phase in ("prefill", "both"):
        if args.clocks == "auto":
            if not sms:
                print("ERROR: cannot list supported graphics clocks (needed for --clocks auto); "
                      "pass an explicit --clocks list instead")
                raise SystemExit(1)
            n = max(args.clock_grid_n, 4)
            lo, hi = sms[0], sms[-1]
            # exponent 0.8 packs points toward the HIGH end: the efficiency knee (superlinear
            # power region) lives near f_max, the voltage-floor region below ~0.5*f_max is linear
            targets = [lo + (hi - lo) * (i / (n - 1)) ** 0.8 for i in range(n)]
        else:
            targets = [float(x) for x in args.clocks.split(",")]
        if sms:
            clks = sorted({min(sms, key=lambda s: abs(s - t)) for t in targets})
        else:
            clks = sorted({int(round(t)) for t in targets})
        if len(clks) < 4:
            print(f"[warn] only {len(clks)} clock point(s) -- prefill e(f) curve will be degenerate")
        pre_cap_w = int(round(mx))               # prefill sweeps the clock; the cap sits at the wall
    else:
        pre_cap_w = int(round(mx))
    meta["prefill_clk_grid_mhz"] = clks
    meta["prefill_cap_w"] = pre_cap_w
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    # cooling thresholds relative to the device's own slowdown threshold (fallback: V100 constants)
    slow = meta.get("temp_slowdown_c")
    if slow and slow >= 60:                      # Hopper may report margin-to-threshold semantics;
        COOL_HOT = min(max(slow - 20, 60), 78)   # a tiny/absurd value would push COOL_HOT negative
        COOL_TARGET = COOL_HOT - 12              # and stall 90 s per point -> clamp to a sane band

    sampler = PowerSampler(interval_s=C.SAMPLE_INTERVAL_S); sampler.start(); time.sleep(0.3)
    print(f"GPU{GPU} {sampler.name} | cap range [{mn:.0f},{mx:.0f}] restore-to {start_limit:.0f}W\n"
          f"decode caps {caps}\n"
          f"prefill locked clocks {clks} (cap pinned at {pre_cap_w}W)\n"
          f"energy-window power: {sampler.has_energy} | throttle telemetry: {sampler.has_throttle} "
          f"| cool at >{COOL_HOT:.0f}C -> {COOL_TARGET:.0f}C\n"
          f"workloads: {[w['id'] for w in todo]}", flush=True)
    try:
        for w in todo:
            run_workload(w, sampler, caps, clks, pre_cap_w, outdir, phase=args.phase)
            # keep the sampler ring small: a multi-hour sweep would otherwise accumulate
            # ~500k samples and make every stats_between()/cool() scan crawl
            sampler.samples = sampler.samples[-20000:]
    finally:
        # unlock clocks FIRST, then restore the cap that was in force BEFORE we started (not the
        # factory default -- on shared boxes an admin may have set a lower fleet-wide cap)
        rr = sudo_rgc()
        rp = sudo_pl(round(start_limit))
        if rr.returncode != 0 or rp.returncode != 0:
            print(f"\n[reset] WARNING: restore failed (rgc rc={rr.returncode}, pl rc={rp.returncode}) "
                  f"-- run `sudo nvidia-smi -i {GPU} -rgc; sudo nvidia-smi -i {GPU} -pl "
                  f"{start_limit:.0f}` by hand", flush=True)
        else:
            print(f"\n[reset] -rgc; -pl {start_limit:.0f}W", flush=True)
        sampler.stop(); sampler.shutdown(); pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
