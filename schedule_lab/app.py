"""Schedule Lab — hand-craft a GPU workload orchestration, run it on ALL cards at once, write it locally.

The workload stresses BOTH on-chip subsystems at once: a COMPUTE load (fp16 GEMMs, tensor cores /
FMA units) and a MEMORY load (a STREAM-triad `z = x + y` over a large HBM buffer, pure bandwidth).
They run on two separate CUDA streams so the SMs and the memory controllers heat up concurrently.

Give a TOTAL workload (number of fp16 GEMMs) and a SCHEDULE: a list of steps that repeats until
the total is consumed. Each step runs some GEMMs at full clock (each carrying its share of memory
traffic), then optionally pauses (fixed seconds, or idle until the hottest die cools to a target
temperature). The SAME schedule runs on every visible GPU simultaneously; click Run -> it executes
across the cards, logs each card's temperature / SM clock / power at 20 Hz, and writes to local files:

    schedule_lab/result.png   (cumulative work / per-card temperature / clock / power vs time + JCT)
    schedule_lab/result.csv   (the raw 20 Hz telemetry, one temp/clk/pw column set per card)

The web page just loads result.png from disk (so the figure is a real local file you can also
open in the editor). No sudo, no clock locking — you observe the system's own thermal behaviour.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 schedule_lab/app.py   # then open http://localhost:8000
"""
from __future__ import annotations
import csv, json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # default: run on four cards at once
import torch
import pynvml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PORT = 8000
INNER = 4             # GEMMs per sync/log granule
# --- workload size DEFAULTS (all three are overridable per-run from the web UI) ---
GEMM_N = 8192         # matmul edge length — COMPUTE load  (one 8192³ fp16 GEMM ~ 1.1 TFLOP, ~12 ms on a V100)
MEM_MB = 512          # size of EACH STREAM-triad buffer x,y,z — MEMORY load, per card (3 buffers held).
                      # HBM bandwidth saturates once a buffer exceeds L2 (~6 MB); bigger only adds capacity
                      # pressure and slows each pass — it does NOT add more bandwidth/heat.
# Engine batches per granule are AUTO-BALANCED at run time (see run_schedule) so every enabled unit
# stays ~100% busy for the whole granule — no manual per-engine ratio knob.
# --- safety caps ---
MAX_TOTAL = 1000000   # cap on total fp16 GEMMs (per card) — high so a long time-based soak isn't cut short
MAX_HOLD_S = 1800.0   # cap on a single full-load hold (soak) step, seconds
MAX_GEMM_N = 24576    # cap on matmul edge length
MAX_MEM_MB = 65536    # cap on triad buffer size (further clamped to fit real VRAM at run time)
VRAM_FRAC = 0.85      # fraction of each card's VRAM the workload may claim (rest = context + headroom)
MAX_PAUSE = 120.0     # cap on a single fixed pause
COOL_MAX = 180.0      # cap on a cool-to-temp wait

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "result.png")
OUT_CSV = os.path.join(HERE, "result.csv")

# torch local index i  ->  physical GPU PHYS[i]  ->  NVML handle HANDLES[i]
PHYS = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip() != ""]
NGPU = len(PHYS)
DEVICES = [f"cuda:{i}" for i in range(NGPU)]
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]

pynvml.nvmlInit()
HANDLES = [pynvml.nvmlDeviceGetHandleByIndex(p) for p in PHYS]


def _name(h):
    n = pynvml.nvmlDeviceGetName(h)
    return n.decode() if isinstance(n, bytes) else n


BASE_NAME = _name(HANDLES[0])
NAME = f"{NGPU}× {BASE_NAME}" if NGPU > 1 else BASE_NAME
try:
    GPU_MAX = int(pynvml.nvmlDeviceGetTemperatureThreshold(HANDLES[0], getattr(pynvml, "NVML_TEMPERATURE_THRESHOLD_GPU_MAX", 3)))
except pynvml.NVMLError:
    GPU_MAX = 83

_run_lock = threading.Lock()
_T = {}            # fp16 GEMM buffers, per device (also the work/progress reference)
_M = {}            # memory STREAM-triad buffers, per device
_ENG_STREAMS = {}  # (dev, engine) -> cuda.Stream

# On-die power domains we can drive from PyTorch. Each runs on its own stream so they overlap.
# NOTE: fp16/fp32/fp64/sfu all share the SM compute domain (they time-slice, bounded by the power
# cap); 'mem' is a SEPARATE power domain (HBM) that genuinely ADDS to package power. Enabling every
# engine maximises which pipelines are exercised and pins the card at its power/thermal ceiling.
ENGINE_NAMES = ["fp16", "fp32", "fp64", "sfu", "mem"]
DEFAULT_ENGINES = list(ENGINE_NAMES)
CAL_REPS = 4          # ops timed per engine when auto-balancing granule length
MAX_ITERS = 256       # cap on per-engine ops per granule (keeps the sync barrier fine-grained)


def tensors(dev, n=GEMM_N):
    cur = _T.get(dev)
    if cur is None or cur[0].shape[0] != n:          # (re)build when the requested size changes
        _T.pop(dev, None); torch.cuda.empty_cache()  # free the old buffers first
        a = torch.randn(n, n, device=dev, dtype=torch.float16)
        _T[dev] = (a, torch.randn_like(a), torch.empty_like(a))
    return _T[dev]


def mem_tensors(dev, mb=MEM_MB):
    n = (mb * 1024 * 1024) // 2  # fp16 = 2 bytes/elem; triad moves ~3× this per pass
    cur = _M.get(dev)
    if cur is None or cur[0].numel() != n:
        _M.pop(dev, None); torch.cuda.empty_cache()
        x = torch.randn(n, device=dev, dtype=torch.float16)
        _M[dev] = (x, torch.randn_like(x), torch.empty_like(x))
    return _M[dev]


def _stream(dev, name):
    key = (dev, name)
    if key not in _ENG_STREAMS:
        with torch.cuda.device(dev):
            _ENG_STREAMS[key] = torch.cuda.Stream()
    return _ENG_STREAMS[key]


def _eng_sizes(gemm_n):
    """Per-engine GEMM edge lengths, scaled off the compute knob (fp32/fp64 are slower & wider/elem)."""
    return {"fp16": gemm_n, "fp32": max(1024, gemm_n // 2), "fp64": max(512, gemm_n // 4)}


def build_engine(dev, name, gemm_n, mem_mb):
    """Allocate buffers + return a zero-arg closure that hammers one on-die unit."""
    ez = _eng_sizes(gemm_n)
    if name == "fp16":                                                  # tensor cores
        a, b, c = tensors(dev, ez["fp16"]);              return lambda: torch.matmul(a, b, out=c)
    if name == "fp32":                                                  # fp32 CUDA cores (TF32 off)
        n = ez["fp32"]; a = torch.randn(n, n, device=dev, dtype=torch.float32)
        b = torch.randn_like(a); c = torch.empty_like(a); return lambda: torch.matmul(a, b, out=c)
    if name == "fp64":                                                  # fp64 units
        n = ez["fp64"]; a = torch.randn(n, n, device=dev, dtype=torch.float64)
        b = torch.randn_like(a); c = torch.empty_like(a); return lambda: torch.matmul(a, b, out=c)
    if name == "sfu":                                                   # special-function units
        p = torch.rand(1 << 20, device=dev, dtype=torch.float32) + 0.5  # ~4 MB, fits L2 -> SFU-bound
        q = torch.empty_like(p)
        def sfu():
            torch.sin(p, out=q); torch.exp(q, out=p); torch.rsqrt(p, out=q); torch.log(q, out=p)
        return sfu
    if name == "mem":                                                   # HBM bandwidth (separate domain)
        x, y, z = mem_tensors(dev, mem_mb);              return lambda: torch.add(x, y, out=z)
    raise ValueError(name)


def plan_sizes(gemm_n, mem_mb):
    """Clamp requested sizes to caps and to real VRAM so a run can never OOM. Returns (g, mb, note)."""
    g = max(256, min(int(gemm_n), MAX_GEMM_N))
    mb = max(16, min(int(mem_mb), MAX_MEM_MB))
    total_vram = min(torch.cuda.get_device_properties(i).total_memory for i in range(NGPU))
    # fixed cost if every compute engine is on: fp16 6g² + fp32 3g² + fp64 1.5g² + SFU/slack
    fixed = int(10.5 * g * g) + 32 * 1024 * 1024
    budget = int(total_vram * VRAM_FRAC) - fixed         # bytes left for the 3 triad buffers
    max_mb = max(16, budget // 3 // (1024 * 1024))
    note = ""
    if mb > max_mb:
        note = f"内存 buffer 从 {mb} 夹紧到 {max_mb} MB 以适配显存"
        mb = max_mb
    return g, mb, note


def temp(h):
    return pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)


def hottest():
    return max(temp(h) for h in HANDLES)


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = False
        self.rows = []          # each: (t, [temp_i], [clk_i], [pw_i], phase, work)
        self.phase = "run"
        self.work = 0
        self.t0 = time.perf_counter()

    def run(self):
        while not self.stop_flag:
            t = time.perf_counter() - self.t0
            temps, clks, pws = [], [], []
            for h in HANDLES:
                temps.append(temp(h))
                clks.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
                try:
                    pws.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
                except pynvml.NVMLError:
                    pws.append(float("nan"))
            self.rows.append((t, temps, clks, pws, self.phase, self.work))
            time.sleep(0.05)


def _sync_all():
    for d in DEVICES:
        torch.cuda.synchronize(d)


def _calibrate(ops0):
    """Time one op of each engine (device 0) and pick a per-engine BATCH size so each thread does
    ~INNER fp16 GEMMs of work between stream syncs — small enough to react to stop, big enough that
    sync overhead is negligible. Batch sizing does NOT couple engines (each runs on its own thread)."""
    d0 = DEVICES[0]; per_op = {}
    for e, op in ops0.items():
        for _ in range(2):
            op()                                    # warm (first launch compiles/plans)
        torch.cuda.synchronize(d0)
        t0 = time.perf_counter()
        for _ in range(CAL_REPS):
            op()
        torch.cuda.synchronize(d0)
        per_op[e] = max((time.perf_counter() - t0) / CAL_REPS, 1e-6)
    batch_s = per_op["fp16"] * INNER
    return {e: (INNER if e == "fp16" else max(1, min(MAX_ITERS, round(batch_s / per_op[e])))) for e in ops0}


def run_schedule(total, steps, gemm_n=GEMM_N, mem_mb=MEM_MB, engines=DEFAULT_ENGINES):
    engines = [e for e in ENGINE_NAMES if e in engines] or ["fp16"]
    if "fp16" not in engines:
        engines = ["fp16"] + engines                # fp16 GEMM is the work/progress reference
    torch.backends.cuda.matmul.allow_tf32 = False   # route fp32 GEMM to real FP32 cores (not TF32 tensor cores)
    torch.backends.cudnn.allow_tf32 = False

    ops = {d: {e: build_engine(d, e, gemm_n, mem_mb) for e in engines} for d in DEVICES}
    iters = _calibrate(ops[DEVICES[0]])
    for d in DEVICES:                               # warm every card's buffers
        for e in engines:
            ops[d][e]()
    _sync_all()

    s = Sampler(); s.start(); time.sleep(0.2)
    s.t0 = time.perf_counter()
    done_lock = threading.Lock()
    done = {"n": 0}

    # One worker per (card, engine). Each keeps ITS OWN stream continuously full and syncs only itself,
    # so the high-power tensor engine never stalls waiting on a slow low-power one (e.g. SFU). This is
    # what actually pins every card at its power/thermal cap.
    def spin(d, e, stop):
        op, st, it, is_ref = ops[d][e], _stream(d, e), iters[e], (e == "fp16")
        while not stop.is_set():
            with torch.cuda.device(d):
                with torch.cuda.stream(st):
                    for _ in range(it):
                        op()
            st.synchronize()                        # bound queue depth; releases the GIL while waiting
            if is_ref:
                with done_lock:
                    done["n"] += it

    def run_span(keep_going):
        """Run all engines flat-out on all cards until keep_going() is False or total is reached."""
        stop = threading.Event()
        ths = [threading.Thread(target=spin, args=(d, e, stop), daemon=True)
               for d in DEVICES for e in engines]
        for t in ths:
            t.start()
        while keep_going() and done["n"] < total:
            s.work = done["n"]
            time.sleep(0.05)
        stop.set()
        for t in ths:
            t.join()
        s.work = done["n"]

    pause_total = 0.0
    # A schedule with any `gemms` step is COUNT-driven: repeat the list until `total` GEMMs are done.
    # A purely time/pause-driven schedule (only hold_s / pause_s / cool_to) runs the list ONCE then stops
    # — otherwise a single {"hold_s": N} soak would repeat until it hit the `total` safety cap (~forever).
    had_count = any(int(st.get("gemms", 0) or 0) > 0 for st in steps)
    while done["n"] < total:
        for step in steps:
            if done["n"] >= total:
                break
            hold = float(step.get("hold_s", 0) or 0)
            k = int(step.get("gemms", 0) or 0)
            if hold > 0:                            # time-based soak: full load for hold_s seconds
                s.phase = "run"; t_end = time.perf_counter() + min(hold, MAX_HOLD_S)
                run_span(lambda: time.perf_counter() < t_end)
            elif k > 0:                             # count-based: full load until this step's GEMMs done
                s.phase = "run"; tgt = min(done["n"] + k, total)
                run_span(lambda tgt=tgt: done["n"] < tgt)
            if done["n"] >= total:
                break
            if step.get("cool_to") is not None:
                s.phase = "cool"; ct = float(step["cool_to"]); tc = time.perf_counter()
                while hottest() > ct and time.perf_counter() - tc < COOL_MAX:
                    time.sleep(0.2)
                pause_total += time.perf_counter() - tc
            elif float(step.get("pause_s", 0) or 0) > 0:
                s.phase = "cool"; p = min(float(step["pause_s"]), MAX_PAUSE)
                time.sleep(p); pause_total += p
        if not had_count:                           # time/pause-only schedule: one pass is the whole run
            break
    jct = time.perf_counter() - s.t0
    s.stop_flag = True; s.join()
    ops.clear(); torch.cuda.empty_cache()           # free per-run fp32/fp64/sfu buffers
    return s.rows, jct, pause_total, done["n"], iters


def write_outputs(rows, jct, pause_total, done, total, steps):
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        head = ["t_s", "phase", "work_done"]
        for p in PHYS:
            head += [f"temp_c_g{p}", f"sm_clk_mhz_g{p}", f"power_w_g{p}"]
        w.writerow(head)
        for r in rows:
            t, temps, clks, pws, phase, work = r
            out = [round(t, 3), phase, work]
            for gi in range(NGPU):
                pw = pws[gi]
                out += [temps[gi], clks[gi], round(pw, 1) if pw == pw else ""]
            w.writerow(out)

    t = np.array([r[0] for r in rows])
    temp_a = np.array([r[1] for r in rows])      # (T, NGPU)
    clk_a = np.array([r[2] for r in rows])
    pw_a = np.array([r[3] for r in rows])
    work = np.array([r[5] for r in rows])
    cool = np.array([r[4] == "cool" for r in rows])

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 1.2]})

    def shade(a):
        d = np.diff(np.concatenate([[0], cool.astype(int), [0]]))
        for si, ei in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            a.axvspan(t[min(si, len(t)-1)], t[min(ei-1, len(t)-1)], color="#1f77b4", alpha=.10)

    a = ax[0]; shade(a)
    a.plot(t, work, "#1f77b4", lw=2.2); a.plot(t[-1], work[-1], "o", color="#1f77b4", ms=8, mec="k")
    if total <= max(done, 1) * 2:                    # only show the target line for a real count target
        a.axhline(total, color="gray", ls="--", alpha=.6)
        goal = f"{done}/{total}"
    else:                                            # time-driven soak: `total` is just the safety cap
        goal = f"{done}"
    a.set_ylabel("cumulative GEMMs / card")
    a.set_title(f"JCT = {jct:.1f}s   |   {goal} GEMMs × {NGPU} cards   |   idle/pause {pause_total:.1f}s "
                f"({100*pause_total/max(jct,1e-9):.0f}%)   |   blue = pause")
    a.grid(alpha=.3)

    a = ax[1]; shade(a)
    for gi in range(NGPU):
        a.plot(t, temp_a[:, gi], COLORS[gi % len(COLORS)], lw=1.6, label=f"GPU{PHYS[gi]}")
    a.axhline(GPU_MAX, color="k", ls="--", alpha=.6, label=f"throttle {GPU_MAX}°C")
    a.set_ylabel("temp (°C)"); a.legend(fontsize=8, ncol=NGPU + 1, loc="lower right"); a.grid(alpha=.3)

    a = ax[2]; shade(a)
    for gi in range(NGPU):
        a.plot(t, clk_a[:, gi], COLORS[gi % len(COLORS)], lw=1.4, alpha=.9)
    a.set_ylabel("clock (MHz)"); a.grid(alpha=.3)
    a2 = a.twinx()
    tot_pw = np.nansum(pw_a, axis=1)
    a2.plot(t, tot_pw, "#7f7f7f", lw=1.2, alpha=.7)
    a2.set_ylabel("total power (W)", color="#7f7f7f")
    a.set_xlabel("time (s)")

    fig.suptitle(f"{NAME} — schedule: {json.dumps(steps)}", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight"); plt.close(fig)


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Schedule Lab</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 .wrap{max-width:1100px;margin:0 auto;padding:18px}
 h1{font-size:18px;margin:0 0 2px} .sub{color:#9aa;font-size:13px;margin-bottom:14px}
 .row{display:flex;gap:18px;flex-wrap:wrap}
 .card{background:#181b22;border:1px solid #262b36;border-radius:10px;padding:14px;flex:1;min-width:320px}
 label{display:block;font-size:12px;color:#9aa;margin:8px 0 3px}
 input,textarea{width:100%;box-sizing:border-box;background:#0f1115;border:1px solid #2a2f3a;color:#e6e6e6;border-radius:7px;padding:8px;font-family:ui-monospace,monospace;font-size:13px}
 textarea{height:150px;resize:vertical}
 button{background:#2d6cdf;color:#fff;border:0;border-radius:7px;padding:9px 16px;font-size:14px;cursor:pointer;margin-top:10px}
 button.ghost{background:#262b36;font-size:12px;padding:6px 10px;margin:3px 5px 0 0}
 button:disabled{opacity:.5;cursor:wait}
 .err{color:#ff8080;font-size:13px;white-space:pre-wrap}
 img{width:100%;border-radius:8px;margin-top:10px;background:#fff}
 code{background:#0f1115;padding:1px 5px;border-radius:4px;color:#9cf}
 .hint{font-size:12px;color:#889;line-height:1.5}
 #engs{font-size:12px;color:#cdd;margin-bottom:4px}
 #engs span{display:inline-block;margin:2px 14px 2px 0;white-space:nowrap}
 input[type=checkbox]{width:auto;margin:0 5px 0 0;vertical-align:middle}
</style></head><body><div class=wrap>
<h1>Schedule Lab — 手动编排 GPU 工作负载（多卡同时）</h1>
<div class=sub>__NAME__ · 降频阈值 __GPUMAX__°C · 全芯片压力:<b>张量核 / FP32 / FP64 / SFU / 显存带宽</b> 各占一条 stream 并发拉满,自动配平 · 同一编排在 __NGPU__ 张卡上同时执行 · 结果写入 <code>schedule_lab/result.png</code> + <code>result.csv</code></div>
<div class=row>
 <div class=card>
  <label>总工作量 total（每张卡 fp16 GEMM 上限，≤ __MAXTOTAL__；时长 soak 时当安全上限）</label>
  <input id=total value=1000000>
  <div style="display:flex;gap:10px">
   <div style="flex:1"><label>计算 GEMM 边长（≤ __MAXGEMMN__）</label><input id=gemm_n value=8192></div>
   <div style="flex:1"><label>显存 buffer MB／块（自动夹到显存）</label><input id=mem_mb value=512></div>
  </div>
  <label>发热引擎（勾选=并发拉满对应片上单元）</label>
  <div id=engs>
   <span><input type=checkbox class=eng value=fp16 checked>张量核 FP16</span>
   <span><input type=checkbox class=eng value=fp32 checked>FP32 核</span>
   <span><input type=checkbox class=eng value=fp64 checked>FP64 单元</span>
   <span><input type=checkbox class=eng value=sfu checked>SFU 特殊函数</span>
   <span><input type=checkbox class=eng value=mem checked>显存带宽</span>
  </div>
  <label>编排 schedule（JSON 步骤列表，循环执行直到做满 total）</label>
  <textarea id=steps>[
  {"hold_s": 300}
]</textarea>
  <div class=hint>每个步骤三选一驱动：<code>hold_s</code>=满载持续 N 秒(soak,推荐用于顶满)，或 <code>gemms</code>=这步每卡跑多少个 GEMM；然后可选停顿 <code>pause_s</code>=固定停 N 秒 / <code>cool_to</code>=空闲到最热的卡温度≤该值。步骤列表循环。<b>__NGPU__ 张卡跑同一份编排、同步推进；各引擎自动配平,granule 内全部 pipe 持续繁忙。</b></div>
  <div>
   <button class=ghost onclick="preset('max')">🔥 顶满</button>
   <button class=ghost onclick="preset('burst')">一次性灌满</button>
   <button class=ghost onclick="preset('trickle')">涓流(小块+小停)</button>
   <button class=ghost onclick="preset('coolgate')">冷却闸(块+冷到72°C)</button>
   <button class=ghost onclick="preset('ramp')">自定义阶梯</button>
  </div>
  <button id=run onclick="run()">▶ 运行</button>
  <div id=msg class=err></div>
 </div>
 <div class=card>
  <div id=result><div class=hint>配置好左侧,点"运行"。结果图会写到 <code>schedule_lab/result.png</code> 并显示在这里(也可直接在编辑器里打开该文件)。</div></div>
 </div>
</div>
<script>
const ALLENG=['fp16','fp32','fp64','sfu','mem'];
const PRE = {
 max:    {total:1000000, gemm_n:8192, mem_mb:512, engines:ALLENG, steps:[{hold_s:300}]},
 burst:  {total:2000, engines:ALLENG, steps:[{gemms:2000,pause_s:0}]},
 trickle:{total:2000, engines:ALLENG, steps:[{gemms:50,pause_s:1.0}]},
 coolgate:{total:2000, engines:ALLENG, steps:[{gemms:400,cool_to:72}]},
 ramp:   {total:2000, engines:ALLENG, steps:[{gemms:800,pause_s:0},{gemms:200,pause_s:4},{gemms:100,pause_s:8}]}
};
function preset(k){const p=PRE[k];document.getElementById('total').value=p.total;
 if(p.gemm_n)document.getElementById('gemm_n').value=p.gemm_n;
 if(p.mem_mb)document.getElementById('mem_mb').value=p.mem_mb;
 if(p.engines)document.querySelectorAll('.eng').forEach(x=>x.checked=p.engines.includes(x.value));
 document.getElementById('steps').value=JSON.stringify(p.steps,null,2);}
async function run(){
 const msg=document.getElementById('msg'); msg.textContent='';
 let steps; const total=parseInt(document.getElementById('total').value);
 const gemm_n=parseInt(document.getElementById('gemm_n').value);
 const mem_mb=parseInt(document.getElementById('mem_mb').value);
 const engines=[...document.querySelectorAll('.eng:checked')].map(x=>x.value);
 if(!engines.length){msg.textContent='至少勾选一个发热引擎';return;}
 try{steps=JSON.parse(document.getElementById('steps').value);}catch(e){msg.textContent='schedule JSON 解析失败: '+e;return;}
 const btn=document.getElementById('run'); btn.disabled=true; btn.textContent='⏳ 运行中...';
 document.getElementById('result').innerHTML='<div class=hint>运行中,请稍候…(图会在完成后写入 result.png 并显示)</div>';
 try{
  const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({total,steps,gemm_n,mem_mb,engines})});
  const j=await r.json();
  if(j.error){msg.textContent=j.error;document.getElementById('result').innerHTML='';}
  else{const eng=Object.keys(j.iters||{}).map(e=>e+'×'+j.iters[e]).join(' ');
   document.getElementById('result').innerHTML='<b>JCT = '+j.jct.toFixed(1)+'s</b> · 停顿 '+j.pause.toFixed(1)+'s · 峰值 '+j.peak+'°C · 最低频 '+j.minclk+'MHz · 计算 '+j.gemm_n+'² · 引擎/granule '+eng+(j.note?' · <span style=color:#ffb86c>'+j.note+'</span>':'')+' · 已写入 result.png/csv<img src="/result.png?t='+j.ts+'">';}
 }catch(e){msg.textContent='运行失败: '+e;}
 btn.disabled=false; btn.textContent='▶ 运行';
}
</script></div></body></html>"""

PAGE = (PAGE.replace("__NAME__", NAME).replace("__GPUMAX__", str(GPU_MAX))
        .replace("__MAXTOTAL__", str(MAX_TOTAL)).replace("__NGPU__", str(NGPU))
        .replace("__MAXGEMMN__", str(MAX_GEMM_N)))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?", 1)[0]
        if p in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif p == "/result.png" and os.path.exists(OUT_PNG):
            with open(OUT_PNG, "rb") as f:
                self._send(200, f.read(), "image/png", {"Cache-Control": "no-store"})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/run":
            self._send(404, b"{}"); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            total = int(req.get("total", 0)); steps = req.get("steps", [])
            gemm_n = int(req.get("gemm_n", GEMM_N) or GEMM_N)
            mem_mb = int(req.get("mem_mb", MEM_MB) or MEM_MB)
            engines = req.get("engines", DEFAULT_ENGINES)
            engines = [e for e in ENGINE_NAMES if e in engines]
            if total <= 0 or total > MAX_TOTAL:
                raise ValueError(f"total 必须在 1..{MAX_TOTAL}")
            if not isinstance(steps, list) or not steps:
                raise ValueError("steps 必须是非空列表")
            if not any((int(s.get("gemms", 0) or 0) > 0 or float(s.get("hold_s", 0) or 0) > 0) for s in steps):
                raise ValueError("至少有一个步骤要有 gemms>0 或 hold_s>0")
            if not engines:
                raise ValueError("至少启用一个发热引擎")
        except Exception as e:
            self._send(200, json.dumps({"error": f"输入错误: {e}"}).encode()); return

        if not _run_lock.acquire(blocking=False):
            self._send(200, json.dumps({"error": "已有一个运行在进行中,请等它结束"}).encode()); return
        try:
            gemm_n, mem_mb, note = plan_sizes(gemm_n, mem_mb)
            rows, jct, pause_total, done, iters = run_schedule(total, steps, gemm_n, mem_mb, engines)
            write_outputs(rows, jct, pause_total, done, total, steps)
            peak = max(max(r[1]) for r in rows)
            run_clks = [min(r[2]) for r in rows if r[4] == "run"]
            minclk = min(run_clks) if run_clks else 0
            self._send(200, json.dumps({"ok": True, "jct": jct, "pause": pause_total,
                                        "peak": peak, "minclk": minclk, "ts": int(time.time()),
                                        "gemm_n": gemm_n, "mem_mb": mem_mb, "iters": iters,
                                        "note": note}).encode())
        except Exception as e:
            self._send(200, json.dumps({"error": f"运行出错: {e}"}).encode())
        finally:
            _run_lock.release()


if __name__ == "__main__":
    print(f"Schedule Lab on http://localhost:{PORT}  ({NAME}, throttle {GPU_MAX}°C)")
    print(f"cards: physical {PHYS}  ->  torch {DEVICES}")
    print(f"results -> {OUT_PNG}  +  {OUT_CSV}")
    print(f"engines: {'/'.join(ENGINE_NAMES)}  (each on its own stream, auto-balanced per granule)")
    print("warming up fp16 + memory buffers on every card...")
    for d in DEVICES:
        tensors(d); mem_tensors(d)
    _sync_all()
    print("ready. open the URL (VS Code auto-forwards the port).")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
