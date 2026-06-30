"""Schedule Lab — hand-craft a GPU workload orchestration, run it, and write the result locally.

Give a TOTAL workload (number of fp16 GEMMs) and a SCHEDULE: a list of steps that repeats until
the total is consumed. Each step runs some GEMMs at full clock, then optionally pauses (fixed
seconds, or idle until the die cools to a target temperature). Click Run -> it executes on the
GPU, logs temperature / SM clock / power at 20 Hz, and writes the result straight to local files:

    schedule_lab/result.png   (cumulative work / temperature / clock vs time + JCT)
    schedule_lab/result.csv   (the raw 20 Hz telemetry)

The web page just loads result.png from disk (so the figure is a real local file you can also
open in the editor). No sudo, no clock locking — you observe the system's own thermal behaviour.

  CUDA_VISIBLE_DEVICES=1 python3 schedule_lab/app.py      # then open http://localhost:8000
"""
from __future__ import annotations
import csv, json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() in ("", "0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # always GPU1, never device 0
import torch
import pynvml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PORT = 8000
GEMM_N = 8192          # matmul size (one GEMM ~ 1.1 TFLOP; ~12 ms at full clock on a V100)
INNER = 4             # GEMMs per sync/log granule
MAX_TOTAL = 20000     # safety cap on total workload
MAX_PAUSE = 120.0     # safety cap on a single fixed pause
COOL_MAX = 180.0      # safety cap on a cool-to-temp wait

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "result.png")
OUT_CSV = os.path.join(HERE, "result.csv")

pynvml.nvmlInit()
H = pynvml.nvmlDeviceGetHandleByIndex(int((os.environ.get("CUDA_VISIBLE_DEVICES","1").split(",")[0] or "1")))
NAME = pynvml.nvmlDeviceGetName(H)
NAME = NAME.decode() if isinstance(NAME, bytes) else NAME
try:
    GPU_MAX = int(pynvml.nvmlDeviceGetTemperatureThreshold(H, getattr(pynvml, "NVML_TEMPERATURE_THRESHOLD_GPU_MAX", 3)))
except pynvml.NVMLError:
    GPU_MAX = 83

_run_lock = threading.Lock()
_T = {}


def tensors():
    if not _T:
        a = torch.randn(GEMM_N, GEMM_N, device="cuda", dtype=torch.float16)
        _T["a"], _T["b"], _T["c"] = a, torch.randn_like(a), torch.empty_like(a)
    return _T["a"], _T["b"], _T["c"]


def gpu_temp():
    return pynvml.nvmlDeviceGetTemperature(H, pynvml.NVML_TEMPERATURE_GPU)


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = False
        self.rows = []
        self.phase = "run"
        self.work = 0
        self.t0 = time.perf_counter()

    def run(self):
        while not self.stop_flag:
            t = time.perf_counter() - self.t0
            try:
                pw = pynvml.nvmlDeviceGetPowerUsage(H) / 1000.0
            except pynvml.NVMLError:
                pw = float("nan")
            self.rows.append((t, gpu_temp(), pynvml.nvmlDeviceGetClockInfo(H, pynvml.NVML_CLOCK_SM), pw, self.phase, self.work))
            time.sleep(0.05)


def run_schedule(total, steps):
    a, b, c = tensors()
    torch.backends.cuda.matmul.allow_tf32 = True
    for _ in range(3):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()

    s = Sampler(); s.start(); time.sleep(0.2)
    s.t0 = time.perf_counter()
    done, i, pause_total, guard = 0, 0, 0.0, 0
    while done < total:
        step = steps[i % len(steps)]; i += 1
        k = min(int(step.get("gemms", 0)), total - done)
        if k <= 0 and not (step.get("pause_s") or step.get("cool_to")):
            guard += 1
            if guard > len(steps):
                break
            continue
        guard = 0
        s.phase = "run"
        j = 0
        while j < k:
            n = min(INNER, k - j)
            for _ in range(n):
                torch.matmul(a, b, out=c)
            torch.cuda.synchronize()
            j += n; done += n; s.work = done
        if done >= total:
            break
        if step.get("cool_to") is not None:
            s.phase = "cool"; tgt = float(step["cool_to"]); tc = time.perf_counter()
            while gpu_temp() > tgt and time.perf_counter() - tc < COOL_MAX:
                time.sleep(0.2)
            pause_total += time.perf_counter() - tc
        elif float(step.get("pause_s", 0) or 0) > 0:
            s.phase = "cool"; p = min(float(step["pause_s"]), MAX_PAUSE)
            time.sleep(p); pause_total += p
    jct = time.perf_counter() - s.t0
    s.stop_flag = True; s.join()
    return s.rows, jct, pause_total, done


def write_outputs(rows, jct, pause_total, done, total, steps):
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_s", "temp_c", "sm_clk_mhz", "power_w", "phase", "work_done"])
        for r in rows:
            w.writerow([round(r[0], 3), r[1], r[2], round(r[3], 1) if r[3] == r[3] else "", r[4], r[5]])

    t = np.array([r[0] for r in rows]); temp = np.array([r[1] for r in rows])
    clk = np.array([r[2] for r in rows]); pw = np.array([r[3] for r in rows])
    work = np.array([r[5] for r in rows]); cool = np.array([r[4] == "cool" for r in rows])

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 1.2]})

    def shade(a):
        d = np.diff(np.concatenate([[0], cool.astype(int), [0]]))
        for si, ei in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            a.axvspan(t[min(si, len(t)-1)], t[min(ei-1, len(t)-1)], color="#1f77b4", alpha=.10)

    a = ax[0]; shade(a)
    a.plot(t, work, "#1f77b4", lw=2.2); a.plot(t[-1], work[-1], "o", color="#1f77b4", ms=8, mec="k")
    a.axhline(total, color="gray", ls="--", alpha=.6); a.set_ylabel("cumulative GEMMs")
    a.set_title(f"JCT = {jct:.1f}s   |   {done}/{total} GEMMs   |   idle/pause {pause_total:.1f}s "
                f"({100*pause_total/max(jct,1e-9):.0f}%)   |   blue = pause")
    a.grid(alpha=.3)

    a = ax[1]; shade(a)
    a.plot(t, temp, "#d62728", lw=1.8)
    a.axhline(GPU_MAX, color="#d62728", ls="--", alpha=.7, label=f"throttle {GPU_MAX}°C")
    a.set_ylabel("temp (°C)"); a.legend(fontsize=8, loc="lower right"); a.grid(alpha=.3)

    a = ax[2]; shade(a)
    a.plot(t, clk, "#2ca02c", lw=1.6); a.set_ylabel("clock (MHz)", color="#2ca02c"); a.grid(alpha=.3)
    a2 = a.twinx(); a2.plot(t, pw, "#7f7f7f", lw=1, alpha=.6); a2.set_ylabel("power (W)", color="#7f7f7f")
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
</style></head><body><div class=wrap>
<h1>Schedule Lab — 手动编排 GPU 工作负载</h1>
<div class=sub>__NAME__ · 降频阈值 __GPUMAX__°C · 每个 GEMM = fp16 __GEMMN__² 矩阵乘 · 结果写入 <code>schedule_lab/result.png</code> + <code>result.csv</code></div>
<div class=row>
 <div class=card>
  <label>总工作量 total（GEMM 个数，≤ __MAXTOTAL__）</label>
  <input id=total value=2000>
  <label>编排 schedule（JSON 步骤列表，循环执行直到做满 total）</label>
  <textarea id=steps>[
  {"gemms": 2000, "pause_s": 0}
]</textarea>
  <div class=hint>每个步骤：<code>gemms</code>=这步跑多少个；然后二选一停顿——<code>pause_s</code>=固定停 N 秒，或 <code>cool_to</code>=空闲到温度≤该值。步骤列表循环,直到累计做满 total。</div>
  <div>
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
const PRE = {
 burst:  {total:2000, steps:[{gemms:2000,pause_s:0}]},
 trickle:{total:2000, steps:[{gemms:50,pause_s:1.0}]},
 coolgate:{total:2000, steps:[{gemms:400,cool_to:72}]},
 ramp:   {total:2000, steps:[{gemms:800,pause_s:0},{gemms:200,pause_s:4},{gemms:100,pause_s:8}]}
};
function preset(k){document.getElementById('total').value=PRE[k].total;
 document.getElementById('steps').value=JSON.stringify(PRE[k].steps,null,2);}
async function run(){
 const msg=document.getElementById('msg'); msg.textContent='';
 let steps; const total=parseInt(document.getElementById('total').value);
 try{steps=JSON.parse(document.getElementById('steps').value);}catch(e){msg.textContent='schedule JSON 解析失败: '+e;return;}
 const btn=document.getElementById('run'); btn.disabled=true; btn.textContent='⏳ 运行中...';
 document.getElementById('result').innerHTML='<div class=hint>运行中,请稍候…(图会在完成后写入 result.png 并显示)</div>';
 try{
  const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({total,steps})});
  const j=await r.json();
  if(j.error){msg.textContent=j.error;document.getElementById('result').innerHTML='';}
  else{document.getElementById('result').innerHTML='<b>JCT = '+j.jct.toFixed(1)+'s</b> · 停顿 '+j.pause.toFixed(1)+'s · 峰值 '+j.peak+'°C · 最低频 '+j.minclk+'MHz · 已写入 result.png/csv<img src="/result.png?t='+j.ts+'">';}
 }catch(e){msg.textContent='运行失败: '+e;}
 btn.disabled=false; btn.textContent='▶ 运行';
}
</script></div></body></html>"""

PAGE = (PAGE.replace("__NAME__", NAME).replace("__GPUMAX__", str(GPU_MAX))
        .replace("__GEMMN__", str(GEMM_N)).replace("__MAXTOTAL__", str(MAX_TOTAL)))


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
            if total <= 0 or total > MAX_TOTAL:
                raise ValueError(f"total 必须在 1..{MAX_TOTAL}")
            if not isinstance(steps, list) or not steps:
                raise ValueError("steps 必须是非空列表")
            if not any(int(s.get("gemms", 0)) > 0 for s in steps):
                raise ValueError("至少有一个步骤的 gemms > 0")
        except Exception as e:
            self._send(200, json.dumps({"error": f"输入错误: {e}"}).encode()); return

        if not _run_lock.acquire(blocking=False):
            self._send(200, json.dumps({"error": "已有一个运行在进行中,请等它结束"}).encode()); return
        try:
            rows, jct, pause_total, done = run_schedule(total, steps)
            write_outputs(rows, jct, pause_total, done, total, steps)
            peak = max(r[1] for r in rows)
            minclk = min(r[2] for r in rows if r[4] == "run")
            self._send(200, json.dumps({"ok": True, "jct": jct, "pause": pause_total,
                                        "peak": peak, "minclk": minclk, "ts": int(time.time())}).encode())
        except Exception as e:
            self._send(200, json.dumps({"error": f"运行出错: {e}"}).encode())
        finally:
            _run_lock.release()


if __name__ == "__main__":
    print(f"Schedule Lab on http://localhost:{PORT}  ({NAME}, throttle {GPU_MAX}°C)")
    print(f"results -> {OUT_PNG}  +  {OUT_CSV}")
    print("warming up GEMM tensors...")
    tensors(); torch.cuda.synchronize()
    print("ready. open the URL (VS Code auto-forwards the port).")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
