"""Real-time thermal-throttling demo.

Continuously samples GPU temperature + SM clock (+ power, util, NVML throttle reasons) at 20 Hz.
Timeline:  BASELINE (idle) -> LOAD (slam full fp16 GEMM to heat the die) -> COOLDOWN (load off).
Expected: under load the temperature climbs past the hardware SLOWDOWN threshold, at which point
the GPU AUTO-THROTTLES the SM clock (NVML reports HW/SW thermal-slowdown) to pull temperature
back down -- exactly the behaviour we want to capture. We deliberately do NOT lock the clock:
the point is to observe the system's own thermal governor.

  CUDA_VISIBLE_DEVICES=0 python3 thermal_throttle/monitor_throttle.py
Writes throttle.csv + meta.json (read by plot_throttle.py). No sudo needed.
"""
from __future__ import annotations
import csv, json, os, threading, time
import pynvml

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE_S, LOAD_S, COOLDOWN_S = 15.0, 90.0, 50.0
HZ = 20
GEMM_N = 8192                     # fp16 matmul size; big enough to peg power and heat fast


def _bit(*names):
    for n in names:
        v = getattr(pynvml, n, None)
        if v is not None:
            return v
    return 0


THERMAL = (_bit("nvmlClocksThrottleReasonSwThermalSlowdown", "nvmlClocksEventReasonSwThermalSlowdown")
           | _bit("nvmlClocksThrottleReasonHwThermalSlowdown", "nvmlClocksEventReasonHwThermalSlowdown"))
POWERCAP = (_bit("nvmlClocksThrottleReasonSwPowerCap", "nvmlClocksEventReasonSwPowerCap")
            | _bit("nvmlClocksThrottleReasonHwPowerBrakeSlowdown", "nvmlClocksEventReasonHwPowerBrakeSlowdown"))


class Sampler(threading.Thread):
    """Background NVML poller: logs at HZ and prints a status line ~1/s (runs during the GEMM too)."""
    def __init__(self, h, slowdown):
        super().__init__(daemon=True)
        self.h, self.slowdown = h, slowdown
        self.dt = 1.0 / HZ
        self.stop_flag = False
        self.samples = []
        self.t0 = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.t0

    def run(self):
        i = 0
        while not self.stop_flag:
            t = self.elapsed()
            temp = pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU)
            clk = pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM)
            try:
                pwr = pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
            except pynvml.NVMLError:
                pwr = float("nan")
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.h).gpu
            except pynvml.NVMLError:
                util = -1
            try:
                reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.h)
            except pynvml.NVMLError:
                reasons = 0
            th_therm = int(bool(reasons & THERMAL))
            th_pow = int(bool(reasons & POWERCAP))
            self.samples.append((t, temp, clk, pwr, util, th_therm, th_pow, reasons))
            if i % HZ == 0:
                flag = " <<< THERMAL-THROTTLE" if th_therm else (" (power-cap)" if th_pow else "")
                hot = " HOT!" if self.slowdown and temp >= self.slowdown else ""
                print(f"  t={t:6.1f}s  temp={temp:3d}C{hot}  clk={clk:4d}MHz  pwr={pwr:5.0f}W  util={util:3d}%{flag}", flush=True)
            i += 1
            time.sleep(self.dt)


def main():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(h)
    if isinstance(name, bytes):
        name = name.decode()

    def thr(t):
        try:
            return int(pynvml.nvmlDeviceGetTemperatureThreshold(h, t))
        except pynvml.NVMLError:
            return None
    slowdown = thr(pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)
    shutdown = thr(pynvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)
    gpumax = thr(getattr(pynvml, "NVML_TEMPERATURE_THRESHOLD_GPU_MAX", 3))
    print(f"{name} | thermal thresholds: slowdown={slowdown}C  shutdown={shutdown}C  gpu_max={gpumax}C", flush=True)
    print(f"timeline: baseline {BASELINE_S:.0f}s -> LOAD {LOAD_S:.0f}s (fp16 {GEMM_N}^2 GEMM) -> cooldown {COOLDOWN_S:.0f}s\n", flush=True)

    s = Sampler(h, slowdown)
    s.start()

    print("== BASELINE (idle) ==", flush=True)
    time.sleep(BASELINE_S)

    print("== LOAD ON: slamming full GEMM to heat the die ==", flush=True)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.randn(GEMM_N, GEMM_N, device="cuda", dtype=torch.float16)
    b = torch.randn(GEMM_N, GEMM_N, device="cuda", dtype=torch.float16)
    c = torch.empty_like(a)
    t_on = s.elapsed()
    t_end = time.perf_counter() + LOAD_S
    while time.perf_counter() < t_end:
        for _ in range(30):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize()
    t_off = s.elapsed()
    del a, b, c
    torch.cuda.empty_cache()

    print("== LOAD OFF: cooldown ==", flush=True)
    time.sleep(COOLDOWN_S)

    s.stop_flag = True
    s.join()
    pynvml.nvmlShutdown()

    with open(os.path.join(HERE, "throttle.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "temp_c", "sm_clk_mhz", "power_w", "util_pct", "throttle_thermal", "throttle_powercap", "reasons_raw"])
        for row in s.samples:
            w.writerow([round(row[0], 3), row[1], row[2], round(row[3], 1) if row[3] == row[3] else "", row[4], row[5], row[6], row[7]])
    meta = dict(name=name, slowdown_c=slowdown, shutdown_c=shutdown, gpu_max_c=gpumax,
                load_on_s=round(t_on, 2), load_off_s=round(t_off, 2),
                peak_temp_c=max(r[1] for r in s.samples),
                min_clk_under_load_mhz=min((r[2] for r in s.samples if t_on <= r[0] <= t_off), default=None),
                thermal_throttle_samples=sum(r[5] for r in s.samples))
    json.dump(meta, open(os.path.join(HERE, "meta.json"), "w"), indent=2)
    print(f"\npeak temp {meta['peak_temp_c']}C (slowdown {slowdown}C) | "
          f"min clk under load {meta['min_clk_under_load_mhz']}MHz | "
          f"{meta['thermal_throttle_samples']} thermal-throttle samples", flush=True)
    print("wrote throttle.csv + meta.json", flush=True)


if __name__ == "__main__":
    main()
