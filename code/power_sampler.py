"""High-frequency GPU telemetry via NVML.

A background thread samples power / utilisation / clocks / temperature with a
wall-clock timestamp on every sample. A benchmark records the synchronized
[t0, t1] wall-clock window it ran in and asks `stats_between` to average only the
samples inside that window -- so warmup and tail never contaminate the result,
and power is averaged over *exactly* the interval throughput is computed over.
"""
from __future__ import annotations
import os
import statistics
import threading
import time
import pynvml


def _visible_gpu_index() -> int:
    """Physical NVML index of the GPU torch is using. NVML ignores CUDA_VISIBLE_DEVICES and
    indexes physically, so when CUDA_VISIBLE_DEVICES=1 we must sample physical GPU 1, not 0."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(cvd) if cvd.isdigit() else 0   # single healthy GPU now enumerates at index 0


class PowerSampler:
    def __init__(self, device_index: int | None = None, interval_s: float = 0.02):
        self.device_index = _visible_gpu_index() if device_index is None else device_index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict] = []
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self.power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._h) / 1000.0
        self.name = pynvml.nvmlDeviceGetName(self._h)
        if isinstance(self.name, bytes):
            self.name = self.name.decode()

    def _loop(self):
        h = self._h
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                sm = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                mem_clk = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                self.samples.append({
                    "t": t, "power_w": power, "util_gpu": u.gpu,
                    "util_mem": u.memory, "sm_clk": sm, "mem_clk": mem_clk,
                    "temp": temp,
                })
            except pynvml.NVMLError:
                pass
            time.sleep(self.interval_s)

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    @staticmethod
    def now() -> float:
        return time.perf_counter()

    def stats_between(self, t0: float, t1: float) -> dict | None:
        """Aggregate samples whose timestamp falls in the half-open [t0, t1)."""
        win = [s for s in self.samples if t0 <= s["t"] < t1]
        if not win:
            return None
        n = len(win)
        avg = lambda k: sum(s[k] for s in win) / n
        powers = [s["power_w"] for s in win]
        return {
            "n_samples": n,
            "power_avg_w": avg("power_w"),
            "power_max_w": max(powers),
            "power_p50_w": statistics.median(powers),
            "power_std_w": statistics.pstdev(powers) if n > 1 else 0.0,
            "util_gpu_avg": avg("util_gpu"),
            "util_mem_avg": avg("util_mem"),
            "sm_clk_avg": avg("sm_clk"),
            "sm_clk_min": min(s["sm_clk"] for s in win),
            "sm_clk_max": max(s["sm_clk"] for s in win),
            "mem_clk_avg": avg("mem_clk"),
            "temp_avg": avg("temp"),
            "window_s": t1 - t0,
        }

    def shutdown(self):
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass
