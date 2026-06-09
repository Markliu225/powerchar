"""High-frequency GPU telemetry sampler using NVML.

Runs in a background thread, sampling power / utilization / clocks / temp.
Samples are timestamped so a benchmark can later average over an exact
[t_start, t_end] measurement window (excluding warmup and tail).
"""
import statistics
import threading
import time
import pynvml


class PowerSampler:
    def __init__(self, device_index=0, interval_s=0.02):
        self.device_index = device_index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None
        self.samples = []  # list of dicts: t, power_w, util_gpu, util_mem, sm_clk, temp
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._h) / 1000.0

    def _loop(self):
        h = self._h
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                sm = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                self.samples.append({
                    "t": t, "power_w": power, "util_gpu": u.gpu,
                    "util_mem": u.memory, "sm_clk": sm, "temp": temp,
                })
            except pynvml.NVMLError:
                pass
            # busy-aware sleep
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

    def now(self):
        return time.perf_counter()

    def stats_between(self, t0, t1):
        """Aggregate samples whose timestamp falls in the half-open [t0, t1).

        Half-open avoids double-counting a boundary sample if windows abut.
        Warmup/settle samples are naturally excluded: SETTLE_S elapses before
        the caller records t0, so no pre-steady-state sample is in range.
        """
        win = [s for s in self.samples if t0 <= s["t"] < t1]
        if not win:
            return None
        n = len(win)
        def avg(k):
            return sum(s[k] for s in win) / n
        def mx(k):
            return max(s[k] for s in win)
        powers = [s["power_w"] for s in win]
        return {
            "n_samples": n,
            "power_avg_w": avg("power_w"),
            "power_max_w": mx("power_w"),
            "power_p50_w": statistics.median(powers),
            "util_gpu_avg": avg("util_gpu"),
            "util_mem_avg": avg("util_mem"),
            "sm_clk_avg": avg("sm_clk"),
            "temp_avg": avg("temp"),
            "window_s": t1 - t0,
        }

    def shutdown(self):
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass
