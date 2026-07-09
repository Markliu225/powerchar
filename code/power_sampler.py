"""High-frequency GPU telemetry via NVML.

A background thread samples power / utilisation / clocks / temperature with a
wall-clock timestamp on every sample. A benchmark records the synchronized
[t0, t1] wall-clock window it ran in and asks `stats_between` to average only the
samples inside that window -- so warmup and tail never contaminate the result,
and power is averaged over *exactly* the interval throughput is computed over.

Window power uses the ENERGY-ACCUMULATOR method when the driver supports it
(energy-accumulator method): P = (E(t1)-E(t0)) / (t1-t0) from the
monotonic nvmlDeviceGetTotalEnergyConsumption counter. On Hopper the classic
nvmlDeviceGetPowerUsage is a ~1 s sliding average that smears and lags power-cap
steps, so the sampled mean is kept only as `power_sample_avg_w` for comparison;
on Volta the two agree to ~1%. Falls back to the sampled mean when the energy
counter is unsupported.

Each sample also records the CLOCKS-EVENT/THROTTLE reason bitmask when available
(throttle-reason gating): stats_between returns the window OR-mask plus the fraction
of samples showing thermal bits, so sweep code can gate out thermally
contaminated points instead of publishing them.
"""
from __future__ import annotations
import os
import statistics
import threading
import time
import pynvml

# Thermal / protective throttle-reason bits (throttle-reason gating). Power-cap and
# applications-clocks bits are the EXPECTED reasons during a cap sweep.
THERMAL_BITS = 0x20 | 0x40 | 0x08 | 0x80   # SwThermal | HwThermal | HwSlowdown | HwPowerBrake


def _visible_gpu_index() -> int:
    """Physical NVML index of the GPU torch is using. NVML ignores CUDA_VISIBLE_DEVICES and
    indexes physically, so when CUDA_VISIBLE_DEVICES=1 we must sample physical GPU 1, not 0."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(cvd) if cvd.isdigit() else 0


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
        # capability probes (unsupported on some GPUs/drivers -> feature off, no error spam)
        self.has_energy = self._probe(lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(self._h))
        self.has_throttle = self._probe(lambda: self._throttle_reasons())
        mt = self._mem_temp()
        self.has_mem_temp = mt is not None and 0 < mt < 130   # sane HBM junction range

    @staticmethod
    def _probe(fn) -> bool:
        try:
            fn()
            return True
        except Exception:
            return False

    def _throttle_reasons(self) -> int:
        """Clocks-event (throttle) reason bitmask; NVML renamed this API across versions."""
        for attr in ("nvmlDeviceGetCurrentClocksEventReasons",
                     "nvmlDeviceGetCurrentClocksThrottleReasons"):
            fn = getattr(pynvml, attr, None)
            if fn is not None:
                return fn(self._h)
        raise pynvml.NVMLError(pynvml.NVML_ERROR_NOT_SUPPORTED)

    def _mem_temp(self):
        """HBM junction temperature via the field-values API (NVML_FI_DEV_MEMORY_TEMP).
        (pynvml has no NVML_TEMPERATURE_MEM sensor enum -- passing a made-up sensor id to
        nvmlDeviceGetTemperature is undefined; the field-values route is the correct one.)"""
        fid = getattr(pynvml, "NVML_FI_DEV_MEMORY_TEMP", None)
        if fid is None:
            return None
        try:
            fv = pynvml.nvmlDeviceGetFieldValues(self._h, [fid])[0]
            if getattr(fv, "nvmlReturn", 1) != 0:
                return None
            v = fv.value.uiVal or fv.value.ullVal
            return float(v) if v else None
        except Exception:
            return None

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
                s = {
                    "t": t, "power_w": power, "util_gpu": u.gpu,
                    "util_mem": u.memory, "sm_clk": sm, "mem_clk": mem_clk,
                    "temp": temp,
                }
                if self.has_energy:
                    s["energy_mj"] = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
                if self.has_throttle:
                    s["throttle"] = self._throttle_reasons()
                if self.has_mem_temp:
                    mt = self._mem_temp()
                    if mt is not None:
                        s["mem_temp"] = mt
                self.samples.append(s)
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
        out = {
            "n_samples": n,
            "power_sample_avg_w": avg("power_w"),
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
        # primary power figure: energy-accumulator over the window when available
        es = [s["energy_mj"] for s in win if "energy_mj" in s]
        if len(es) >= 2 and win[-1]["t"] > win[0]["t"]:
            out["power_energy_w"] = (es[-1] - es[0]) / 1000.0 / (win[-1]["t"] - win[0]["t"])
            out["power_avg_w"] = out["power_energy_w"]
        else:
            out["power_avg_w"] = out["power_sample_avg_w"]
        # throttle gating info (throttle-reason gating)
        ths = [s.get("throttle", 0) for s in win if "throttle" in s]
        if ths:
            mask = 0
            for m in ths:
                mask |= m
            out["throttle_mask"] = mask
            out["thermal_frac"] = sum(1 for m in ths if m & THERMAL_BITS) / len(ths)
        if any("mem_temp" in s for s in win):
            mts = [s["mem_temp"] for s in win if "mem_temp" in s]
            out["mem_temp_avg"] = sum(mts) / len(mts)
        return out

    def shutdown(self):
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass
