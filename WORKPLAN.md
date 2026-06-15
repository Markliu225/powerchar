# Work Plan — Prefill vs Decode: token-throughput ↔ power characterisation

**Goal.** For a *given* LLM and parameter configuration, separately measure the
relationship between **token throughput** and **GPU power** in the two phases of
inference — **prefill** (prompt ingestion) and **decode** (autoregressive
generation) — on real hardware, then build an **analytic model** and check the
measurements against it.

## Given configuration (the fixed inputs)

| | |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, fp16, SDPA attention |
| GPU | NVIDIA GeForce RTX 5060, 8 GB, ~149 W enforced cap (Blackwell) |
| Host | Windows 11, driver 591.86 / CUDA 13.1, PyTorch 2.11+cu128, transformers 5.x |
| Telemetry | NVML (`pynvml`) sampled at 50 Hz in a background thread |
| Knob | **offered load** (prefill: total tokens/forward; decode: batch size). GPU clock/power-limit control needs admin on this Windows box and is **not** available, so DVFS is treated analytically, not measured. |

Everything that defines the experiment lives in [code/config.py](code/config.py);
the rest of the code is a pure function of it.

## Measurement protocol (applies to every point)

- Warm up `WARMUP_S` (1 s) → settle `SETTLE_S` (0.3 s) → timed window `MEASURE_S` (4 s).
- Throughput is computed over the **exact synchronized `[t0,t1]` wall-clock
  window** that power is averaged over; CUDA events cross-check timing.
- Power/util/clocks averaged only over samples inside `[t0,t1]` (warmup/tail excluded).
- OOM points are caught and skipped — they mark the 8 GB memory ceiling.

---

## Steps (each produces ≥1 figure + a written analysis)

### Step 0 — Configuration & machine characterisation
`code/model_info.py` → `results/model_info.json`, `figures/step0_roofline.png`
- Extract the model's architecture (layers, d_model, heads, KV-heads, params) —
  the parameter configuration that feeds the analytic model.
- **Measure** this card's peak fp16 matmul throughput and peak memory bandwidth
  (so the roofline uses real constants, not datasheet numbers).
- Deliverable figure: the **roofline** with prefill (high arithmetic intensity)
  and decode (low arithmetic intensity ≈ batch) regimes placed on it.

### Step 1 — Prefill characterisation (compute-bound)
`code/measure.py --phase prefill` → `results/prefill.csv`, then `analyze.py --step 1`
- Sweep total prefill tokens (64 → ~130 k) at `use_cache=False`.
- Figures: **throughput vs power**, throughput & power vs offered load, tok/J vs load.
- Expected: power pinned near the cap (compute-bound, clock at ceiling);
  throughput rises with parallelism then falls as attention O(S²) grows.

### Step 2 — Decode characterisation (memory-bandwidth-bound)
`code/measure.py --phase decode` → `results/decode.csv`, then `analyze.py --step 2`
- Sweep batch size (1 → 256) at fixed 256-token context, steady-state single-token steps.
- Figures: **throughput vs power**, throughput & power vs batch, tok/J vs batch.
- Expected: power climbs from idle-ish to the cap as batch grows; throughput
  rises ~linearly (bandwidth-bound) then bends toward the roofline ridge.

### Step 3 — Analytic model & validation
[ANALYTIC_MODEL.md](ANALYTIC_MODEL.md) + `analyze.py --step 3` → `results/fit_summary.json`
and `figures/step3_*.png`
- Derive, from first principles: prefill compute roofline with the O(S²)
  attention term; decode memory-bandwidth roofline; the DVFS power law
  P ≈ P_static + k·f·V² (≈ cubic in clock); and a power-vs-utilisation
  saturation law.
- Fit the measured data to each model (numpy least-squares), overlay
  predicted vs measured, and report MAPE / R² and the implied efficiencies
  (MFU, bandwidth utilisation).

### Step 4 — Synthesis
`analyze.py --step 4` → `figures/step4_*.png` + README results table
- One axis comparing prefill and decode throughput-vs-power.
- Energy-efficiency comparison (tok/J) quantifying why prefill is far cheaper
  per token than decode.

---

## Repository layout (after refactor)

```
config.py / power_sampler.py / measure.py / model_info.py / analyze.py   (code/)
WORKPLAN.md          this file
ANALYTIC_MODEL.md    derivations + measured-vs-theory comparison
README.md            overview, how-to-run, results summary
results/             *.csv, model_info.json, fit_summary.json
figures/             step0..step4 PNGs
```

## Reproduce

```
pip install -r requirements.txt
python code/model_info.py            # Step 0
python code/measure.py --phase both  # Steps 1 & 2 (data)
python code/analyze.py --step all    # Steps 1-4 (figures + fits)
```
