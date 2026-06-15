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
- Sweep prefill tokens (64 → 4096 at batch 1) at `use_cache=False`.
- Figures (x-axis = **token throughput, tok/s**): **power vs throughput**,
  efficiency (tok/J) vs throughput.
- Expected: once the GPU saturates, power pins at ~cap while throughput varies
  with attention O(S²) → **power decoupled from throughput**.

### Step 2 — Decode characterisation (memory-bandwidth-bound)
`code/measure.py --phase decode` → `results/decode.csv`, then `analyze.py --step 2`
- Sweep batch size (1 → 256) at fixed 256-token context, steady-state single-token steps.
- Figures (x-axis = **token throughput, tok/s**): **power vs throughput**,
  efficiency (tok/J) vs throughput.
- Expected: power and throughput **rise together** with batch (coupled), toward
  the bandwidth/power ceiling.

### Step 3 — Analytic model `P(T)` & validation
[ANALYTIC_MODEL.md](ANALYTIC_MODEL.md) + `analyze.py --step 3` → `results/fit_summary.json`
and `figures/step3_*.png`
- Derive **power as a function of throughput** from one principle:
  `P = P_static + (P_cap−P_static)·u` and `T = R/c`. Decode: R ramps with batch →
  coupled `P(T)`. Prefill: R pinned at the compute roof → flat `P(T)`, T set by
  the per-token cost `c(S) = C + k_attn·S`. Plus the DVFS context law.
- Overlay each `P(T)` model on the measured points; report MAPE / R², MFU,
  power CV, and the throughput asymptote.

### Step 4 — Synthesis
`analyze.py --step 4` → `figures/step4_*.png` + README results table
- Both phases on **one power-vs-throughput axis** (the coupled vs decoupled laws).
- Energy efficiency (tok/J) vs throughput, quantifying why prefill is far cheaper
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
