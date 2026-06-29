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
| Knob | **batch (concurrency)** at a fixed sequence/context length — the controlled variable that makes throughput monotone, so `P(T)` is single-valued. GPU clock/power-limit control needs admin on this Windows box and is **not** available, so the DVFS knob is treated analytically, not measured. |

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

Both phases use the SAME **controlled experiment**: fix the sequence/context
length, sweep **batch (concurrency)**. Holding the per-token cost constant makes
throughput a *monotone* function of the one swept variable, so `P(T)` is
single-valued (one throughput → one power). Sweeping sequence length instead
folds prefill's `P(T)` (throughput is non-monotone in S) — the failure mode this
design fixes.

### Step 1 — Prefill characterisation (compute-bound)
`code/measure.py --phase prefill` → `results/prefill.csv`, then `analyze.py --step 1`
- Fixed prompt length **S=128**, sweep batch **1 → 16**, `use_cache=False`.
- Figures (x-axis = **token throughput, tok/s**): **power vs throughput**,
  efficiency (tok/J) vs throughput.
- Expected: power rises with throughput toward the **compute roof** (~12 k tok/s).

### Step 2 — Decode characterisation (memory-bandwidth-bound)
`code/measure.py --phase decode` → `results/decode.csv`, then `analyze.py --step 2`
- Fixed context **ctx=256**, sweep batch **1 → 48** (steady-state single-token steps).
- Figures (x-axis = **token throughput, tok/s**): **power vs throughput**,
  efficiency (tok/J) vs throughput.
- Expected: power rises with throughput toward the **memory/overhead ceiling**
  (~1.5 k tok/s) — ~14× lower than prefill.

### Step 3 — Analytic model `P(T)` & validation
[THEORY.zh.md](THEORY.zh.md) + `analyze.py --step 3` → `results/fit_summary.json`
and `figures/step3_*.png`
- Compose two measured batch-domain laws into a single-valued `P(T)`: affine step
  time `t(B)=t_fixed+β·B` (→ throughput ceiling `n/β`) and saturating power
  `P(B)=P_idle+A(1−e^{−B/B₀})`. The ceiling is set by the bottleneck — compute
  roof `Φ·MFU/c` (prefill) vs memory/overhead `1/β` (decode).
- Overlay each `P(T)` model on the measured points; report MAPE / R², MFU, and
  the throughput ceiling.

### Step 4 — Synthesis
`analyze.py --step 4` → `figures/step4_*.png` + README results table
- Both phases on **one power-vs-throughput axis** (same rising shape, ~14×
  different throughput ceiling).
- Energy efficiency (tok/J) vs throughput, quantifying why prefill is far cheaper
  per token than decode.

### Step 5 — DVFS / the ≈cubic law (the *other* knob)
`code/measure_dvfs.py` (needs admin clock-lock) → `analyze.py --step dvfs` →
`figures/step5_dvfs_cubic.png`
- Pin one workload, sweep the SM clock 600→2700 MHz. For compute-bound prefill
  `T ∝ f` and `P ∝ f^~2.5`, so `P ∝ T^~3` — the **measured cubic law**
  (`P ≈ 31 + k·T^2.94`, R²=0.989). Decode `T ∝ f^0.5` (clock barely helps).
- Shows that the concurrency sweep (linear P–T) and the frequency sweep (cubic
  P–T) are two different knobs on the same GPU.

---

## Repository layout (after refactor)

```
config.py / power_sampler.py / measure.py / model_info.py / analyze.py   (code/)
WORKPLAN.md          this file
THEORY.zh.md    derivations + measured-vs-theory comparison
README.md            overview, how-to-run, results summary
results/             *.csv, model_info.json, fit_summary.json, dvfs.csv
figures/             step0..step5 PNGs
```

## Reproduce

```
pip install -r requirements.txt
python code/model_info.py            # Step 0
python code/measure.py --phase both  # Steps 1 & 2 (data)
python code/analyze.py --step all    # Steps 1-4 (figures + fits)
```
