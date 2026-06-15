# LLM Inference Power Characterisation — Prefill vs Decode

Real measurements on real hardware of **token throughput vs GPU power** for the
two phases of LLM inference, plus an **analytic model** validated against the
data. Everything is reproducible from a single config file.

- **What is measured** — for a *given* model + parameter config, the
  throughput↔power relationship in **prefill** (prompt ingestion, compute-bound)
  and **decode** (token generation, memory-bandwidth-bound), swept separately.
- **The model** — roofline + DVFS derivations in
  [ANALYTIC_MODEL.md](ANALYTIC_MODEL.md), fitted to the data with R² and MAPE.
- **The plan** — the step-by-step requirements in [WORKPLAN.md](WORKPLAN.md).

## Setup (measured, not assumed)

| | |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` — 1.544 B params, 28 layers, d=1536, GQA 12q:2kv×128, fp16, SDPA |
| GPU | NVIDIA GeForce RTX 5060, 8 GB, **149 W** cap (Blackwell sm_120) |
| Measured peak | **38.1 TFLOP/s** fp16, **372 GB/s** bandwidth → roofline ridge **102 FLOP/byte** |
| Host | Windows 11, driver 591.86 / CUDA 13.1, PyTorch 2.11+cu128, transformers 5.x |
| Telemetry | NVML (`pynvml`) sampled at 50 Hz, averaged over the exact timed window |

## How to run

```bash
pip install -r requirements.txt
python code/model_info.py            # Step 0: model+GPU constants, roofline
python code/measure.py --phase both  # Steps 1-2: prefill + decode sweeps -> CSVs
python code/analyze.py --step all    # Steps 1-4: all figures + model fits
```

## Results

### Step 0 — the chip
![roofline](figures/step0_roofline.png)

Decode (arithmetic intensity ≈ batch) lives left of the ridge → **memory-bound**;
prefill (intensity ≈ seq_len) lives far right → **compute-bound**.

The relationship is read off **power (W) vs token throughput (tok/s)** — and it
is qualitatively different in the two phases.

### Step 1 — Prefill: power DECOUPLED from throughput (compute-bound)
![prefill power vs throughput](figures/step1_prefill_power_vs_throughput.png)

Once the sequence fills the GPU (S ≳ 256), **power locks to ~142 W (±1.5 %) while
throughput sweeps 3.7 → 8.7 k tok/s** — a 2.4× throughput range at constant
power. The matmul units run pinned at the compute roof, so power can't move;
throughput is set entirely by the per-token attention cost (O(S²)). You change
throughput by changing the sequence, **not** by spending more power.

### Step 2 — Decode: power COUPLED to throughput (memory-bandwidth-bound)
![decode power vs throughput](figures/step2_decode_power_vs_throughput.png)

Power and throughput **rise together** with batch, 70 W/29 tok/s → 135 W/749
tok/s: each step re-reads all 3.09 GB of weights to emit only `batch` tokens, so
raising throughput means raising bandwidth utilisation, which costs power. Beyond
b≈48 the KV cache exhausts the 8 GB VRAM (shared ~1.7 GB with the desktop) and
WDDM spills to host, collapsing throughput — a hard wall, documented and excluded.

### Steps 3–4 — analytic model `P(T)` & synthesis
| | |
|---|---|
| ![decode model](figures/step3_decode_model.png) | ![prefill model](figures/step3_prefill_model.png) |

Both follow from one principle — `P = P_static + (P_cap−P_static)·u` and
`T = R/c` (utilisation-driven power, throughput = bottleneck-rate ÷ per-token
cost):

| phase | `P(T)` model | fit | key result |
|---|---|---|---|
| decode | `P(T)` = compose `T=B/(t_fixed+β·B)` with `P=P_idle+A(1−e^{−B/B₀})` | **R²=0.982**, MAPE 3.5 % | coupled; 54→139 W, asymptote 1.5 k tok/s |
| prefill | `P ≈ P_cap` (R pinned at roof); `1/T = a+b·S` sets T | power **CV 1.5 %**; T-law R²=0.996 | decoupled; 142 W ⊥ throughput, MFU 78 % |

![combined](figures/step4_combined_power_vs_throughput.png)

**Energy:** prefill **26–60 tok/J** vs decode **0.4–5.6 tok/J** — ~11× more
efficient per token at best (`figures/step4_combined_efficiency_vs_throughput.png`).

Full derivations and the measured-vs-theory discussion: [ANALYTIC_MODEL.md](ANALYTIC_MODEL.md).

## Caveats / next experiment

GPU clock-lock and power-limit control need admin on this Windows host (denied),
so the **DVFS "≈cubic" power law** `P ∝ tput^γ` could not be measured directly —
it is derived analytically (ANALYTIC_MODEL §5). At the fixed clock ceiling the
two phases instead trace the vertical (prefill) and diagonal (decode) loci above.
Measuring the cubic law directly is the natural follow-up on a clock-controllable
host. Also, no flash/mem-efficient SDPA kernel exists for this sm_120 build, so
prefill attention is O(S²) in memory and hits the 8 GB wall at S≈5 k.

## Files
```
code/config.py        all experiment parameters (model, sweeps, timing)
code/model_info.py    Step 0: arch extraction + peak-FLOP/BW microbench + roofline
code/measure.py       Steps 1-2: prefill & decode sweeps -> results/*.csv
code/analyze.py       Steps 1-4: fits + every figure
code/power_sampler.py 50 Hz NVML sampler with windowed aggregation
results/              *.csv, model_info.json, fit_summary.json
figures/              step0..step4 PNGs
```
