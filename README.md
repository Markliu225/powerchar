# LLM Inference Power Characterisation — Prefill vs Decode

Real measurements on real hardware of **GPU power vs token throughput** for the two
phases of LLM inference — **prefill** (prompt ingestion, compute-bound) and
**decode** (token generation, memory-bandwidth-bound) — plus **analytic `P↔T`
models** validated against the data, and downstream **power-capping** and
**workload-planning** studies built on top of the measured curves.

## Hardware / setup (measured, not assumed)

| | |
|---|---|
| GPU | **NVIDIA Tesla V100-DGXS-32GB** (sm_70), HBM2 fixed at 877 MHz, **250 W** cap |
| Default model | `microsoft/Phi-3-mini-4k-instruct` (fp16, MHA); cross-model set below |
| Multi-model set | `facebook/opt-1.3b`, `Qwen2.5-1.5B / 3B / 7B-Instruct`, `Phi-3-mini-4k-instruct` |
| Control knob | **power cap** (`nvidia-smi -pl`) — sets the operating power, which sets both `T` and `P`; frequency is the intermediate mechanism (see the model docs) |
| Telemetry | NVML (`pynvml`) sampled in a background thread, averaged over the exact timed window |

The whole single-model pipeline is a pure function of [code/config.py](code/config.py)
(model, sweep grids, timing) — override the model with `POWERCHAR_MODEL=…`.

## The analytic models (`P ↔ T`)

- [MODEL_AND_RESULTS.zh.md](MODEL_AND_RESULTS.zh.md) — **the single authoritative
  reference**: full prefill + decode theory (first-principles limits → measured forms)
  plus ALL validation results. Prefill is a single convex `V²f` curve
  `P(T)=P₀+κ·T·(1+ρT)²` (R² 0.956–0.995 on 9/10 workload types); decode is the
  **additive three-stage** model `T(P)=B/(T_mem+C_c[x(P)^{-p}−1])` — compute-dominated
  rise → compute–memory mix → hard **bandwidth plateau** `T_max=B·BW_eff/(W+B·C·kv)`
  (fixed because the HBM clock cannot DVFS), validated over a ~140× ceiling span.
- [pt_cap_gpu1/decode_model_theory.md](pt_cap_gpu1/decode_model_theory.md) — decode
  derivation details + single-card calibration; the old piecewise `min(V²f, T_max)`
  is its perfect-overlap limiting case (kept as the comparison baseline).

**Punchline (energy).** At the same near-cap power, prefill delivers far more tokens/J
than decode (~10×): prefill reuses each weight tile across the whole prompt, while
decode re-streams all weights every step to emit only `batch` tokens.

## The measurement engine — `code/`

```
code/config.py        all experiment parameters (model, sweeps, timing, roofline)
code/model_info.py    arch extraction + peak-FLOP/BW microbench + roofline
code/power_sampler.py NVML sampler with windowed aggregation
code/measure.py       prefill & decode batch sweeps  -> results/*.csv  (core; imported by the sweeps)
code/measure_dvfs.py  clock-locked DVFS sweep (needs admin)
code/analyze.py       single-model fits + figures    -> figures/
code/*_sweep.py       power-cap / clock / goodput sweeps that feed the sub-experiments
                      (decode_*_sweep.py, pt_cap_sweep.py -> pt_cap_gpu1/;
                       decode_sat*.py -> results/decode_saturation*.csv;
                       goodput_cap_sweep.py -> results/goodput_*.csv)
code/plot_*.py        standalone plotters for the above
results/              measured CSVs + model_info.json + fit_summary.json + multi-model mm_*.{csv,json}
figures/              regenerated locally by the pipeline (not committed)
```

## Sub-experiments (each self-contained, with its own doc + data + figures)

| dir | what it is | doc |
|---|---|---|
| [pt_cap_gpu1/](pt_cap_gpu1/) | decode power-capping `P(T)`: additive `T_mem+T_comp` three-stage model, fitted to a V100 clock/power sweep | [decode_model_theory.md](pt_cap_gpu1/decode_model_theory.md) |
| [rack_power_capping/](rack_power_capping/) | rack-level economics/optimisation on the `P(T)` curves — how to split prefill/decode GPUs and cap each to maximise goodput per watt / payback | [WORKLOAD_PORTFOLIO.zh.md](rack_power_capping/WORKLOAD_PORTFOLIO.zh.md) |
| [workload_analysis/](workload_analysis/) | prefill:decode token ratios by use-case, from real production traces (Azure, BurstGPT) + Dolly-15k | [README](workload_analysis/README.md) · [REFERENCES.zh.md](workload_analysis/REFERENCES.zh.md) |
| [schedule_lab/](schedule_lab/) | interactive tool to hand-schedule a GPU workload and watch temperature/clock/power; includes a thermal-throttle probe | [README](schedule_lab/README.md) · [thermal_throttle](schedule_lab/thermal_throttle/README.md) |

## How to run

```bash
pip install -r requirements.txt

# single model: model_info -> measure(prefill+decode) -> analyze (figures + fits)
./run.sh

# cross-model validation of the P(T) model (locks the SM clock; needs sudo)
SUDO_PASS='<pw>' bash run_multimodel.sh
```

Each sub-experiment is run from its own directory — see the per-directory docs above.
