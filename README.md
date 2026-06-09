# LLM Inference Power Curves: Prefill vs Decode

Real measurements on real hardware of token-throughput and GPU power for the two
phases of LLM inference, swept from light load to full GPU saturation.

## Setup (measured, not assumed)

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 (Blackwell, sm_120), 8 GB, **145 W** power cap |
| Driver / CUDA | 580.95.05 / CUDA 13 runtime, torch 2.9.1+cu128 |
| Model | **Qwen/Qwen2.5-1.5B-Instruct**, fp16 (~3.09 GB weights), SDPA attention |
| Stack | transformers 5.10.2, Python 3.10 (`ebpf-cupti` conda env) |
| Telemetry | NVML (`pynvml`) sampled at 50 Hz in a background thread |

## Method

- **Prefill** (`bench.run_prefill_point`): repeated full forwards over a
  `(batch × seq_len)` block with `use_cache=False`, swept by total tokens per
  forward (64 → 32 768). `logits_to_keep=1` so only the last position is
  projected to vocab — this matches real prefill (only the first output token's
  logits are needed) and avoids a `(B, S, 151936)` fp16 logits tensor that OOMs
  an 8 GB card past ~12 k tokens.
- **Decode** (`bench.run_decode_point`): manual single-token autoregressive
  steps reusing a `DynamicCache`, swept by batch size (1 → 256) at a fixed
  256-token context. The KV cache is seeded in small chunks so setup activations
  don't spike (lets batch reach 256); only the steady-state single-token steps
  are timed — `generate()` is deliberately avoided to keep prefill cost out.
- Each point runs a **sustained ~4 s window** after a 1 s warmup + 0.3 s settle,
  so power reaches steady state and 200–500 NVML samples are collected.
  Throughput is computed over the **exact synchronized `[t0, t1]` wall-clock
  window** that power is averaged over; CUDA events cross-check the timing.
- OOM points are caught and skipped (they mark the memory ceiling).

The methodology was adversarially reviewed by a 5-agent workflow before running;
the one material fix it surfaced (the logits-tensor OOM, root-caused over the
attention-matrix red herring) is applied above.

## Results

### Prefill — compute-bound: power saturates immediately, ~45–85 tok/J
| load (B×S) | throughput (tok/s) | power (W) | util | tok/J |
|---|---|---|---|---|
| 1×64    |  5 079 | 113.0 | 87% | 44.9 |
| 1×512   | 11 208 | 136.5 | 97% | 82.1 |
| 1×1024  | **11 603** | 137.2 | 77% | **84.6** |
| 1×8192  |  9 449 | 139.2 | 100% | 67.9 |
| 4×8192  |  9 440 | **142.2** | 91% | 66.4 |
| 2×16384 |  8 073 | 140.9 | 92% | 57.3 |

Power hits ~136 W at just 128 tokens and stays 136–142 W everywhere — prefill is
compute-bound, so it draws near-cap power regardless of throughput. Throughput
peaks ~11.6 k tok/s near 1 k tokens, then falls as attention's O(S²) cost grows.

### Decode — memory-bandwidth-bound: power & throughput scale with batch
| batch | throughput (tok/s) | power (W) | util | tok/J |
|---|---|---|---|---|
| 1   |    83 | 113.6 | 94% | 0.7 |
| 8   |   642 |  99.1 | 92% | 6.5 |
| 32  | 1 780 | 122.9 | 95% | 14.5 |
| 64  | 3 520 | 138.6 | 95% | 25.4 |
| 96  | 4 122 | **144.8** | 96% | 28.5 |
| 256 | **5 265** | 145.1 | 98% | **36.3** |

Decode power climbs from ~90 W to the 145 W cap by batch ≈ 96; throughput rises
monotonically to ~5.3 k tok/s. Per-token energy efficiency is **2–60× worse than
prefill** at comparable points because each step re-reads all weights to emit
only `batch` tokens — the textbook memory-bandwidth bottleneck. (batch=1 power is
elevated to ~114 W by max clock-boost on the latency-bound single stream.)

## Files
- `power_sampler.py` — 50 Hz NVML sampler, windowed aggregation
- `bench.py` — the benchmark (run: `python bench.py --out results`)
- `plot.py` — generates the three PNGs from the CSVs
- `results_prefill.csv`, `results_decode.csv` — raw per-point data
- `curves_throughput_vs_power.png` — the two requested curves
- `curves_efficiency.png` — log-x comparison + tokens/joule
- `curves_vs_load.png` — throughput & power vs offered load (clearest view)

Reproduce: `HF_HUB_DISABLE_XET=1 python bench.py && python plot.py`
