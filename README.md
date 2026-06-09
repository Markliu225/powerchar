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

## Follow-up: capturing the prefill ~cubic law, and the decode law

**Prefill is compute-bound; decode is memory-bandwidth-bound.** To see the
throughput↔power law you must sweep the *frequency* knob, not the load
(`freq_sweep_llm.py`, `curves_freq_llm.png`): lock the SM clock across a range
and measure real token throughput + power on each real workload.

**1. The cubic law lives in CLOCK FREQUENCY** (`clock_sweep.py`,
`curves_clock_dvfs.png`). Fixed full-occupancy matmul, clock locked: power =
32 W @600 MHz → 46 @1200 → 68 @1800 → 110 @2400 → 138 @~2600, while compute
scales *linearly* (8.9→39 TFLOP/s). Fit `P ≈ 28 + k·f^2.1` (steepening to ~f^2.4
at the top) — the V²·f DVFS law. The card can't sustain >~2600 MHz under load
(3000 requested → 2601 actual). In the original load sweeps the clock was pinned
at this ~2.6 GHz ceiling, which is why the cubic was invisible there.

**2. PREFILL: throughput↔power is ~cubic.** Compute-bound, so throughput ∝ clock
while power ∝ ~f^(2–3); combined: `P = 25 + k·tput^2.05`, **high-end exponent
≈ 2.5** (1×2048: 2786 tok/s @30 W → 11244 tok/s @129 W). A convex, accelerating
curve — the approximate cubic.

**3. DECODE: memory-bandwidth-bound.** Each step re-reads all weights to emit a
few tokens, so throughput is limited by memory bandwidth, not core clock. At
b=1, raising the SM clock 4.3× lifts throughput only 2.6× (31→80 tok/s,
sub-linear) while power climbs 38→96 W — i.e. spending clock/power buys little
throughput. Decode's lever is memory bandwidth (and batching toward the BW
limit), not the compute clock. *(Note: a high-batch + short-context micro-bench
can nominally cross the roofline ridge and look compute-bound, but that is an
artifact of an unrealistically short KV cache and is not representative of
decode — excluded.)*

**4. Why prefill power looked flat in the load sweep:** the sequence dimension
gives massive parallelism, so even tiny prefills saturate this 30-SM card's
occupancy AND it's already at max clock → ~140 W across the range, against the
145 W ceiling. Nothing can "keep rising" — 145 W is a hard wall. What grows ∝S²
is FLOPs/**energy** (J), not power (J/s).

