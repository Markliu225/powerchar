# Analytic Model: GPU power as a function of token throughput

The question is **how GPU power `P` relates to token throughput `T`** in each
phase of inference. This document derives `P(T)` from one principle and validates
it against the measurements. All machine constants are **measured on this card**
(Step 0, `results/model_info.json`); fitted parameters and the overlay figures
come from `code/analyze.py --step 3` (`results/fit_summary.json`,
`figures/step3_*.png`).

## 1. One governing principle

Two equations explain every curve:

```
(1)  P = P_static + (P_cap − P_static) · u        power tracks utilisation u ∈ [0,1]
(2)  T = R / c                                     throughput = bottleneck rate / per-token cost
```

- `u` is how busy the **bottleneck resource** is. Dynamic GPU power is
  `∝ activity`, so power rises from a static floor toward the cap as `u → 1`.
- `R` is the **rate of the bottleneck resource**: FLOP/s when compute-bound,
  byte/s when memory-bandwidth-bound. `u = R / R_peak`.
- `c` is the **work per token**: FLOPs/token (prefill) or bytes/token (decode).

The whole result follows from *which variable moves* when you change the knob:

| phase | bottleneck | what the knob (seq_len / batch) changes | consequence |
|---|---|---|---|
| **prefill** | compute | `c` grows (attention O(S²)); `R` stays pinned at the roof | `P` fixed, `T` falls → **decoupled** |
| **decode** | memory BW | `R` grows (more bandwidth used); `c` falls (weights amortised) | `P` and `T` rise together → **coupled** |

So **prefill** gives a near-horizontal `P(T)` (power independent of throughput),
**decode** a rising-saturating `P(T)` (you buy throughput with power). The rest of
the document derives both quantitatively.

## 2. Notation and measured constants

| symbol | meaning | value |
|---|---|---|
| `P_tot` / `P_ne` | total / non-embedding params | 1.544 B / 1.310 B |
| `L, d, h_q, h_kv, d_h` | layers, hidden, q/kv heads, head dim | 28, 1536, 12, 2, 128 |
| `Φ` (`R_peak`, compute) | peak fp16 matmul rate (measured) | 38.1 TFLOP/s |
| `β` (`R_peak`, memory) | peak bandwidth (measured) | 372 GB/s |
| `P_cap` | enforced power cap | 149 W |
| `C = 2·P_ne` | dense FLOPs/token | 2.62 GFLOP |
| `W = 2·P_tot` | weight bytes streamed per decode step | 3.09 GB |
| `k_attn = 2·L·h_q·d_h` | attention FLOP slope (per token, per S) | 86 016 |

`C` excludes the `lm_head`: prefill runs with `logits_to_keep=1`, so it fires only
on the last position, not per token.

## 3. Decode — coupled `P(T)` (memory-bandwidth-bound)

**Throughput.** Each step streams all weights `W` once (reused across the batch)
plus per-sequence KV, and on WDDM carries a fixed kernel-launch overhead with no
CUDA-graph amortisation. The step time is **affine in batch**:

```
t_step(B) = t_fixed + β_m · B           t_fixed = W/(β·e) + launch overhead
T(B) = B / t_step(B)                     → saturates at 1/β_m as B → ∞
```

**Power.** Raising `B` fills more of each step with active streaming, so the
bandwidth utilisation `u = R/β` rises and saturates; by eq. (1):

```
P(B) = P_idle + A · (1 − e^{−B/B₀})       A = P_cap-region swing
```

**Eliminating `B`** (the knob) between the two laws gives the **power–throughput
characteristic** `P(T)` directly — a curve that rises from `P_idle` and saturates
at `P_idle + A` as `T → 1/β_m`. Because both component laws are measured, `P(T)`
is their composition; no extra fit is needed. This is the decode "buy throughput
with power" curve.

## 4. Prefill — decoupled `P(T)` (compute-bound)

**Throughput.** A length-`S` prefill costs `S·C` dense FLOPs plus
`2·L·h_q·d_h·S²` causal-attention FLOPs. Dividing by the achievable compute rate
`Φ·u` (u = MFU) and by `S`:

```
            Φ · MFU
T(S) = ───────────────────        equivalently   1/T = a + b·S
        C + k_attn · S            a = C/(Φ·MFU),  b = k_attn/(Φ·MFU)
```

Throughput **falls** as `S` grows (the per-token cost `c = C + k_attn·S` grows).

**Power.** Here is the crux: the bottleneck **rate** is `R = T · c = Φ · MFU`,
which is **independent of `S`** — prefill always runs the matmul units at the
compute roof. By eq. (1), `u ≈ MFU` is fixed, so

```
P ≈ P_static + (P_cap − P_static)·MFU ≈ P_cap      (constant, ⊥ throughput)
```

Power is **pinned** while throughput sweeps a wide range purely through `c(S)`.
(At very small `S` the sequence cannot fill the SMs, so `R < Φ·MFU`, `u < 1`, and
power dips — the *occupancy ramp*. Once `S ≳ 256` the chip is full and power
locks to the cap.)

## 5. Energy per token

`E = P / T` (joules/token) follows immediately:

- **decode** `E(B) = P(B)·t_step(B)/B` — falls as batching amortises the fixed
  per-step weight read over more tokens (efficiency *improves* with throughput).
- **prefill** `E(S) ≈ P_cap·(C + k_attn·S)/(Φ·MFU)` — rises with `S` because each
  token costs more attention FLOPs at fixed power.

Prefill's `E` is far lower than decode's because prefill does useful work on
*every* token of a long sequence per weight-load, whereas decode reloads all
weights to emit only `B` tokens.

## 6. Why `P ∝ u`: the DVFS power law (context)

Eq. (1) is linear in `u` only at fixed clock. The deeper law is dynamic CMOS
power `P = P_static + α·C_load·V²·f`. Over the usable range `V` rises ~linearly
with clock `f`, so `P ≈ P_static + k·f^γ`, `γ ≈ 2–3`. For a compute-bound load
`T ∝ f`, giving the convex `P ≈ P_static + k′·T^γ` ("≈cubic") law. **Observing it
needs clock-locking**, which requires admin rights unavailable on this Windows
host (clock and power-limit control both return *Insufficient Permissions*). At
the fixed clock ceiling we instead measure the operating-point loci of §3–4. The
cubic law is the natural follow-up on a clock-controllable host.

---

## 7. Validation — measured vs model

Numbers from `results/fit_summary.json`; figures `figures/step3_*.png`.

### 7.1 Decode `P(T)` — `figures/step3_decode_model.png`

The composed `P(T)` curve tracks the measured points with **MAPE 3.5 %,
R² = 0.982**:

| quantity | value | meaning |
|---|---|---|
| `t_fixed` | 28.3 ms | fixed per-step cost (≈8.3 ms ideal weight-stream + ~20 ms launch overhead) |
| `β_m` | 0.65 ms/seq | marginal per-sequence cost |
| throughput asymptote `1/β_m` | 1.53 k tok/s | hard ceiling on this card |
| `P_idle → P_asymptote` | 54 → 139 W | power floor to near-cap |
| `B₀` | ~13 | batch scale on which power fills |

Power rises monotonically with throughput and saturates just under the cap —
exactly the coupled behaviour eqs. (1)+(2) predict for a memory-bound phase.

### 7.2 Prefill `P(T)` — `figures/step3_prefill_model.png`

In the compute-bound regime (`S ≥ 256`) power is **constant at 142 W with only
1.5 % coefficient of variation**, while throughput spans **3.7 → 8.7 k tok/s (a
2.4× range)** — power is **decoupled** from throughput, as eq. (1) with fixed
`u ≈ MFU` predicts. Supporting fits:

| quantity | value | meaning |
|---|---|---|
| implied MFU | 78 % | dense matmuls reach 78 % of the 38.1 TFLOP/s peak |
| compute-bound power | 142 W (±1.5 %) | pinned regardless of throughput |
| attention doubles cost at | S ≈ 1948 | where `k_attn·S = C` |

The throughput law `1/T = a + b·S` itself fits the post-peak branch at
**R² = 0.996, MAPE 1.2 %** (it is what sets `T` at fixed power); the sub-256
occupancy ramp is correctly outside it.

### 7.3 Summary — the power↔throughput relationship

| | prefill | decode |
|---|---|---|
| bottleneck | compute (rate pinned at roof) | memory bandwidth (rate ramps with batch) |
| `P(T)` shape | **flat at ~142 W** (decoupled) | **rising→saturating** 54→139 W (coupled) |
| how to get more tok/s | shorten sequence (less attention) — power unchanged | raise batch — costs more power |
| best energy efficiency | 26–60 tok/J | 0.4–5.6 tok/J (~11× worse) |

`figures/step4_combined_power_vs_throughput.png` overlays both on one throughput
axis: decode is the rising curve at low throughput, prefill the flat high-power
band at high throughput — two qualitatively different power↔throughput laws from
the same chip, exactly as the model predicts.
