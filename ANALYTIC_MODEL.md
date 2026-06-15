# Analytic Model: token throughput ↔ GPU power for prefill and decode

This document derives, from first principles, how **token throughput** and **GPU
power** should behave in each phase of LLM inference, then compares the
predictions to the measurements (Steps 1–2). The fitted numbers and the overlay
figures are produced by `code/analyze.py --step 3`
(`results/fit_summary.json`, `figures/step3_*.png`).

All machine constants below are **measured on this card** (Step 0,
`results/model_info.json`), not taken from a datasheet.

## 1. Notation

| symbol | meaning | value (this study) |
|---|---|---|
| `P_tot` | total parameters | 1.544 B |
| `P_ne` | non-embedding parameters | 1.310 B |
| `L` | transformer layers | 28 |
| `d` | hidden size | 1536 |
| `h_q`, `h_kv` | query / KV heads (GQA) | 12 / 2 |
| `d_h` | head dim | 128 |
| `b` | bytes per weight (fp16) | 2 |
| `Φ` | peak fp16 matmul rate (measured) | 38.1 TFLOP/s |
| `β` | peak memory bandwidth (measured) | 372 GB/s |
| `P_cap` | enforced power cap | 149 W |

Two derived per-token constants:

- **Dense compute per token** `C = 2·P_ne ≈ 2.62 GFLOP/token`
  (every non-embedding weight does one multiply-add = 2 FLOPs per token). The
  `lm_head` output projection is excluded — prefill runs with `logits_to_keep=1`,
  so it fires only on the *last* position, not per token, making this the correct
  per-token cost for the measured workload.
- **Weights to stream per decode step** `W = b·P_tot ≈ 3.09 GB`
  (each weight is read once per step, reused across the whole batch).
- **KV bytes per token** `κ = 2·L·h_kv·d_h·b ≈ 28.7 KB` (K and V, all layers).

## 2. The roofline that separates the two phases

A kernel achieves `min(Φ, β·I)` FLOP/s, where `I` is **arithmetic intensity**
(FLOP per byte of memory traffic). The **ridge** is at `I* = Φ/β ≈ 102 FLOP/byte`.

- **Prefill** reuses each weight across all `S` tokens of the sequence, so
  `I ≈ S` (exactly `S·P_ne/P_tot ≈ 0.85·S`). Even `S = 128` sits far right of the
  ridge → **compute-bound**.
- **Decode** reuses each weight across only the `B` tokens in the batch, so
  `I ≈ B` (same ~0.85 factor). For `B < 102` it sits left of the ridge →
  **memory-bandwidth-bound**; only at very large batch does it approach the
  compute roof.

This single picture predicts everything below: prefill power is pinned by
compute, decode power and throughput climb with batch toward the bandwidth (and
power) ceiling.

## 3. Prefill: compute-bound throughput

Total FLOPs to prefill a length-`S` sequence (batch 1):

```
FLOPs(S) = S·C            (dense matmuls)
         + 2·L·h_q·d_h·S²  (causal attention QKᵀ + AV, ∑ over the triangle)
```

Dividing by the achievable compute rate `Φ·u` (`u` = MFU, the achieved fraction
of peak) and by `S` gives throughput:

```
            Φ·u
tput(S) = ─────────────────         with   k_attn = 2·L·h_q·d_h
          C + k_attn·S
```

Equivalently **`1/tput = a + b·S`** is linear in `S`, with
`a = C/(Φ·u)` and `b = k_attn/(Φ·u)`. We fit `a, b` by least squares; the
intercept gives the implied **MFU**, and the ratio `b/a` gives the sequence
length at which attention doubles the per-token cost.

**Power.** Because prefill saturates compute at the clock ceiling, dynamic power
is near-constant: `P ≈ P_cap` for all but the smallest loads. The
throughput↔power relationship is therefore a near-**vertical** line at `P_cap` —
throughput varies (with `S`) while power barely moves. *Power is not the free
variable here; energy per token is* (`E/token = P/tput`, which rises with `S`).

## 4. Decode: memory-bandwidth-bound throughput

Each decode step reads all weights once (`W` bytes) plus every sequence's KV
cache (`B·ℓ·κ` bytes for context length `ℓ`), and does `B·C` FLOPs. Step time is
the larger of the memory and compute times:

```
t_step(B) = max(  (W + B·ℓ·κ) / (β·e) ,  B·C / (Φ·u)  )
                   └── memory ──┘          └─ compute ─┘
tput(B) = B / t_step(B)             (e = achieved bandwidth fraction)
```

- **Low batch** (`B ≪ I*`): memory term dominates and `W ≫ B·ℓ·κ`, so
  `t_step ≈ W/(β·e)` is *constant* → **`tput ≈ B·β·e/W` grows linearly in B**.
  The weights are re-read in full to emit only `B` tokens — the textbook
  memory-bandwidth bottleneck.
- **High batch** (`B → I*`): the compute term overtakes; throughput bends below
  the linear trend toward the `Φ·u/C` compute roof. The crossover batch is
  `≈ I* = Φ/β ≈ 102` (modulated by `e`, `u`, and the KV term).

**Power.** At `B = 1` a single latency-bound stream still boosts the clock high
(elevated idle-ish power); as `B` grows, more SMs do useful work each step and
power rises monotonically toward `P_cap`. We model it as a saturating approach
(§6).

## 5. The DVFS power law (why "throughput ∝ power" is really "∝ frequency")

Dynamic CMOS power is `P_dyn = α·C_load·V²·f`. Over a GPU's usable range supply
voltage `V` rises roughly linearly with clock `f`, so

```
P(f) ≈ P_static + k·f·V²(f) ≈ P_static + k'·f^γ,   γ ≈ 2–3
```

For a **compute-bound** workload throughput is `∝ f`, so eliminating `f` gives a
**convex, super-linear** throughput↔power curve `P ≈ P_static + k''·tput^γ` — the
"approximately cubic" law. **This law lives in the frequency domain.** Observing
it requires *locking the SM clock* across a range, which needs admin privileges
that are unavailable on this Windows host (clock and power-limit control both
return *Insufficient Permissions*). The GPU here runs at its clock ceiling under
load, so the load sweeps trace the **operating points at fixed near-max `f`**,
not the DVFS curve. We therefore present the DVFS law as theory and validate the
*roofline* (throughput) and *utilisation–power saturation* laws, which the
no-admin load sweeps can measure directly.

## 6. Power vs utilisation (the part we CAN measure without DVFS)

At fixed clock, dynamic power scales with the fraction of the chip switching.
For the decode batch sweep, that fraction grows with `B` and saturates as the
GPU fills, so:

```
P(B) ≈ P_idle + A·(1 − e^(−B/B₀))      → asymptote P_idle + A ≲ P_cap
```

`P_idle` is the active-but-near-empty draw, `A` the dynamic swing, `B₀` the batch
scale over which the SMs fill. We fit `P_idle, A, B₀` (grid on `B₀`, linear LS on
the rest).

---

## 7. Validation — measured vs model

Numbers from `results/fit_summary.json`; overlay figures in `figures/step3_*.png`.

### 7.1 Prefill — `figures/step3_prefill_model.png`

The compute+attention model `tput(S) = 1/(a + b·S)` fits the **compute-bound
branch (S ≥ 512)** almost exactly:

| quantity | value | meaning |
|---|---|---|
| implied **MFU** | **78 %** | dense matmuls reach 78 % of the measured 38.1 TFLOP/s peak |
| compute roof (attn→0) | 11.3 k tok/s | `Φ·MFU/C`; the S→0 ceiling of the branch |
| attention doubles cost at | **S ≈ 1948** | where `b·S = a` (attention = dense per-token cost) |
| effective attention slope | 15.6× ideal | the O(S²) coefficient is ~16× the bare FLOP count |
| **MAPE / R²** | **1.2 % / 0.996** | |

Two findings worth flagging. (1) The **rising branch (S < 512)** is *occupancy-
limited* — too few tokens to fill 30 SMs — and is correctly outside the model
(it is excluded from the fit and shown hollow). (2) The attention slope being
~16× its ideal FLOP count is internally consistent with Step 0's discovery that
**no flash/mem-efficient SDPA kernel exists for this sm_120 build**: attention
runs the O(S²)-memory math path, so each attention "FLOP" is far more expensive
than a peak matmul FLOP. The functional form is still exactly `1/(a+b·S)`, so the
fit is excellent — the inefficiency is absorbed into `b`.

### 7.2 Decode — `figures/step3_decode_model.png`

The pure weight-streaming roofline **over-predicts by ~3.4×**: it assumes step
time = `W/β` (read all weights once) and nothing else. The measured step time is
**closely affine in batch**, `t_step = t_fixed + β·B` (a linear approximation to
the curved `max(memory, compute)` law of §4 — the residual is the U-shaped
curvature that shows up as the 9.2 % MAPE, vs prefill's 1.2 %):

| quantity | value | meaning |
|---|---|---|
| `t_fixed` | **28.3 ms** | fixed per-step cost (weights + launch overhead) |
| ideal weight-stream | 8.3 ms | `W/β_peak` if bandwidth were 100 % |
| launch overhead | **~20 ms** | the gap — WDDM has no CUDA-graph amortisation |
| fixed-cost BW efficiency | 29 % | decode gemv hits only ~29 % of peak bandwidth |
| `β` (marginal) | 0.65 ms/seq | KV read + compute per extra sequence |
| throughput asymptote `1/β` | **1.53 k tok/s** | hard ceiling on this card/setup |
| half-max batch `t_fixed/β` | 43 | where throughput reaches ½ the asymptote |
| **MAPE / R²** | **9.2 % / 0.981** | |

This is the decode story made quantitative: each step re-reads all 3.09 GB of
weights to emit only `B` tokens, and on Windows/WDDM a large fixed kernel-launch
overhead sits on top, so throughput saturates near **1.5 k tok/s** — far below
both the bandwidth roofline and the compute ridge (`B* ≈ 102`), which is never
reached because VRAM runs out first (the spill wall at `B ≈ 56`).

### 7.3 Power vs utilisation — `figures/step3_power_model.png`

The saturation law `P(B) = P_idle + A·(1 − e^(−B/B₀))` fits the decode power
sweep with **R² = 0.982**: `P_idle ≈ 54 W`, swing `A ≈ 85 W`, scale
`B₀ ≈ 13`, asymptote **≈ 139 W** (just under the 149 W cap). Power fills the
chip on a batch scale of ~13 and then flattens — exactly the utilisation-driven
behaviour §6 predicts. (The `B = 1` point sits a little high, the latency-bound
clock-boost effect.)

### 7.4 The throughput↔power relationship, summarised

| phase | bound by | power behaviour | throughput↔power shape |
|---|---|---|---|
| **prefill** | compute (clock at ceiling) | pinned at ~140 W (cap) | near-**vertical** — throughput varies (attention), power fixed |
| **decode** | memory bandwidth + launch overhead | rises 54→139 W with batch | rising **diagonal** — both climb with batch toward the ceiling |

The "approximately cubic" `P ∝ tput^γ` DVFS law (§5) is **not** observed here
because it requires sweeping the clock, which needs unavailable admin rights; at
the fixed clock ceiling the two phases instead trace the vertical (prefill) and
diagonal (decode) operating-point loci above. Validating the cubic law directly
would need clock-lock permissions (a Linux host or an elevated session) — it is
the natural next experiment.
