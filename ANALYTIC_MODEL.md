# Analytic Model: GPU power as a (single-valued) function of token throughput

We want `P(T)` — GPU power as a function of token throughput — for prefill and
decode. For that to even be a *function*, each throughput must map to **one**
power. That forces a **controlled experiment**, and dictates the whole model.

Machine constants are measured on this card (Step 0, `results/model_info.json`);
fitted parameters and overlays come from `analyze.py --step 3`
(`results/fit_summary.json`, `figures/step3_*.png`).

## 1. Why the experiment must be controlled (the fix)

Throughput `T` and power `P` are **both outputs** of an operating point. To get a
single-valued `P(T)` you must sweep **one** control variable `θ` such that `T(θ)`
is **monotone** — then eliminating `θ` gives a function.

A natural-looking choice fails: sweeping **sequence length** `S` for prefill makes
throughput **non-monotone** — it rises while the GPU fills, then *falls* as
attention's O(S²) cost grows. The same throughput then occurs at two different
`S` (a short prompt at low power, a long prompt at the power cap), so `P(T)`
**folds** — one `T`, two `P`. That is not a relationship; it is a measurement
error.

**The controlled knob is batch (concurrency) at a fixed sequence/context length.**
With the per-token cost held constant, throughput is monotone in batch, so `P(T)`
is single-valued. Both phases use it:

| phase | fixed | swept (θ) | why monotone |
|---|---|---|---|
| prefill | prompt length `S = 128` | batch `B = 1…16` | per-token cost fixed; more sequences ⇒ more tok/s until the compute roof |
| decode | context `ctx = 256` | batch `B = 1…48` | per-token cost fixed; more sequences ⇒ more tok/s until the memory ceiling |

(`S=128` is small enough that one sequence does not saturate the GPU, leaving room
for the sweep to climb from light load to the ceiling.)

## 2. The model

Two laws in the **batch domain**, then composed to eliminate `B`.

**(a) Forward/step time is affine in batch** — a fixed per-call cost (kernel
launch + weight/setup) plus a marginal per-unit cost:

```
t(B) = t_fixed + β·B
T(B) = n·B / t(B)        n = tokens added per unit batch  (S for prefill, 1 for decode)
     →  ceiling  T_max = n/β   as  B → ∞
```

**(b) Power saturates as concurrency fills the chip.** Dynamic power tracks the
fraction `u` of the GPU that is busy (`P = P_static + (P_cap−P_static)·u`), and
`u` rises and saturates with batch:

```
P(B) = P_idle + A·(1 − e^{−B/B₀})      → asymptote P_idle + A ≲ P_cap
```

**Compose** (a)+(b) over the batch grid → a single-valued, saturating **`P(T)`**:
power climbs from the light-load draw to ~the cap as throughput climbs to `T_max`.

## 3. What sets the ceiling `T_max` (the whole difference between the phases)

`T_max = n/β` is fixed by the **bottleneck resource**:

- **Prefill — compute roof.** Each token does `c = C + k_attn·S` FLOPs
  (`C = 2·P_ne` dense + attention). The roof is `T_max ≈ Φ·MFU / c`. With
  `Φ = 38.1 TFLOP/s`, `S = 128` (`c ≈ 2.63 GFLOP`), the ideal (MFU=1) roof is
  **14.5 k tok/s**; the sweep reaches **11.8 k** (MFU ≈ 74 %).
- **Decode — memory/overhead limit.** Each step streams all weights `W = 3.09 GB`
  once and carries a fixed launch overhead, so `t_fixed` dominates and
  `T_max = 1/β = ` **1.5 k tok/s** — an order of magnitude below prefill, and far
  below the bandwidth roofline (overhead-limited, no CUDA graphs on WDDM).

So both phases give the *same shape* of `P(T)` (rise → saturate at ~cap), but the
**throughput ceiling differs ~8–14×**. At the same near-cap power, **prefill
delivers ~14× the tokens/s of decode**, because a prefill token does useful
compute on weights shared across the whole sequence, whereas a decode step
reloads every weight to emit only `B` tokens.

## 4. Energy per token

`E = P/T` (J/token). Both phases get *more efficient* as throughput rises
(concurrency amortises the fixed per-call cost): prefill **43 → 74 tok/J**,
decode **0.4 → 5.6 tok/J**. Prefill is ~13× more efficient at best — same reason
as the throughput gap.

## 5. Two different knobs — and only one gives the ≈cubic law

A natural expectation is that prefill should show **`P ∝ T³`** (power growing
~cubically with throughput). That is true — but only for the *frequency* knob,
not the *concurrency* knob. The two experiments raise throughput by different
physical means:

| knob (how T rises) | mechanism | clock | `P(T)` |
|---|---|---|---|
| **frequency `f`** (DVFS, §below) | each core runs faster, `P_dyn = C·V²·f`, `V ∝ f` | rises | **`P ≈ P_static + k·T^γ`, γ ≈ 2–3** (≈cubic) |
| **concurrency `B`** (this study) | more cores active, clock pinned at the boost ceiling | ~constant | **`P` ≈ linear in T, then saturates at the cap** |

The cubic law lives in `f`: `T ∝ f` (compute-bound) and `P ≈ P_static + k·f^γ`
(`γ ≈ 2–3` because core voltage must rise with clock) ⇒ `P ∝ T^γ`. Our batch
sweep instead holds the clock fixed and fills idle units, so adding throughput
adds a *proportional* number of switching units → power rises ~linearly, then
hits the cap. The measured clock confirms this: across the prefill sweep the SM
clock sat at ~2700–2840 MHz and even *dropped* slightly as batch rose (2840 →
2699 MHz while throughput went 3.7 → 10.7 k tok/s), so the throughput gain was
pure occupancy, not frequency — the cubic law cannot appear.

**Measured (DVFS sweep, `code/measure_dvfs.py` run elevated, clock 600→2687 MHz,
`figures/step5_dvfs_cubic.png`):**

| workload | T vs clock | P vs throughput | range |
|---|---|---|---|
| **prefill** (S=512, B=4) | `T ∝ f^0.91` (≈linear, compute-bound) | **`P ≈ 31 + k·T^2.94`, R²=0.989** | 2.2→8.6 k tok/s, 32→125 W |
| **decode** (B=16) | `T ∝ f^0.50` (sub-linear, memory-bound) | T compressed → steep `T^4.5` | 0.24→0.48 k tok/s, 36→88 W |

So the **prefill power–throughput law is `P ∝ T^2.94` — the ≈cubic law, confirmed
directly** (γ ≈ 3, R² = 0.989). The mechanism is in the left panel: prefill
throughput scales with clock (`T ∝ f`), so the `V²·f` dynamic-power law in `f`
becomes a cubic in `T`. Decode throughput barely responds to clock (`T ∝ f^0.5`),
so raising the clock spends power for almost no throughput — the wrong knob for a
memory-bound phase (its large fitted exponent is an artefact of `T` being nearly
constant, not extra "cubicness").

This is the same prefill GPU as the batch sweep (§3), measured two ways: vary the
**clock** → power is ≈cubic in throughput; vary the **batch** at fixed clock →
power is ≈linear-then-saturating in throughput. Both are correct — they are
different knobs.

---

## 6. Validation — measured vs model

From `results/fit_summary.json`; overlays `figures/step3_*.png`.

### 6.1 Prefill `P(T)` — `figures/step3_prefill_model.png`
Single-valued; the composed model tracks the points with **MAPE 1.0 %,
R² = 0.991**.

| quantity | value |
|---|---|
| power range (measured) | 86 → 146 W (asymptote 145 W ≈ cap) |
| throughput ceiling `S/β` | 11.8 k tok/s |
| ideal compute roof `Φ/c` | 14.5 k tok/s (MFU ≈ 74 %) |
| affine time | `t_fixed = 13.6 ms`, `β = 10.8 ms/batch` |

### 6.2 Decode `P(T)` — `figures/step3_decode_model.png`
Single-valued; **MAPE 3.5 %, R² = 0.982**.

| quantity | value |
|---|---|
| power | `P_idle 54 → 139 W` asymptote (≈ cap) |
| throughput ceiling `1/β` | 1.5 k tok/s |
| affine step time | `t_fixed = 28.3 ms`, `β = 0.65 ms/seq` |

### 6.3 Summary

| | prefill | decode |
|---|---|---|
| controlled sweep | fixed S=128, batch 1–16 | fixed ctx=256, batch 1–48 |
| `P(T)` shape | rise → saturate at ~145 W | rise → saturate at ~139 W |
| bottleneck / ceiling | compute roof, **11.8 k tok/s** | memory+overhead, **1.5 k tok/s** |
| best energy efficiency | 74 tok/J | 5.6 tok/J |

`figures/step4_combined_power_vs_throughput.png` overlays both: two
single-valued, rising `P(T)` curves on one throughput axis — identical in shape,
separated by the ~14× throughput ceiling that compute-bound vs memory-bound work
implies.
