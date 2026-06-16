# Theoretical Model: GPU Power vs Token Throughput for Prefill and Decode

**Result, stated first:**

$$\boxed{\ P_{\text{prefill}} \;\approx\; P_0 + k_c\,T^{3}\quad(\text{cubic})\qquad\qquad P_{\text{decode}} \;\approx\; P_0 + k_m\,T\quad(\text{linear})\ }$$

This document derives both laws **from first principles only** — the Transformer cost structure, CMOS/DRAM power physics, and the roofline. **No measured data is used or fitted.** We work in the idealized regime where the GPU is free to scale its voltage and frequency to deliver throughput, with **no power cap and no thermal limit**. (Real, capped/throttled measurements are the *clipped shadow* of these laws — see §6.)

---

## 1. The control variable: frequency/voltage (DVFS), not batch

To make `P(T)` a clean single-valued law we raise throughput by running the **bottleneck unit faster** — i.e. by the DVFS knob (operating frequency `f` with its coupled voltage `V`). Each phase has a *different* bottleneck unit (compute vs memory), and those two units obey *different* power-vs-speed physics. That difference is the whole story.

## 2. Two power primitives (the physics)

**(a) Logic / compute dynamic power.** A switching CMOS array dissipates

$$P_{\text{logic}} = \alpha\,C\,V^{2}\,f .$$

For reliable switching the supply voltage must rise with frequency; in the active range (above the threshold floor) `V ≈ γ·f`, so

$$P_{\text{logic}} \;\propto\; V^{2} f \;\propto\; f^{2}\cdot f \;=\; f^{3}.$$

Equivalently, **energy per operation** `E_op ∝ C V² ∝ f²` — computing faster costs *quadratically* more energy per op.

**(b) Memory / data-movement power.** Moving one bit dissipates a roughly **fixed** energy `E_bit` (charging/discharging fixed wire and cell capacitances at the ~fixed memory-domain voltage; HBM does not aggressively voltage-scale with its data clock). So

$$P_{\text{mem}} = E_{\text{bit}}\cdot(\text{bit rate}) = E_{\text{bit}}\cdot \text{BW} \;\propto\; \text{BW}.$$

Memory power is **linear** in bandwidth, because energy-per-bit is constant (the familiar "~pJ/bit").

> **The asymmetry that creates everything:** compute energy/op *rises with speed* (`∝V²∝f²`); memory energy/bit *does not*. Cubic vs linear follows directly.

## 3. Phase bottlenecks (roofline)

Per token, prefill and decode do nearly identical FLOPs, but differ decisively in **weight reuse**, hence arithmetic intensity `I = FLOP/byte` relative to the ridge `I* = Φ/β_mem`:

- **Prefill — compute-bound** (`I ≫ I*`). Each weight tile, loaded once, is reused across all `B·S` sequence positions, so the kernel is arithmetic-limited. Throughput tracks the **compute rate**:
$$T_{\text{prefill}} \;\propto\; (\text{FLOP/s}) \;\propto\; f .$$
- **Decode — memory-bound** (`I ≪ I*`). Every step must re-stream *all* weights (no reuse) to emit only `B` tokens, so it is bandwidth-limited. Throughput tracks **memory bandwidth**:
$$T_{\text{decode}} \;\propto\; \text{BW}.$$

## 4. Combine → the two laws

**Prefill (compute-bound).** `T ∝ f ⇒ f ∝ T`. Power is dominated by the compute logic:

$$P_{\text{prefill}} \;\approx\; P_0 + P_{\text{logic}} \;\propto\; f^{3} \;\propto\; T^{3} \qquad\Longrightarrow\qquad \boxed{P_{\text{prefill}}(T) = P_0 + k_c\,T^{3}.}$$

**Decode (memory-bound).** `T ∝ BW`. Power is dominated by memory traffic:

$$P_{\text{decode}} \;\approx\; P_0 + P_{\text{mem}} \;\propto\; \text{BW} \;\propto\; T \qquad\Longrightarrow\qquad \boxed{P_{\text{decode}}(T) = P_0 + k_m\,T.}$$

`P_0` is the common static/leakage floor.

**One-line intuition:**
- *Prefill:* to emit tokens faster you must clock the **math units** faster, and faster logic needs higher voltage (`V∝f`), so `P = (∝f²)·(∝f) = f³ = T³`.
- *Decode:* to emit tokens faster you must **stream weights/KV** faster, and moving bits costs a fixed energy each, so `P = E_bit·(bytes/s) ∝ T`.

## 5. Energy-per-token corollary (the practical punchline)

With `E/token = P/T`:

| phase | `P(T)` | `E/token = P/T` | implication |
|---|---|---|---|
| **prefill** | `∝ T³` | **`∝ T²`** | running faster costs *quadratically* more energy/token ⇒ a strong incentive to run at a **lower clock**; the energy-optimal prefill point is at minimum frequency. |
| **decode** | `∝ T` | **`≈ constant`** | per-token energy is ~independent of clock ⇒ speed is "free" energy-wise; the only lever is **moving fewer bytes** (batching to amortize the weight stream, GQA/MQA, KV compression). |

This is why energy-aware serving slows down (down-clocks) the compute-bound prefill but attacks the memory traffic of decode instead.

## 6. Honest scope of the idealized law

1. **Voltage floor.** `V ∝ f` holds only in the upper range. Below a floor `V ≈ V_min`, logic power reverts to `∝ f` (linear), so the cubic flattens at low clocks. Full curve: `P ≈ P_0 + a·f + b·f³` — cubic-dominant only near the top.
2. **Power cap / thermal limit clip the cubic.** Under a hard cap, the prefill cubic cannot be traversed: power pins at the cap and the governor *lowers* the clock to stay there, so you observe **flat power**, not `T³`. **This is exactly why the theoretical model must set the measured (capped) data aside** — the cap masks the underlying cubic.
3. **Decode offsets.** Pure linearity assumes bandwidth-bound operation with constant energy/bit; launch overhead and the static floor `P_0` add an affine intercept but do not change the `∝T` slope.
4. **Knob, not batch.** These laws are for the **frequency/voltage** knob (run the bottleneck unit faster). The orthogonal *concurrency* (batch) knob raises throughput by filling idle units at fixed clock; it obeys a different (saturating) `P(T)` and is not the subject here.

## 7. Summary

| | prefill | decode |
|---|---|---|
| roofline regime | compute-bound (`I ≫ I*`) | memory-bound (`I ≪ I*`) |
| throughput scales with | core frequency, `T ∝ f` | memory bandwidth, `T ∝ BW` |
| dominant power | logic, `P ∝ V²f ∝ f³` (`V∝f`) | data movement, `P ∝ BW` (`E_bit` const) |
| **power law** | **`P = P_0 + k_c T³`** (cubic) | **`P = P_0 + k_m T`** (linear) |
| energy/token | `∝ T²` (rises fast) | `≈ const` |

The cubic and the linear law are two faces of one fact: **compute pays `V²` to go faster, memory pays nothing extra per bit.**
