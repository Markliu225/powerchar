"""Central configuration for the prefill/decode power-characterisation study.

Everything that defines *what* we measure lives here so the rest of the code is
a pure function of this file: the model, the numeric format, the sweep grids and
the measurement timing. Change a value here and re-run -- nothing else needs to
be touched.
"""
from __future__ import annotations
import os

# ---------------------------------------------------------------------------
# GPU pinning — the box now presents a single healthy GPU at index 0 (the other
# slots fault/disappear after a driver wedge; post-reboot only GPU0 enumerates and
# CUDA initialises there). Default to device 0; an explicit CUDA_VISIBLE_DEVICES
# still wins. Order by PCI bus so indices are deterministic.
# ---------------------------------------------------------------------------
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

# ---------------------------------------------------------------------------
# Model under test (the "given LLM inference model + parameter configuration")
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("POWERCHAR_MODEL", "microsoft/Phi-3-mini-4k-instruct")  # override via env for multi-model sweeps
DTYPE = torch.float16          # weights + activations
DEVICE = "cuda"
ATTN_IMPL = "sdpa"             # scaled-dot-product attention (PyTorch fused)

# ---------------------------------------------------------------------------
# Measurement timing (per sweep point)
# ---------------------------------------------------------------------------
WARMUP_S = 1.0                 # spin the workload before the timed window
SETTLE_S = 0.3                 # let power reach steady state after warmup
MEASURE_S = 4.0                # sustained timed window (power averaged over it)
SAMPLE_INTERVAL_S = 0.02       # NVML telemetry period -> 50 Hz

# ---------------------------------------------------------------------------
# PREFILL sweep -- CONTROLLED EXPERIMENT.
# We hold the prompt length S FIXED and sweep only the batch (concurrency). With
# the per-token cost fixed, throughput is a MONOTONE function of the single swept
# variable (batch), so the power-vs-throughput relation is single-valued -- no
# fold. (Sweeping S instead makes throughput non-monotone: it rises, then falls
# as attention's O(S^2) cost grows, so the same throughput maps to two different
# power points. That is why we fix S.) One full forward over (batch x S) tokens,
# use_cache=False, logits_to_keep=1.
#
# S=128 is chosen so a single sequence does NOT yet saturate the GPU (batch=1 is
# ~46% util), leaving room for the batch sweep to climb from light load to the
# compute/power ceiling. Longer prompts saturate at batch=1 and leave no range.
# ---------------------------------------------------------------------------
PREFILL_SEQ_LEN = 128
# Up to the compute-roof saturation (batch 16); throughput is strictly monotone
# over this range so P(T) is single-valued. Past ~16 throughput plateaus at the
# roof (~10.7k tok/s) and adds no range, so we stop there.
# V100 32 GB + 3.8B model: extend the batch range so the sweep climbs from light
# load all the way to the compute roof (saturation comes higher than on the 5060).
PREFILL_BATCHES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]

# ---------------------------------------------------------------------------
# DECODE sweep: single-token autoregressive steps reusing a KV cache, swept by
# batch size (concurrent sequences) at a fixed context length. Memory-bandwidth
# bound. Batch is the natural knob: it moves the operating point from a single
# latency-bound stream up to the bandwidth ceiling.
# ---------------------------------------------------------------------------
# Dense coverage of the clean memory-bound regime (b<=64), plus a few points
# past it: on this 8 GB card (~1.7 GB taken by the Windows desktop -> ~6.4 GB
# usable) weights + KV exhaust VRAM near b~80, beyond which WDDM spills to host
# over PCIe and throughput collapses. Those points are kept to document the wall
# but excluded from the bandwidth fit (analyze.py marks them automatically).
# Phi-3-mini uses full MHA (32 KV heads) -> a large KV cache, so even on this
# 32 GB V100 weights + KV exhaust VRAM in the b~50-70 range; OOM points are
# caught/skipped and mark the memory wall. Dense coverage of the clean
# memory-bound regime below that.
DECODE_BATCHES = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80]
DECODE_CTX = 256               # fixed KV context length per sequence
DECODE_SEED_CHUNK = 32         # seed the KV cache in chunks so high batch fits

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
