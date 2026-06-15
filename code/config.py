"""Central configuration for the prefill/decode power-characterisation study.

Everything that defines *what* we measure lives here so the rest of the code is
a pure function of this file: the model, the numeric format, the sweep grids and
the measurement timing. Change a value here and re-run -- nothing else needs to
be touched.
"""
from __future__ import annotations
import os
import torch

# ---------------------------------------------------------------------------
# Model under test (the "given LLM inference model + parameter configuration")
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
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
# PREFILL sweep: one full forward over (batch x seq_len) tokens, no KV cache.
# Compute-bound. We sweep the *offered load* (sequence length at batch=1) so the
# x-axis is a clean monotone "offered tokens". logits_to_keep=1 keeps only the
# last position's logits (real prefill only needs the first output token).
#
# Ceiling note: this Blackwell sm_120 + torch build has NO flash / mem-efficient
# SDPA kernel ("No available kernel"), so attention runs the math path with
# O(S^2) activation memory. Peak memory hits the 8 GB wall at S ~= 5k (S=4096 ->
# 5.1 GB fits; S=8192 -> 11 GB spills to host over WDDM and collapses). We keep
# the sweep inside the regime that fits in VRAM; the wall itself is documented.
# ---------------------------------------------------------------------------
PREFILL_GRID = [
    (1, 64), (1, 96), (1, 128), (1, 192), (1, 256), (1, 384), (1, 512),
    (1, 640), (1, 768), (1, 1024), (1, 1280), (1, 1536), (1, 2048),
    (1, 2560), (1, 3072), (1, 3584), (1, 4096),
]

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
DECODE_BATCHES = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64,
                  96, 160, 256]
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
