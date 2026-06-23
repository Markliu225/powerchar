"""Step 0 -- characterise the *given* model and *this* GPU.

Extracts the model's architecture (the parameter configuration that drives the
analytic model) and MEASURES the two machine constants the roofline needs --
peak fp16 matmul throughput and peak memory bandwidth -- on this exact card,
rather than trusting datasheet numbers. Writes results/model_info.json and a
roofline figure that places the prefill and decode regimes on the chip.
"""
from __future__ import annotations
import json
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoConfig

import config as C
from power_sampler import PowerSampler


def measure_peak_matmul_tflops(n: int = 8192, secs: float = 3.0) -> float:
    """Sustained square fp16 matmul -> peak achievable TFLOP/s on this card."""
    a = torch.randn(n, n, device=C.DEVICE, dtype=C.DTYPE)
    b = torch.randn(n, n, device=C.DEVICE, dtype=C.DTYPE)
    for _ in range(5):                       # warmup
        c = a @ b
    torch.cuda.synchronize()
    iters = 0
    t0 = time.perf_counter()
    t_end = t0 + secs
    while time.perf_counter() < t_end:
        c = a @ b
        torch.cuda.synchronize()
        iters += 1
    dt = time.perf_counter() - t0
    flops = iters * 2.0 * n ** 3
    del a, b, c
    torch.cuda.empty_cache()
    return flops / dt / 1e12


def measure_peak_bw_gbs(n: int = 1 << 26, secs: float = 3.0) -> float:
    """Memory-bound elementwise c = a + b on big fp16 vectors.

    Reads 2N + writes N = 3N bytes per pass -> effective DRAM bandwidth.
    """
    a = torch.randn(n, device=C.DEVICE, dtype=C.DTYPE)
    b = torch.randn(n, device=C.DEVICE, dtype=C.DTYPE)
    c = torch.empty_like(a)
    for _ in range(5):
        torch.add(a, b, out=c)
    torch.cuda.synchronize()
    iters = 0
    t0 = time.perf_counter()
    t_end = t0 + secs
    while time.perf_counter() < t_end:
        torch.add(a, b, out=c)
        torch.cuda.synchronize()
        iters += 1
    dt = time.perf_counter() - t0
    bytes_moved = iters * 3 * n * a.element_size()
    del a, b, c
    torch.cuda.empty_cache()
    return bytes_moved / dt / 1e9


def arch_from_model(model, cfg) -> dict:
    total = sum(p.numel() for p in model.parameters())
    embed = model.get_input_embeddings().weight.numel()
    tied = getattr(cfg, "tie_word_embeddings", False)
    # With tied embeddings the lm_head shares the embedding matrix, so the only
    # "extra" non-layer params are the embedding table (a lookup, no per-token
    # matmul FLOPs). Non-embedding params are what every token streams through.
    non_embed = total - embed * (1 if tied else 2)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return {
        "model_id": C.MODEL_ID,
        "dtype": str(C.DTYPE).replace("torch.", ""),
        "bytes_per_param": torch.finfo(C.DTYPE).bits // 8,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_kv_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "head_dim": head_dim,
        "intermediate_size": getattr(cfg, "intermediate_size", getattr(cfg, "ffn_dim", 0)),
        "vocab_size": cfg.vocab_size,
        "tie_word_embeddings": bool(tied),
        "total_params": int(total),
        "embedding_params": int(embed),
        "non_embedding_params": int(non_embed),
    }


def derive_constants(a: dict) -> dict:
    bpp = a["bytes_per_param"]
    P_dense = a["non_embedding_params"]
    # Dense forward FLOPs per token (2 FLOPs per MAC). Attention scores excluded
    # here -- they are an S-dependent term added explicitly in the prefill model.
    dense_flops_per_token = 2.0 * P_dense
    # Decode reads every weight once per step (each weight used in one matmul).
    weight_bytes = bpp * a["total_params"]
    # KV cache: per layer, per token, K and V each = num_kv_heads * head_dim.
    kv_bytes_per_token = (2 * a["num_layers"] * a["num_kv_heads"]
                          * a["head_dim"] * bpp)
    # Attention slope: causal prefill of length S costs, per query token,
    # ~2*n_layers*n_q_heads*head_dim*S FLOPs (QK^T + AV), averaged S/2 over the
    # block -> coefficient on S in flops/token.
    attn_flops_per_token_per_S = (2.0 * a["num_layers"]
                                  * a["num_attention_heads"] * a["head_dim"])
    return {
        "dense_flops_per_token": dense_flops_per_token,
        "weight_bytes": weight_bytes,
        "kv_bytes_per_token": kv_bytes_per_token,
        "attn_flops_per_token_per_S": attn_flops_per_token_per_S,
    }


def plot_roofline(info: dict, path: str):
    peak = info["peak_matmul_flops"]          # FLOP/s
    bw = info["peak_bw_bytes_s"]              # byte/s
    ridge = peak / bw                         # FLOP/byte
    ai = np.logspace(-1, 3.5, 400)
    achievable = np.minimum(peak, bw * ai)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(ai, achievable / 1e12, "k-", lw=2, label="roofline")
    ax.axvline(ridge, color="gray", ls=":", label=f"ridge = {ridge:.0f} FLOP/byte")
    ax.axhline(peak / 1e12, color="C3", ls="--", alpha=.6)

    # Decode operating points: AI in FLOP/byte ~= batch (weights reused across
    # the batch, ~2 FLOP per 2 bytes per token). Memory-bound side of the ridge.
    for b in (1, 8, 32, 128, 256):
        ax.scatter([b], [min(peak, bw * b) / 1e12], color="C0", zorder=5)
    ax.scatter([], [], color="C0", label="decode (AI ≈ batch)")
    # Prefill: AI ~= seq_len (weights reused across the sequence). Compute-bound.
    for s in (64, 512, 4096):
        ax.scatter([s], [min(peak, bw * s) / 1e12], color="C1", marker="s", zorder=5)
    ax.scatter([], [], color="C1", marker="s", label="prefill (AI ≈ seq_len)")

    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("achievable fp16 throughput (TFLOP/s)")
    ax.set_title(f"Roofline -- {info['name']}\n"
                 f"peak {peak/1e12:.1f} TFLOP/s, BW {bw/1e9:.0f} GB/s, "
                 f"cap {info['power_cap_w']:.0f} W")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = AutoConfig.from_pretrained(C.MODEL_ID)
    print("loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        C.MODEL_ID, dtype=C.DTYPE, attn_implementation=C.ATTN_IMPL).to(C.DEVICE).eval()

    arch = arch_from_model(model, cfg)
    const = derive_constants(arch)

    sampler = PowerSampler(interval_s=C.SAMPLE_INTERVAL_S)
    print("measuring peak matmul TFLOP/s...", flush=True)
    peak_tflops = measure_peak_matmul_tflops()
    print(f"  peak = {peak_tflops:.1f} TFLOP/s", flush=True)
    print("measuring peak memory bandwidth...", flush=True)
    peak_bw = measure_peak_bw_gbs()
    print(f"  BW = {peak_bw:.0f} GB/s", flush=True)

    info = {
        "name": sampler.name,
        "power_cap_w": sampler.power_limit_w,
        "peak_matmul_tflops": peak_tflops,
        "peak_matmul_flops": peak_tflops * 1e12,
        "peak_bw_gbs": peak_bw,
        "peak_bw_bytes_s": peak_bw * 1e9,
        **arch,
        **const,
    }
    sampler.shutdown()

    out = os.path.join(C.RESULTS_DIR, "model_info.json")
    with open(out, "w") as f:
        json.dump(info, f, indent=2)
    print(f"wrote {out}")

    # Console summary
    print("\n=== model + machine ===")
    print(f"  model            {info['model_id']}")
    print(f"  params           {info['total_params']/1e9:.3f} B "
          f"(non-embed {info['non_embedding_params']/1e9:.3f} B)")
    print(f"  layers/d/heads   {info['num_layers']} / {info['hidden_size']} / "
          f"{info['num_attention_heads']}q:{info['num_kv_heads']}kv x {info['head_dim']}")
    print(f"  weight bytes     {info['weight_bytes']/1e9:.2f} GB (fp16)")
    print(f"  dense FLOPs/tok  {info['dense_flops_per_token']/1e9:.2f} GFLOP")
    print(f"  GPU              {info['name']}  cap {info['power_cap_w']:.0f} W")
    print(f"  peak compute     {info['peak_matmul_tflops']:.1f} TFLOP/s")
    print(f"  peak bandwidth   {info['peak_bw_gbs']:.0f} GB/s")
    print(f"  roofline ridge   {info['peak_matmul_flops']/info['peak_bw_bytes_s']:.0f} FLOP/byte")

    plot_roofline(info, os.path.join(C.FIGURES_DIR, "step0_roofline.png"))


if __name__ == "__main__":
    main()
