"""The workload portfolio for validating the P<->T power-cap model across workload TYPES.

Design axis: each workload fixes (seq_len/ctx, batch) and sweeps ONLY the power cap, tracing one
single-valued P<->T curve per phase. The set is chosen so the DECODE bandwidth ceiling
    T_max = B * BW_eff / (weight_bytes + B*C*kv_bytes_per_token)
walks ~2 orders of magnitude (≈20 -> ≈1800 tok/s) by co-varying context length C and MHA-vs-GQA
KV size, while prefill compute intensity is swept light->heavy. Re-running the theory-vs-measured
fit on every workload then validates the model's generality; matching all 8 fitted decode plateaus
to the formula above simultaneously validates the weight term, the KV term, and the 1/C law.

Structure of one workload:
  id, name_en, application, model_id,
  prefill_seq_len, prefill_batch   # prefill point (compute-bound); batch chosen so cap bites (>=250W)
  decode_ctx, decode_batch         # decode point (memory-bound); the DEFINING config of the type

Notes on the eight:
  chat-phi3        the existing validated baseline (weight~=KV crossover, ceiling ~721 measured)
  rag-phi3 / code-phi3   CONTROLLED PAIR: identical decode B*C=32768 -> identical D_mem/T_mem,
                         batch 32 vs 16 -> plateau must be exactly 2:1 (isolates the ∝B factor)
  longform-phi3    deepest MHA KV-dominance at Phi-3's 4k ceiling -> lowest single-model plateau
  summarize-qwen7b only a 32k-context GQA model can hold this; the ~20 tok/s LOW anchor
  translate-qwen3b / fastchat-qwen15b   GQA, weight-dominant -> the HIGH plateau anchors (~1000/1800)
  classify-qwen7b  compute-bound PREFILL extreme (biggest model, heavy prompt, vestigial decode)
"""

PORTFOLIO = [
    dict(id="chat-phi3", name_en="Chat (short-context assistant)", application="chat",
         model_id="microsoft/Phi-3-mini-4k-instruct",
         prefill_seq_len=512, prefill_batch=8, decode_ctx=256, decode_batch=64),

    dict(id="rag-phi3", name_en="RAG / ClosedQA (long retrieved prompt)", application="RAG",
         model_id="microsoft/Phi-3-mini-4k-instruct",
         prefill_seq_len=4096, prefill_batch=2, decode_ctx=1024, decode_batch=32),

    dict(id="code-phi3", name_en="Code completion (interactive, long file)", application="code",
         model_id="microsoft/Phi-3-mini-4k-instruct",
         prefill_seq_len=2048, prefill_batch=4, decode_ctx=2048, decode_batch=16),

    dict(id="longform-phi3", name_en="Long-form / creative generation", application="long-gen",
         model_id="microsoft/Phi-3-mini-4k-instruct",
         prefill_seq_len=256, prefill_batch=16, decode_ctx=4096, decode_batch=8),

    dict(id="summarize-qwen7b", name_en="Summarization (offline 32k document)", application="summarization",
         model_id="Qwen/Qwen2.5-7B-Instruct",
         prefill_seq_len=4096, prefill_batch=2, decode_ctx=32768, decode_batch=4),

    dict(id="translate-qwen3b", name_en="Translation (balanced P:D)", application="translation",
         model_id="Qwen/Qwen2.5-3B-Instruct",
         prefill_seq_len=512, prefill_batch=8, decode_ctx=512, decode_batch=64),

    dict(id="fastchat-qwen15b", name_en="Lightweight chat (small GQA model)", application="chat",
         model_id="Qwen/Qwen2.5-1.5B-Instruct",
         prefill_seq_len=512, prefill_batch=16, decode_ctx=512, decode_batch=64),

    dict(id="classify-qwen7b", name_en="Classification / extraction (prefill-only)", application="classification",
         model_id="Qwen/Qwen2.5-7B-Instruct",
         prefill_seq_len=2048, prefill_batch=4, decode_ctx=256, decode_batch=8),

    # ---- frontier (2025) additions: Qwen3-4B-Instruct-2507 (GQA 8 kv-heads, QK-Norm, 262k ctx).
    # 'reasoning' is the signature 2025 workload: short prompt, thousands of chain-of-thought
    # decode tokens accumulating a long KV -- decode-dominated with a mid-size effective context.
    dict(id="qwen3chat-4b", name_en="Modern chat (Qwen3-4B, 2025)", application="chat",
         model_id="Qwen/Qwen3-4B-Instruct-2507",
         prefill_seq_len=512, prefill_batch=8, decode_ctx=1024, decode_batch=32),

    dict(id="qwen3think-4b", name_en="Reasoning / long chain-of-thought (Qwen3-4B)", application="reasoning",
         model_id="Qwen/Qwen3-4B-Instruct-2507",
         prefill_seq_len=2048, prefill_batch=4, decode_ctx=8192, decode_batch=8),
]

# Power-cap sweep grid (W). Same grid the validated baseline used; run_portfolio filters to [min,max].
CAP_GRID = [100, 120, 140, 160, 180, 200, 220, 250]
