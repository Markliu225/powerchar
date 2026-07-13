"""THE workload-classification figure: use-case classes sorted by aggregate prefill:decode.

One panel: each class as a horizontal bar (aggregate P:D, log x), grouped into three bands
(decode-heavy / balanced / prefill-heavy). Right margin: the measured powerchar workload each
class maps onto (with the scale/accounting caveat where the mapping is not literal — details in
rack_power_capping/v100/WORKLOADS.zh.md §2).

Taxonomy (VERIFIED): InstructGPT Table 1 (Ouyang et al. 2022) defines 10 prompt classes —
Generation, Open QA, Brainstorming, Chat, Rewrite, Summarization, Classification, Other,
Closed QA, Extract. Coverage here: 8 of them carry data (Rewrite has no clean public set;
Other is a catch-all), + 'General QA' which is Dolly's OWN extra free-form category
(7 of Dolly's 8 labels correspond to InstructGPT; General QA was added by Databricks),
+ a production Code-completion class from the Azure code trace.
Data: Dolly-15k (Conover et al. 2023) · Azure LLM inference traces conv+code
(Patel et al., Splitwise, ISCA 2024) · BurstGPT (Wang et al. 2024).
Full citations: REFERENCES.zh.md.   python3 plot.py -> fig_workload_pd.png
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = {"Generation 创作生成": "Generation", "General QA 常识问答": "General QA †",
        "Brainstorming 头脑风暴": "Brainstorming", "Open QA 开放问答": "Open QA",
        "Classification 分类": "Classification", "Summarization 摘要": "Summarization",
        "Extract 信息抽取": "Extract", "Chat 多轮对话": "Chat (dialogue)",
        "Closed QA 闭卷问答": "Closed QA", "Code 代码补全": "Code (completion) ‡"}
# P:D bands (aggregate ratio r = sum P / sum D): the classification the rack planner consumes
BANDS = [("decode-heavy", 0.0, 0.5, "#d62728"),
         ("balanced", 0.5, 2.0, "#7f7f7f"),
         ("prefill-heavy", 2.0, np.inf, "#1f77b4")]
# which measured powerchar workload each class maps onto (verification of the correspondence;
# caveats — KV-reuse accounting for chat, production-scale prompts, 2025-style code gens —
# are spelled out in rack_power_capping/v100/WORKLOADS.zh.md §2)
MAPPED = {"Generation": "↔ long-gen/CoT class (longform-phi3, qwen3think-4b)",
          "General QA †": "band ↔ long-gen/CoT class",
          "Brainstorming": "band ↔ long-gen/CoT class",
          "Open QA": "band ↔ long-gen/CoT class",
          "Classification": "band ↔ balanced class (translate ≈ Rewrite)",
          "Summarization": "↔ summarize-qwen7b (32k prod scale → 30:1)",
          "Extract": "↔ classify-qwen7b (prod scale → 100:1)",
          "Chat (dialogue)": "↔ chat class ×3 (KV-reuse accounting → 1:2)",
          "Closed QA": "↔ rag-phi3 (prod prompts ~4k → 10:1)",
          "Code (completion) ‡": "↔ code-phi3 ('25-style longer gens → 10:1)"}
REFS = ("Taxonomy: InstructGPT Table 1 — Ouyang et al. 2022, arXiv:2203.02155 (10 classes; Rewrite/Other uncovered)  ·  "
        "† Dolly's own extra category (7 of its 8 labels are InstructGPT's)  ·  ‡ added production class\n"
        "Data: Dolly-15k — Conover et al. 2023  ·  Azure LLM inference traces (conv+code) — "
        "Patel et al., Splitwise, ISCA 2024  ·  BurstGPT — Wang et al. 2024, arXiv:2401.17644")

rows = list(csv.DictReader(open(os.path.join(HERE, "workload_ratios.csv"))))
g = lambda r, k: float(r[k])
band_of = lambda r: next(b for b in BANDS if b[1] <= g(r, "ratio_agg") < b[2])
ratio_str = lambda x: f"{x:.1f}:1" if x >= 1 else f"1:{1/x:.0f}"

fig, a = plt.subplots(figsize=(11.5, 7.2))
sr = sorted(rows, key=lambda r: g(r, "ratio_agg"))
y = np.arange(len(sr))
vals = [g(r, "ratio_agg") for r in sr]
cols = [band_of(r)[3] for r in sr]

X_LO, X_HI = 0.03, 30000                       # extra right room for the mapping labels
for name, lo, hi, c in BANDS:                  # band shading + header with class count
    a.axvspan(max(lo, X_LO), min(hi, X_HI), color=c, alpha=.06)
    n_in = sum(1 for v in vals if lo <= v < hi)
    a.text(np.sqrt(max(lo, X_LO) * min(hi, 200)), len(sr) - 0.15, f"{name}\n({n_in})",
           ha="center", fontsize=9, color=c, weight="bold")

a.barh(y, vals, color=cols, alpha=.88, log=True, height=0.62)
a.axvline(1, color="k", ls="--", lw=1)
for i, r in enumerate(sr):
    v = g(r, "ratio_agg")
    a.text(v * 1.25, i, ratio_str(v), va="center", ha="left", fontsize=9, color=cols[i],
           weight="bold")
    a.text(0.985, i, MAPPED[NAME[r["klass"]]], va="center", ha="right", fontsize=7.8,
           color="#52514e", transform=a.get_yaxis_transform())
a.set_yticks(y)
a.set_yticklabels([f"{NAME[r['klass']]}{' *' if r['kind'] == 'prod-trace' else ''}" for r in sr],
                  fontsize=10)
a.set_xlabel("aggregate prefill : decode  =  Σprefill / Σdecode   (log)")
a.set_xlim(X_LO, X_HI)
a.set_ylim(-0.6, len(sr) + 0.9)
a.grid(alpha=.3, axis="x", which="both")
a.set_title("LLM inference workload classification by prefill:decode ratio\n"
            "10 use-case classes in 3 bands   (* = production trace, others Dolly-15k)   ·   "
            "gray = the measured powerchar workload each class maps onto", fontsize=11.5)

fig.text(0.5, 0.012, REFS, ha="center", fontsize=7.4, color="#52514e")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(HERE, "fig_workload_pd.png"), dpi=130, bbox_inches="tight")
print("wrote fig_workload_pd.png")
