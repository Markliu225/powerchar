"""Visualise prefill:decode per LLM use-case class (InstructGPT taxonomy), from workload_ratios.csv.

  (a) (prefill, decode) plane, log-log: each class = a point (median) with an IQR box (p25-p75).
      P=D diagonal splits prefill-heavy (below) from decode-heavy (above).
  (b) aggregate P:D ratio per class, log axis, sorted decode-heavy -> prefill-heavy.
"""
from __future__ import annotations
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = {"Generation 创作生成": "Generation", "General QA 常识问答": "General QA",
        "Brainstorming 头脑风暴": "Brainstorming", "Open QA 开放问答": "Open QA",
        "Classification 分类": "Classification", "Summarization 摘要": "Summarization",
        "Extract 信息抽取": "Extract", "Chat 多轮对话": "Chat (dialogue)", "Closed QA 闭卷问答": "Closed QA"}
rows = list(csv.DictReader(open(os.path.join(HERE, "workload_ratios.csv"))))
g = lambda r, k: float(r[k])
ratio_str = lambda x: f"{x:.1f}:1" if x >= 1 else f"1:{1/x:.0f}"

fig, ax = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.2, 1]})

# ---- (a) (prefill, decode) plane ----
a = ax[0]
for r in rows:
    pm, dm = g(r, "pre_med"), g(r, "dec_med")
    xe = [[pm - g(r, "pre_p25")], [g(r, "pre_p75") - pm]]
    ye = [[dm - g(r, "dec_p25")], [g(r, "dec_p75") - dm]]
    c = "#1f77b4" if g(r, "ratio_agg") >= 1 else "#d62728"
    mk = "s" if r["kind"] == "prod-trace" else "o"
    a.errorbar(pm, dm, xerr=xe, yerr=ye, fmt=mk, ms=9, color=c, ecolor=c,
               elinewidth=1.2, capsize=3, alpha=.85, zorder=5)
    a.annotate(NAME[r["klass"]], (pm, dm), textcoords="offset points", xytext=(7, 5), fontsize=9)
a.plot([5, 2000], [5, 2000], "k--", lw=1, alpha=.6)
a.text(900, 700, "P = D", fontsize=9, alpha=.7, rotation=45)
a.text(1500, 12, "prefill-heavy", fontsize=9, color="#1f77b4", ha="right")
a.text(7, 1000, "decode-heavy", fontsize=9, color="#d62728")
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(6, 2000); a.set_ylim(10, 2000)
a.set_xlabel("prefill tokens  (median, box = p25–p75)")
a.set_ylabel("decode tokens  (median, box = p25–p75)")
a.set_title("Each use-case class as a region in the (prefill, decode) plane\n"
            "square = production trace · circle = Dolly-15k (Phi-3 tokenizer)")
a.grid(alpha=.3, which="both")

# ---- (b) aggregate P:D ratio, sorted ----
a = ax[1]
sr = sorted(rows, key=lambda r: g(r, "ratio_agg"))
y = np.arange(len(sr)); vals = [g(r, "ratio_agg") for r in sr]
cols = ["#d62728" if v < 1 else "#1f77b4" for v in vals]
a.barh(y, vals, color=cols, alpha=.85, log=True)
a.axvline(1, color="k", ls="--", lw=1)
for i, r in enumerate(sr):
    v = g(r, "ratio_agg")
    a.text(v * 1.3, i, ratio_str(v), va="center", ha="left", fontsize=9, color=cols[i])
a.set_yticks(y); a.set_yticklabels([NAME[r["klass"]] for r in sr], fontsize=9)
a.set_xlabel("aggregate prefill : decode  (Σprefill / Σdecode, log)")
a.set_title("Effective P:D per use-case class\n(decode-heavy <- 1:1 -> prefill-heavy)")
a.set_xlim(0.05, 30); a.grid(alpha=.3, axis="x", which="both")

fig.suptitle("LLM inference workload by use-case class — InstructGPT taxonomy (Ouyang+ 2022)\n"
             "operationalized with Dolly-15k + Azure/BurstGPT conversation traces", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_workload_pd.png"), dpi=130, bbox_inches="tight")
print("wrote fig_workload_pd.png")
