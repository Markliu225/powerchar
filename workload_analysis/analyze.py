"""prefill:decode token ratios per LLM USE-CASE CLASS — using a referenced taxonomy.

CLASSIFICATION SCHEME (not ad-hoc): the LLM use-case taxonomy from **InstructGPT**
(Ouyang et al., 2022, arXiv:2203.02155, Table 1), induced from REAL OpenAI API traffic into ~9 task
types: Generation, Open QA, Brainstorming, Chat, Rewrite, Summarization, Classification, Closed QA,
Extract. We OPERATIONALIZE it with real data:
  - **Dolly-15k** (databricks-dolly-15k): 8 human-labelled categories that map 1:1 onto the taxonomy;
    prefill = instruction (+ context), decode = response.   (tokenized with the local Phi-3 tokenizer)
  - **Production traces** (Azure conv + BurstGPT conv) for the 'Chat / multi-turn dialogue' class
    (Dolly is single-turn; the traces give real conversational input/output, already token-counted).

Per class we report prefill/decode stats and two ratios:
  ratio_agg = Σprefill/Σdecode (capacity-planning R_eff) ; ratio_med = median(prefill/decode).
Writes workload_ratios.csv.
"""
from __future__ import annotations
import csv, json, os, urllib.request
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")            # Phi-3 tokenizer is cached locally
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def stats(P, D):
    P, D = np.asarray(P, float), np.asarray(D, float)
    m = (P > 0) & (D > 0)
    P, D = P[m], D[m]
    pc = lambda a, p: float(np.percentile(a, p))
    return dict(n=int(len(P)),
                pre_mean=float(P.mean()), pre_p10=pc(P, 10), pre_p25=pc(P, 25), pre_med=pc(P, 50),
                pre_p75=pc(P, 75), pre_p90=pc(P, 90),
                dec_mean=float(D.mean()), dec_p10=pc(D, 10), dec_p25=pc(D, 25), dec_med=pc(D, 50),
                dec_p75=pc(D, 75), dec_p90=pc(D, 90),
                ratio_agg=float(P.sum() / D.sum()), ratio_med=float(np.median(P / D)))


def read_trace(fname, pcol, dcol, where=None):
    P, D = [], []
    for r in csv.DictReader(open(os.path.join(DATA, fname))):
        if where and not where(r):
            continue
        try:
            P.append(float(r[pcol])); D.append(float(r[dcol]))
        except (ValueError, KeyError):
            pass
    return P, D


def hf_rows(dataset, config, split, n):
    out = []
    for off in range(0, n, 100):
        url = (f"https://datasets-server.huggingface.co/rows?dataset={dataset}"
               f"&config={config}&split={split}&offset={off}&length={min(100, n-off)}")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "wl"}), timeout=40) as r:
            out += [x["row"] for x in json.load(r)["rows"]]
    return out


# Dolly category -> InstructGPT-taxonomy class name (cn / en)
DOLLY_MAP = {
    "creative_writing":       "Generation 创作生成",
    "brainstorming":          "Brainstorming 头脑风暴",
    "open_qa":                "Open QA 开放问答",
    "general_qa":             "General QA 常识问答",
    "closed_qa":              "Closed QA 闭卷问答",
    "summarization":          "Summarization 摘要",
    "information_extraction": "Extract 信息抽取",
    "classification":         "Classification 分类",
}


def main():
    rows = []

    # ---- Chat (multi-turn dialogue): real production traces, already token-counted ----
    Pc, Dc = [], []
    for fn, pc, dc, w in [("AzureLLMInferenceTrace_conv.csv", "ContextTokens", "GeneratedTokens", None),
                          ("BurstGPT_sample.csv", "Request tokens", "Response tokens",
                           lambda r: r.get("Log Type") == "Conversation log")]:
        p, d = read_trace(fn, pc, dc, w); Pc += p; Dc += d
    s = stats(Pc, Dc); s.update(klass="Chat 多轮对话", source="Azure+BurstGPT conv", kind="prod-trace")
    rows.append(s)

    # ---- the 8 Dolly categories (tokenize instruction+context -> prefill, response -> decode) ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    nlen = lambda t: len(tok(t, add_special_tokens=False).input_ids) if t else 0
    recs = hf_rows("databricks/databricks-dolly-15k", "default", "train", n=3000)
    by = {}
    for r in recs:
        cat = r.get("category")
        if cat not in DOLLY_MAP:
            continue
        instr, ctx, resp = r.get("instruction") or "", r.get("context") or "", r.get("response") or ""
        by.setdefault(cat, ([], []))
        by[cat][0].append(nlen(instr + ("\n" + ctx if ctx else "")))   # prefill
        by[cat][1].append(nlen(resp))                                  # decode
    for cat, (P, D) in by.items():
        s = stats(P, D); s.update(klass=DOLLY_MAP[cat], source="Dolly-15k", kind="benchmark")
        rows.append(s)

    rows.sort(key=lambda s: s["ratio_agg"])
    for s in rows:
        print(f"{s['klass']:24s} {s['source']:20s} n={s['n']:>6} "
              f"pre~{s['pre_med']:.0f} dec~{s['dec_med']:.0f}  P:D={s['ratio_agg']:.1f}:1")

    cols = ["klass", "source", "kind", "n",
            "pre_mean", "pre_p10", "pre_p25", "pre_med", "pre_p75", "pre_p90",
            "dec_mean", "dec_p10", "dec_p25", "dec_med", "dec_p75", "dec_p90",
            "ratio_agg", "ratio_med"]
    with open(os.path.join(HERE, "workload_ratios.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for s in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()})
    print(f"\nwrote workload_ratios.csv ({len(rows)} classes; taxonomy = InstructGPT/Ouyang 2022)")


if __name__ == "__main__":
    main()
