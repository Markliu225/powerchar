# LLM 推理负载分类:采用 InstructGPT 使用类型分类法

## 为什么用这套分类(文献依据)

我们**不自己拍分类**,而是直接采用文献里公认的一种:**InstructGPT**(Ouyang et al., 2022,[arXiv:2203.02155](https://arxiv.org/abs/2203.02155))从**真实 OpenAI API 流量**归纳出的使用类型分类(该文 Table 1),共约 9 类:

> **Generation(生成)· Open QA(开放问答)· Brainstorming(头脑风暴)· Chat(对话)· Rewrite(改写)· Summarization(摘要)· Classification(分类)· Closed QA(闭卷问答)· Extract(抽取)**

选它的理由:(1) 来自**真实生产使用统计**,不是凭空划分;(2) 9 类落在"6–10 类"的合理粒度;(3) 每一类的 input/output(= prefill/decode)长度特征天然不同,正好是功率规划需要的区分维度。

## 怎么落到真实数据(operationalize)

- **8 类用 Dolly-15k**:[databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)(Databricks 2023 发布的真实人工指令数据)自带 8 个 `category` 标签,**正好一一对应**上面的分类法。按 category 分组,`prefill = instruction + context`、`decode = response`,用本地 Phi-3 分词器计数。
- **Chat 多轮对话用生产 trace**:Dolly 是单轮指令、没有 Chat 类,用真实生产日志 [Azure LLM Inference Trace](https://github.com/Azure/AzurePublicDataset)(conv)+ [BurstGPT](https://github.com/HPMLL/BurstGPT)(Conversation log)补上,token 数是线上实测,共 2.6 万条对话。
- InstructGPT 的 **Rewrite(改写)** 缺乏干净的公开数据集,本轮未单列(改写类 input≈output,P:D≈1:1)。

## 结果:9 类的 prefill : decode

按 prefill 重 → decode 重 排序(`P:D = Σprefill / Σdecode`):

| 使用类型(InstructGPT) | 数据源 | prefill 中位 | decode 中位 | **聚合 P:D** |
|---|---|--:|--:|--:|
| Closed QA 闭卷问答(给定上下文) | Dolly-15k | 222 | 29 | **6.2 : 1** |
| Chat 多轮对话 | Azure + BurstGPT(生产) | 968 | 135 | **4.9 : 1** |
| Extract 信息抽取 | Dolly-15k | 240 | 36 | **3.1 : 1** |
| Summarization 摘要 | Dolly-15k | 244 | 95 | **2.3 : 1** |
| Classification 分类 | Dolly-15k | 30 | 32 | **0.8 : 1**(≈1:1) |
| Open QA 开放问答 | Dolly-15k | 10 | 38 | **1 : 6** |
| Brainstorming 头脑风暴 | Dolly-15k | 13 | 65 | **1 : 7** |
| General QA 常识问答 | Dolly-15k | 10 | 102 | **1 : 8** |
| Generation 创作生成 | Dolly-15k | 14 | 160 | **1 : 11** |

完整统计见 [workload_ratios.csv](workload_ratios.csv),可视化见 `fig_workload_pd.png`。

**结构很清楚**:有给定上下文/原文的任务(闭卷问答、抽取、摘要、对话)是 **prefill 重**;凭知识自由生成的任务(创作、头脑风暴、开放/常识问答)是 **decode 重**;分类正好在 1:1。这正是功率规划要区分的两端。

## 口径与诚实标注

- **Dolly 是精炼指令集,绝对长度偏小**(instruction ~10–30、context ~200、response ~30–160 token),所以这里的 P:D 量级被压缩在 `1:11 ~ 6:1`。它可靠地给出**任务类型之间的相对次序**,但不是生产规模的绝对值。
- **生产规模的极端要大得多**:真实长上下文 RAG / 长文档摘要的 prefill 可达数千到上万 token(对照本表 Chat 类的生产数据 prefill ~968,以及 Azure code trace 的 ~1500、长文档摘要 GovReport 的 ~9600)。也就是说真实部署里 prefill 重的几类会比 Dolly 显示的更极端。
- **decode 用的是真实答案长度**:Dolly 的 response 是人工写的,比一般 benchmark 的参考输出更接近真实生成;生产 trace(Chat)的 decode 是线上实际生成长度。
- 换分词器/模型,绝对 token 数有 ±10~20% 差异,但 P:D 量级稳健。

## 与功率规划的衔接

在 [../rack_power_capping/WORKLOAD_PORTFOLIO.zh.md](../rack_power_capping/WORKLOAD_PORTFOLIO.zh.md) 的 router 假设下,**每一类使用类型的聚合 P:D 就是一个专供该类的机架该喂给 `solve.py` 的 `R_eff`**;再配上该类的上下文长度(≈prefill 中位)去定 decode 天花板 `T_max(C)`、用其 SLO 定功率 cap 下界,即可为每类算出专属机架配方。

## 参考文献

1. Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT;使用类型分类法见 Table 1), 2022. https://arxiv.org/abs/2203.02155
2. Conover et al., *Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM* (databricks-dolly-15k), Databricks, 2023. https://huggingface.co/datasets/databricks/databricks-dolly-15k
3. Patel et al., *Splitwise: Efficient Generative LLM Inference Using Phase Splitting* (Azure LLM Inference Trace), ISCA 2024. https://arxiv.org/abs/2311.18677 · https://github.com/Azure/AzurePublicDataset
4. Wang et al., *BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems*, 2024. https://arxiv.org/abs/2401.17644 · https://github.com/HPMLL/BurstGPT

## 复现

```bash
python3 workload_analysis/analyze.py   # InstructGPT 分类 + Dolly/trace -> workload_ratios.csv
python3 workload_analysis/plot.py      # -> fig_workload_pd.png
```
