# LLM 推理工作负载分析 —— 按使用类型的 prefill : decode 比例

用**文献里公认的分类法**把 LLM 推理负载分成若干使用类型,并用**真实数据**量化每类的 prefill(输入)/ decode(输出)token 比例,为上层机架功率规划([../rack_power_capping/WORKLOAD_PORTFOLIO.zh.md](../rack_power_capping/WORKLOAD_PORTFOLIO.zh.md))提供真实的 `R_eff`,替代拍脑袋的 1:1 / 1:10。

> **分类法的文献依据、数据集出处、引用链接、完整结果与 caveat,见 [REFERENCES.zh.md](REFERENCES.zh.md)。**

## 分类法

采用 **InstructGPT(Ouyang et al., 2022)从真实 OpenAI API 流量归纳的使用类型分类**(约 9 类),用真实数据落地:8 类来自 **Dolly-15k** 的人工 `category` 标签,Chat 类来自 **Azure / BurstGPT 生产对话 trace**。

## 结果(prefill 重 → decode 重)

| 使用类型 | P:D | | 使用类型 | P:D |
|---|--:|---|---|--:|
| Closed QA 闭卷问答 | 6.2:1 | | Open QA 开放问答 | 1:6 |
| Chat 多轮对话 | 4.9:1 | | Brainstorming 头脑风暴 | 1:7 |
| Extract 信息抽取 | 3.1:1 | | General QA 常识问答 | 1:8 |
| Summarization 摘要 | 2.3:1 | | Generation 创作生成 | 1:11 |
| Classification 分类 | ≈1:1 | | | |

**核心规律:有给定上下文/原文的任务(闭卷问答、抽取、摘要、对话)prefill 重;凭知识自由生成的任务(创作、问答、头脑风暴)decode 重;分类居中。** 这正是功率规划要区分的两端。可视化见 `fig_workload_pd.png`。

> Dolly 是精炼指令集,绝对长度偏小、P:D 量级被压缩;它给出可靠的**任务类型相对次序**,生产规模的极端(长上下文 prefill 可达上万)更大——详见 REFERENCES。

## 文件

- `analyze.py` —— 拉数据 + 分词 → `workload_ratios.csv`
- `plot.py` —— → `fig_workload_pd.png`
- `data/` —— 缓存的生产 trace 样本(Azure conv/code、BurstGPT)
- `REFERENCES.zh.md` —— 分类法依据 + 数据集 + 引用链接 + caveat

## 复现

```bash
python3 workload_analysis/analyze.py
python3 workload_analysis/plot.py
```
