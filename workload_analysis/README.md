# LLM 推理工作负载分析 —— 按使用类型的 prefill : decode 比例

用**文献里公认的分类法**把 LLM 推理负载分成若干使用类型,并用**真实数据**量化每类的 prefill(输入)/ decode(输出)token 比例,为上层机架功率规划([PLANNING.zh.md](PLANNING.zh.md))提供真实的 `R_eff`,替代拍脑袋的 1:1 / 1:10。

> **分类法的文献依据、数据集出处、引用链接、完整结果与 caveat,见 [REFERENCES.zh.md](REFERENCES.zh.md)。**
> **论文用的完整方法学底稿(原理/流程/引用,含逐条核验与 BibTeX),见 [CLASSIFICATION.zh.md](CLASSIFICATION.zh.md)。**

## 分类法

采用 **InstructGPT(Ouyang et al., 2022)从真实 OpenAI API 流量归纳的使用类型分类**(该文 Table 1,共 10 类;Rewrite 无干净公开数据、Other 为杂项,未覆盖),用真实数据落地:8 类来自 **Dolly-15k** 的人工 `category` 标签(其中 General QA 为 Dolly 增设类,非 InstructGPT 原类),Chat 类来自 **Azure / BurstGPT 生产对话 trace**;再补一个生产级 **Code 代码补全**类(Azure code trace)——共 **10 类**。

## 结果(prefill 重 → decode 重)

| 使用类型 | P:D | | 使用类型 | P:D |
|---|--:|---|---|--:|
| Code 代码补全(生产) | 73.5:1 | | Open QA 开放问答 | 1:6 |
| Closed QA 闭卷问答 | 6.2:1 | | Brainstorming 头脑风暴 | 1:7 |
| Chat 多轮对话(生产) | 4.9:1 | | General QA 常识问答 | 1:8 |
| Extract 信息抽取 | 3.1:1 | | Generation 创作生成 | 1:11 |
| Summarization 摘要 | 2.3:1 | | Classification 分类 | ≈1:1 |

**核心规律:有给定上下文/原文的任务(闭卷问答、抽取、摘要、对话)prefill 重;凭知识自由生成的任务(创作、问答、头脑风暴)decode 重;分类居中。** 这正是功率规划要区分的两端。可视化见 `fig_workload_pd.png`。

> Dolly 是精炼指令集,绝对长度偏小、P:D 量级被压缩;它给出可靠的**任务类型相对次序**,生产规模的极端(长上下文 prefill 可达上万)更大——详见 REFERENCES。

## 文件

- `analyze.py` —— 拉数据 + 分词 → `workload_ratios.csv`
- `plot.py` —— → `fig_workload_pd.png`
- `curves_lib.py` —— **共享库**(不单独运行):分类法(NAME/MAP/CAVEAT)、曲线加载(load_curves,
  按更新的一阶理论 `fitlib.fit_*_theory` 拟合)、图构造(build_power_figs)。由 v100/ 与 h200/ 的
  包装器、以及 solve_rack_capping 直接 import——一套实现,两硬件不漂移。
- `v100/` —— **V100 版**(与 h200/ 对称):
  - `plot_power_curves.py`(包装器)→ `v100/fig_workload_power_throughput.png` ·
    `fig_workload_power_tokj.png`(prefill/decode 每类曲线,对数轴;tok/J 按实测功耗)·
    `workload_power_curves.csv`
  - `solve_rack_capping.py`(机架物理约束 5 kW / ≤32 槽 / cap∈[100,250] W,OPT vs TDP,优化内核
    import 自 `../../rack_power_capping/solve_workloads.py`)→ `v100/fig_workload_rack_capping.png` ·
    `workload_rack_capping.csv`
- `h200/` —— 同一流程跑在 H200 数据(`../data_h200`,cap 200–700 W;机架场景 14 kW / 32 槽 /
  TDP 700 W),脚本复用 `curves_lib` 与求解器内核,见 [h200/README.md](h200/README.md)
- `plot_profit_model.py` —— **论文正式经济模型**(式 (1)–(15):直线折旧、PUE 电费、运维率、
  价格指数衰减、accrual 累计利润 Φ(t)、ROI、现金流回收期、c_tok)按**混合 workload**的 5 年
  收益:真实负载是 J 个 P:D 类的混合(份额 w_j = 数据集 token 量占比,构成测量法见 ServeGen
  NSDI'26),系统按类分机架、每类机架数 ∝ 需求份额(N_j ∝ w_j/X_j);1 MW 集群 CAP vs TDP,
  V100 & H200 → `fig_profit_model.png` · `profit_model.csv`
- `data/` —— 缓存的生产 trace 样本(Azure conv/code、BurstGPT)
- `REFERENCES.zh.md` —— 分类法依据 + 数据集 + 引用链接 + caveat

## 复现

```bash
python3 workload_analysis/analyze.py
python3 workload_analysis/plot.py
python3 workload_analysis/v100/plot_power_curves.py
python3 workload_analysis/v100/solve_rack_capping.py
python3 workload_analysis/plot_profit_model.py
```
