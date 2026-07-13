# 真实 workload 的分类与机架配方（V100，5 kW / 32 卡）

> 本目录唯一的机架文档：从"为什么按类规划"到"每类的配方与经济性"一篇讲完。
> 求解器（唯一）：[`solve_workloads.py`](solve_workloads.py) · 图：[`fig_workloads.png`](fig_workloads.png) ·
> 表：[`workloads_results.csv`](workloads_results.csv) · 经济性：[`economics.py`](economics.py) →
> [`fig_payback.png`](fig_payback.png)。早期"合成 P:D 比例扫描"求解器与规划长文已并入本文/移除，
> 见 git 历史。

## 1. 规划框架：为什么按 workload 类别规划机架

这里做的是**规划**（planning），不是运行时：部署前离线决定每个机架 prefill / decode 各配几张卡、
每张卡 cap 到多少瓦，产出静态容量蓝图；在线路由与动态调度是另一层，不在本文范围。

早期做法把负载压成一个合成的 prefill:decode 比例，喂给同一对 chat 曲线。这在数学上干净，但等于
假设"全世界的请求都长一个样"。真实流量来自不同应用——聊天、RAG、写长文、批量摘要——每类的
prefill/decode 长度、**上下文量级**、**延迟要求**都不同；把它们画在 `(L_p, L_d)` 平面上是散落的
几团点云，而单一比例只承认一条过原点的射线。

破局的关键假设是 **router：让每个机架只服务一类应用**。这在规划上消掉了最难的异质性——不再有
"256 上下文的聊天与 8k 上下文的推理挤同一条 decode 曲线"的问题，每个机架内部同质，可以独立
规划。于是问题裂成两层：**机架内**（本文：每类应用的配方）与**机架间**（每类分几个机架，按需求
预测配额，未做）。代价也在规划期可见：小众应用填不满一个机架的粒度浪费、以及对需求预测的依赖。

不同类别的机架配方差异来自两个物理事实（详见总纲
[MODEL_AND_RESULTS.zh.md](../../MODEL_AND_RESULTS.zh.md) §4–5）：一是**上下文决定 decode 天花板**
（`T_max ∝ 1/C`）：同为一张 V100，chat 类实测平台 825 tok/s，32k 摘要只剩 6.4 tok/s——产出同一个
token 的功率成本差两个数量级；二是**延迟 SLO 决定 cap 下界**：交互类压 cap 直接抬 TTFT/逐字延迟，
批处理类可以一路压到能效甜点。

## 2. workload 分类

10 个实测 workload（portfolio v3 数据集）归入 **6 个应用类**，按 decode 主导 → 纯 prefill 排序。
每类携带一个类别级请求形状假设（P:D，进入且仅进入求解器的 token 平衡方程；所有展示只用类别名）。
分类与形状定义固化在 `solve_workloads.WORKLOAD_CLASSES`，预测变了改那里重跑。

| 类别 | 形状特征 | 形状假设 P:D | 成员（模型 / decode C×B / 平台 tok/s） |
|---|---|---|---|
| **长生成 / 推理** long-gen/CoT | 短题/短问 → 长文/长思维链，decode 主导 | 1:10 | longform-phi3（Phi-3 / 4096×8 / 109）· qwen3think-4b（Qwen3-4B / 8192×8 / 16） |
| **对话** chat | 短上下文往返，decode 偏重 | 1:2 | chat-phi3（Phi-3 / 256×64 / 825）· fastchat-qwen15b（Qwen2.5-1.5B / 512×64 / 900）· qwen3chat-4b（Qwen3-4B / 1024×32 / 129） |
| **翻译 / 对称** balanced | 输入输出等长 | 1:1 | translate-qwen3b（Qwen2.5-3B / 512×64 / 559） |
| **RAG / 代码** prompt-heavy | 长提示 → 短交互回答，prefill 偏重、TTFT 敏感 | 10:1 | rag-phi3（Phi-3 / 1024×32 / 277）· code-phi3（Phi-3 / 2048×16 / 128） |
| **批量摘要** summarize (batch) | 超长文档 → 短摘要，批处理 | 30:1 | summarize-qwen7b（Qwen2.5-7B / 32768×4 / 6.4） |
| **分类 / 抽取** classify (batch) | ≈纯 prefill，批处理 | 100:1 | classify-qwen7b（Qwen2.5-7B / 256×8 / 149） |

### 分类的文献与数据依据（已核验）

分类图：[../../workload_analysis/fig_workload_pd.png](../../workload_analysis/fig_workload_pd.png)。
分类法不是自拟的，出处已逐条核验：**InstructGPT**[1] 的 Table 1 把真实 OpenAI API 提示流量分为
**10 类**（Generation, Open QA, Brainstorming, Chat, Rewrite, Summarization, Classification,
Other, Closed QA, Extract）。落地数据的覆盖情况如实说明：其中 **8 类有数据**（7 类来自
**Dolly-15k**[2]——其官方卡片写明其 8 个标签中 7 个取自 InstructGPT 分类、**General QA 是
Databricks 增设的自由问答类**；Chat 用生产 trace：Azure conv[3] + BurstGPT[4]）；Rewrite 无干净
公开数据集未单列，Other 为杂项不计；另补一个**生产级 Code 代码补全类**（Azure code trace[3]，
超出 2022 原分类法）。共 10 个使用类型，在 P:D 轴上落成三个带（decode-重 4 / 平衡 1 /
prefill-重 5），统计脚本与完整数字见 [../../workload_analysis/](../../workload_analysis/)。

**文献类 ↔ 实测 workload 的逐条对应**（双向核验；口径差异如实标注）：

| 文献类（数据实测聚合 P:D） | 本文机架类 / workload | 口径说明 |
|---|---|---|
| Generation 1:11 | 长生成/推理（longform-phi3, qwen3think-4b），形状 1:10 | 直接对应；长思维链推理是 2022 分类法没有的 2025 形态，归入同带 |
| Open QA 1:6 · Brainstorming 1:7 · General QA 1:8 | 同 decode-重带，由长生成/推理类覆盖 | 无专门实验；General QA 为 Dolly 增设类 |
| Chat 4.9:1（生产 trace，每请求带全历史） | 对话（chat-phi3, fastchat, qwen3chat），形状 1:2 | **口径不同**：trace 按每轮重算全上下文计 4.9:1；服务端跨轮复用 KV（prefix caching）时每轮只算新增 ≈1:2，与我们 decode 曲线的测量口径一致 |
| Classification ≈1:1（Dolly 短样本） | 平衡带 ↔ 翻译（translate-qwen3b），形状 1:1 | Dolly 分类样本是短提示短答案的 1:1，是**平衡带**的锚；翻译 ≈ InstructGPT 的 Rewrite（输入≈输出），该类无公开数据 |
| Summarization 2.3:1（Dolly，文档中位 244 tok） | 批量摘要（summarize-qwen7b），形状 30:1 | **规模不同**：Dolly 文档短；生产级长文档（本实验 32k）prefill 上万，取生产口径 |
| Extract 3.1:1（Dolly） | 分类/抽取（classify-qwen7b），形状 100:1 | 同上：生产级"整篇文档→标签/字段"的 prefill 远长于 Dolly 样本 |
| Closed QA 6.2:1（Dolly） | RAG（rag-phi3），形状 10:1 | RAG=带检索上下文的闭卷问答；生产检索提示 ~4k（本实验 prefill S=4096），比 Dolly 的 222 长一个量级 |
| Code 73.5:1（Azure code 生产 trace，2023） | 代码（code-phi3），形状 10:1 | **时代不同**：2023 补全式中位只生成 13 tok（73.5:1）；2025 代码对话/agent 生成更长，本文取 10:1，保守于 trace |

**参考文献**

1. Ouyang et al., *Training language models to follow instructions with human feedback*
   (InstructGPT；使用类型分类法见 Table 1), NeurIPS 2022. <https://arxiv.org/abs/2203.02155>
2. Conover et al., *Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM*
   (databricks-dolly-15k), Databricks, 2023.
   <https://huggingface.co/datasets/databricks/databricks-dolly-15k>
3. Patel et al., *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*
   (Azure LLM Inference Traces, conv+code), ISCA 2024. <https://arxiv.org/abs/2311.18677>
4. Wang et al., *BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems*, 2024.
   <https://arxiv.org/abs/2401.17644>

## 3. 输入曲线与求解设定

每个 workload 的 prefill / decode 曲线来自 `pt_cap_gpu1/portfolio/data/`（v3 方法学），经
`fitlib` 统一模型拟合；**x 轴 = cap_w**——机架按强制执行的 cap 供电配容，不按波动的实测功率，
产出的配方就是 `nvidia-smi -pl` 的设定值。约束：

1. **功率预算** `W_RACK = 5000 W`（按 cap 计）；
2. **整数卡、每相 ≥1**（disaggregated serving 两相都要有 worker）；
3. **物理插槽上限 `N_GPU_MAX = 32`**（如 4 节点 × 8 卡）——功率封顶只能加卡加到机箱装满；
4. **cap 限于实测区间 [100, 250] W**（不外推），decode 不超过其饱和 cap（平台上加瓦不加 token）。

优化内核：枚举整数 (Np, Nd)，相内均匀 cap（曲线凹），相间功率劈分扫描取平衡吞吐最大、同吞吐
取更省电者。

## 4. 结果：按类别的机架配方

`OPT` = cap 浮动 + 花满预算 + 插槽感知；`TDP` = 全部 250 W。完整数字见
[`workloads_results.csv`](workloads_results.csv)，图见 [`fig_workloads.png`](fig_workloads.png)。

| 类别 | workload | OPT 配方（Np+Nd @ pre/dec W） | OPT tok/s | TDP tok/s | 增益 | rack tok/J（OPT/TDP） |
|---|---|---|--:|--:|--:|---|
| 长生成 / 推理 | longform-phi3 | 1+31 @100/158 | 3.5k | 2.3k | +54% | 0.70 / 0.45 |
| 长生成 / 推理 | qwen3think-4b | 1+31 @100/158 | 0.54k | 0.33k | +61% | 0.11 / 0.07 |
| 对话 | chat-phi3 | 2+30 @150/157 | 34.1k | 23.5k | +45% | 6.82 / 4.71 |
| 对话 | fastchat-qwen15b | 1+31 @165/156 | 39.9k | 25.7k | +56% | 7.98 / 5.13 |
| 对话 | qwen3chat-4b | 1+31 @106/158 | 5.9k | 3.7k | +62% | 1.19 / 0.74 |
| 翻译 / 对称 | translate-qwen3b | 2+30 @197/154 | 31.4k | 20.1k | +56% | 6.27 / 4.02 |
| RAG / 代码 | rag-phi3 | 11+21 @182/143 | 58.4k | 39.7k | +47% | 11.68 / 7.93 |
| RAG / 代码 | code-phi3 | 5+27 @206/147 | 34.9k | 23.1k | +51% | 6.98 / 4.63 |
| 批量摘要 | summarize-qwen7b | 3+29 @184/137 | 5.7k | 3.6k | +60% | 1.26 / 0.71 |
| 分类 / 抽取 | classify-qwen7b | 27+5 @167/100 | 66.7k | 49.1k | +36% | 13.33 / 9.82 |

（物理插槽上限在所有 10 行都咬合；长生成/对话类多数还叠加 Np=1 地板。）

## 5. 分析

**"decode-heavy 配方"≠"decode-heavy 请求"——上下文才是主宰。** 最反直觉的是批量摘要类：请求
重度 prefill，配方却把 29/32 的卡堆在 decode——32k 上下文把单卡 decode 平台压到 6.4 tok/s，每个
decode token 比 prefill token 贵约两个数量级，卡数劈分跟着 **token 成本比**走而不是 token 数量比。
真正往 prefill 侧堆卡的只有 RAG/代码（11+21、5+27）和分类（27+5）——它们的 decode 还便宜。

**decode 卡的 cap 几乎是常数，prefill 卡的 cap 才是调节旋钮。** 十份配方里 decode cap 全部落在
137–158 W（各自饱和 cap 附近，再高纯属浪费）；prefill cap 则从 100 W（over-provisioned，压到
地板腾预算）到 206 W（prefill 是瓶颈，抬 cap 换容量）浮动。运维含义：decode 池
`nvidia-smi -pl` 一刀切设 ~150 W 即可，prefill 池按类调。

**物理插槽上限在每一类上都咬合。** 5 kW / 32 slot = 156 W/卡，恰好 ≥ 各相能效甜点（~110–180 W），
最优解永远是"先填满 32 个槽，让 cap 落在甜点附近"。拆掉 slot 墙，纯功率最优要 38–45 张卡；钳回
32 槽损失 4–24% 吞吐（decode 越重损失越大）。这也解释了增益结构：TDP 同预算只能装 20 张卡，
OPT 装满 32 张——**+36% ~ +62% 吞吐、同功率**，增益上限由插槽墙（而非预算）决定。

**同一个 5 kW 机架，产出跨 ~124×。** 分类机架 66.7k tok/s，长 CoT 推理机架 0.54k tok/s。按 token
均一计费时，"机架卖哪类 token"远比"机架怎么调"重要；长思维链这种 2025 形态是功率规划里最贵的
一类负载（与总纲 §5.5 一致）。

**能效全面占优。** OPT 的 rack 级 tok/J 在每类上都是 TDP 的 ~1.4–1.6×，增益与吞吐同源：把每瓦
花在曲线的高效区段。

## 6. 经济性：同一个招式，按类分化的回本

Capping 的账是"用同样的电装更多的卡"：每类机架 OPT 比 TDP 多 12 张卡（+$30k CapEx），换多出的
token 收入（能耗两边相同，相抵）。按 $0.05/M 输入、$0.20/M 输出、卡价 $2500 计
（[`economics.csv`](economics.csv) · [`fig_payback.png`](fig_payback.png)）：

| 类别 | 额外 CapEx 回本 |
|---|---|
| 对话（chat-phi3 / fastchat） | **163–219 天** |
| 翻译 / 对称 | 247 天 |
| RAG / 代码 | 291–464 天 |
| 分类 / 抽取 | 384 天 |
| 对话'25（qwen3chat，平台低） | 1,023 天 |
| 长生成 / 推理 | 1,529–**9,104 天**（≈永不回本） |
| 批量摘要（32k） | 2,940 天 |

结论：**capping 值不值，取决于机架卖的是哪类 token**。token 富裕的类别（对话/RAG/分类）一年内
回本；decode 被长上下文饿死的类别（长 CoT、32k 摘要）多买的卡在这个价格下收不回来——那些机架
更该把钱花在换代（更高带宽的卡）而不是加卡上。

## 7. 局限与下一步

- **延迟 SLO 未入约束**：交互类（对话/RAG/代码）压 cap 会抬 TTFT 与逐字延迟，实际部署应给这些类
  的 cap 设下界；当前配方是纯吞吐最优。
- **形状假设是类别级旋钮**（§2），不是实测分布；真实 trace 的比例统计见
  [../../workload_analysis/](../../workload_analysis/)，可直接替换。
- **机架间配额层**（每类分几个机架）是规划的第二层，尚未实现，输入是各类的需求预测。
- **H200 复算**：`data_h200` 决胜数据齐后，同一求解器换数据目录即可出 `rack_power_capping/h200/`。
