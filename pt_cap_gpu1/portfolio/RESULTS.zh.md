# 跨工作负载类型验证 P↔T 功率-cap 模型 —— 结果

用 **8 类不同类型的 workload**，各自扫功率 cap，验证 [`../plot_theory.py`](../plot_theory.py) / [`../fig_theory_vs_measured.png`](../fig_theory_vs_measured.png) 里那套 P↔T 解析模型的**普适性**。数据在 [`data/`](data/)，图在 [`fig_portfolio_grid.png`](fig_portfolio_grid.png)（逐 workload 拟合）与 [`fig_tmax_validation.png`](fig_tmax_validation.png)（decode 天花板：理论 vs 实测），拟合表 [`portfolio_fits.csv`](portfolio_fits.csv)。

硬件：单张 V100-DGXS-32GB，250W cap，HBM2 877MHz 固定。方法：每个 workload 固定 (seq/ctx, batch)，只扫 cap `[100..250]W`，得一条单值 P↔T。

## 8 类 workload

| id | 类型 | 模型/架构 | decode C×B | prefill R² | decode T_max(实测) |
|---|---|---|---|--:|--:|
| chat-phi3 | 对话 | Phi-3-mini (MHA) | 256×64 | 0.972 | 739 |
| rag-phi3 | RAG | Phi-3-mini | 1024×32 | 0.991 | 212 |
| code-phi3 | 代码 | Phi-3-mini | 2048×16 | 0.992 | 129 |
| longform-phi3 | 长文生成 | Phi-3-mini | 4096×8 | 0.956 | 109 |
| summarize-qwen7b | 摘要(32k) | Qwen2.5-7B (GQA) | 32768×4 | 0.995 | 6 |
| translate-qwen3b | 翻译 | Qwen2.5-3B (GQA) | 512×64 | 0.984 | 561 |
| fastchat-qwen15b | 轻量对话 | Qwen2.5-1.5B (GQA) | 512×64 | 0.984 | 904 |
| classify-qwen7b | 分类 | Qwen2.5-7B | 256×8 | 0.991 | 154 |

## 三条结论

**1. Prefill 的 V²f 律（`P=P₀+κT(1+ρT)²`）在全部 8 类上成立** —— R² = 0.956–0.995。无论模型大小（1.5B→7B）、架构（MHA/GQA）、prompt 长度（256→4096）、batch（2→64），compute-bound 的 prefill 功率-吞吐曲线都是同一条凸的 V²f。这是最干净、最强的验证。

**2. Decode 的分段带宽天花板 `T=min(T_V²f, T_max)` 形状普适。** 6 个中等访存受限的 workload（chat/rag/code/longform/translate/fastchat）清晰呈现"V²f 上升 → 平台"三阶段，R² 0.66–0.95。两个极端访存受限的（summarize B=4/C=32k、classify B=8 权重主导）**在最低 100W cap 就已到平台**、全程平坦 —— 这不是反例，而是理论的强确认：**越访存受限，越低功率就饱和**（三阶段退化成"只剩平台"）。

**3. 天花板量级 `T_max = B·BW_eff/(权重 + B·C·kv/tok)` 在 150× 范围内被预测出来。** 实测 T_max 从 6 到 904 tok/s，各点基本贴 y=x（见 T_max 图）。用**正确的可加访存量** `D_mem=权重+KV` 反推，Phi-3 的 chat/rag/code 隐含有效带宽一致落在 136–165 GB/s，直接验证了"权重+KV"这个分解。

## 诚实的偏差（都可物理解释）

- **控制变量三元组** rag/code/longform（B·C 都=32768 → 同 D_mem）实测 T_max = 212/129/109，比值 1.9:1.2:1 vs 理论 4:2:1：**∝B 成立但偏弱**，因为低 batch 时并发流不足以打满带宽、decode 偏**延迟受限**，每流吞吐反而更高（longform B=8 隐含带宽 280 GB/s 即此离群）。
- **有效带宽随工作负载变**（57–270 GB/s）：小访存量（Qwen 小模型，4–7GB/步）带宽利用率低（~57 GB/s）；超长上下文（summarize C=32k，34 GB/s）的稀疏 KV 读也低。所以 `T_max` 公式是**一阶预测**——访存**量**抓得准，但流过这些字节的有效**带宽**本身随 batch/上下文浮动 2–8×。

## 一句话

模型的**函数形状**（prefill 凸 V²f、decode 分段带宽平台）在 8 类差异极大的 workload 上稳健成立；天花板的**量级**由 `B/D_mem` 一阶预测、跨 150× 成立，残差来自低-batch 延迟受限与有效带宽随负载的浮动。复现：`SUDO_PASS=… CUDA_VISIBLE_DEVICES=<idle_gpu> PYTHONPATH=../../code python3 run_portfolio.py` 然后 `python3 plot_portfolio.py`（或用 `auto_run.sh` 等空闲卡自动跑）。
