# 跨工作负载类型验证 P↔T 功率-cap 模型 —— 结果（v2）

> 理论模型与全部结果的**总纲**见仓库根目录 [MODEL_AND_RESULTS.zh.md](../../MODEL_AND_RESULTS.zh.md)；本文件是 portfolio 实验的细节页。

用 **10 类不同类型的 workload × 5 个模型**，各自扫功率 cap，验证 prefill/decode 两条 P↔T
解析律的普适性，并按实测中段偏差**精化了 decode 模型**（min() → 可加三阶段）。

- 数据：[`data/`](data/)（**v3 方法学**采集，伪影修复的证据链见 [DATA_QUALITY.zh.md](DATA_QUALITY.zh.md)）
- 图：[`fig_decode_models.png`](fig_decode_models.png)（**主图**：新旧 decode 模型逐 workload 对比）·
  [`fig_portfolio_grid.png`](fig_portfolio_grid.png)（prefill+decode 总览）·
  [`fig_tmax_validation.png`](fig_tmax_validation.png)（天花板理论 vs 实测）
- 表：[`decode_model_compare.csv`](decode_model_compare.csv) · [`portfolio_fits.csv`](portfolio_fits.csv)

硬件：V100-DGXS-32GB（HBM2 877 MHz 固定，SM 400–1530 MHz），cap 扫 [100..250] W。
方法：每 workload 固定 (seq/ctx, batch)，只动 cap → 单值 P↔T；测量方法学 v3
（chunk=ctx seeding、expandable_segments、步数窗口、重复中位、ctx_eff 记录）。

## 10 类 workload

| id | 类型 | 模型（架构/年代） | decode C×B | decode 平台 tok/s |
|---|---|---|---|--:|
| chat-phi3 | 对话 | Phi-3-mini (MHA, 2024) | 256×64 | 825 |
| rag-phi3 | RAG | Phi-3-mini | 1024×32 | 277 |
| code-phi3 | 代码补全 | Phi-3-mini | 2048×16 | 128 |
| longform-phi3 | 长文生成 | Phi-3-mini | 4096×8 | 109 |
| summarize-qwen7b | 摘要 (32k) | Qwen2.5-7B (GQA) | 32768×4 | 6.4 |
| translate-qwen3b | 翻译 | Qwen2.5-3B (GQA) | 512×64 | 559 |
| fastchat-qwen15b | 轻量对话 | Qwen2.5-1.5B (GQA) | 512×64 | 900 |
| classify-qwen7b | 分类/抽取 | Qwen2.5-7B | 256×8 | 149 |
| **qwen3chat-4b** | 现代对话 | **Qwen3-4B-2507 (GQA+QK-Norm, 2025)** | 1024×32 | 129 |
| **qwen3think-4b** | **推理/长思维链** | Qwen3-4B-2507 | 8192×8 | 16 |

## 结论

**1. Prefill 的 V²f 律（`P=P₀+κT(1+ρT)²`）在 9/10 workload 上成立**，R² 0.956–0.995
（qwen3think 0.974；qwen3chat 0.849 —— 最低 cap 点落在电压地板区之外）。

**2. Decode 精化：平台前分两阶段的可加模型显著优于 min() roofline。**
每 token 耗时 `t = T_mem + T_comp(x)`，`T_comp = C(x^{-p}−1)`，`x=f/f_max`；配 `P = P_s + χx^θ`
得显式 `T(P)`。三阶段：**I 计算主导**（幂律上升）→ **II 计算-访存交替**（可加过渡，旧 min()
在此系统性低估——"完美重叠"假设的必然）→ **III 带宽平台**（`B/T_mem`）。公平对比
（旧模型网格放开收敛、LOO 防过拟合）：

| 指标（10 workload） | 旧 min(V²f,T_max) | **可加三阶段** |
|---|---|---|
| R²（功率空间，全部点） | −0.21–0.93 | **0.90–0.997**（9/10） |
| 相对 RMSE | 3.3–12.1% | **0.6–3.2%**（除 classify 9.8%） |
| 留一交叉验证 | 9.3–21.2% | 2.1–27.5%，5/10 胜（败点均为最低功率点外推伪影） |

极端访存受限（32k 摘要、B=8 分类）不再需要"饱和特判"——它们是同一定律的小 C 极限。
拟合的阶段边界 P₁≈87–97 W（阶段 I 几乎在量程下限以下）、P₂≈109–177 W（越访存受限越靠前）。
注意 p、θ 是**有效**指数（近平坦曲线上弱可辨识，CSV `exponents_railed` 列标记）。

**3. 天花板量级 `T_max = B·BW_eff/(权重 + B·C_eff·kv/tok)` 跨 ~140×（6.4→900 tok/s）成立**
（按模型标定 BW_eff，C_eff 为漂移修正的有效上下文）。BW_eff 本身是"模型×硬件"属性：
Phi-3 ≈186、Qwen2.5-7B ≈260、Qwen2.5-3B/1.5B 与 Qwen3-4B ≈52–65 GB/s——小 hidden、多层的
架构带宽利用低；transformers DynamicCache 每步全 KV cat（读+写拷贝，每步实际访存≈权重+~3×KV）
也被标定吸收。显著残差：32k 摘要（长上下文稀疏 KV 读，有效带宽再打对折）。

**4. 前沿模型复核（2025）**：Qwen3-4B-Instruct-2507 上两类新负载（现代对话、长思维链推理）
两条律均成立（decode 可加模型 R² 0.99/0.92）。推理型负载把同一模型的天花板从 129 压到
16 tok/s（C 1024→8192）——decode 极重 + 长 KV 的 2025 负载形态在功率规划里代价极高。

## 诚实的偏差

- **classify-qwen7b（B=8 欠饱和）**：cap≥180 W 后实际功率只到 ~150 W，cap 不再咬合，DVFS
  调速器自行浮动，吞吐在平台附近散动（两张卡复现）——"cap 失效区"的真实行为，如实呈现。
- **∝B 弱化**：同 D_mem 的 rag/code/longform 平台比 2.2:1.2:1（理论 4:2:1）——低 batch 时
  并发流不足以打满带宽，decode 偏延迟受限。
- p、θ 触网格 / 弱可辨识的行在 CSV 有标记，阶段边界相应抑制。

## 复现

```bash
# 采集（空闲 GPU 上；方法学 v3 已固化在 runner 里）
SUDO_PASS=… CUDA_VISIBLE_DEVICES=<idle_gpu> PYTHONPATH=../../code python3 run_portfolio.py
# 拟合出图
python3 plot_decode_models.py   # 新旧 decode 模型对比（主图）
python3 plot_portfolio.py       # 总览 + T_max 验证
```
