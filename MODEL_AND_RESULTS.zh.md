# LLM 推理的功率↔吞吐模型与全部实验结果 —— 总纲

> **本文件是唯一权威入口**：完整给出 prefill 与 decode 两个阶段的理论模型（从第一性原理到实测形式），
> 并汇总全部实验结果（V100 上 10 类 workload × 5 个模型的验证）。
> 各子文档只保留细节：测量方法学见 [pt_cap_gpu1/portfolio/DATA_QUALITY.zh.md](pt_cap_gpu1/portfolio/DATA_QUALITY.zh.md)，
> decode 模型的推导细节与单卡标定见 [pt_cap_gpu1/decode_model_theory.md](pt_cap_gpu1/decode_model_theory.md)，
> H200 运行指引见 [H200_操作手册.md](H200_操作手册.md)。

---

## 0. 一页速览

**建模对象是 `P ↔ T`**（功率 ↔ token 吞吐）。旋钮 = **功率上限**（`nvidia-smi -pl`）：设定后 GPU 自选
能维持的核心频率，功率与吞吐同时移动；频率只是中间机制，不出现在最终模型里。

| 阶段 | 瓶颈（roofline） | 功率↔吞吐模型 | 形状 |
|---|---|---|---|
| **prefill** | 计算受限（`I ≫ I*`） | `T(P) = T_fmax·((P−P_s)/χ)^{p/θ}`（`T_mem→0` 退化） | **单段**幂律、凹，量程内无平台 |
| **decode** | 访存受限（`I ≪ I*`） | `T(P) = B / (T_mem + C_c·[x(P)^{-p} − 1])`，`x(P)=((P−P_s)/χ)^{1/θ}` 钳到 ≤1 | **三阶段**：幂律上升 → 交替过渡 → 带宽平台 `T_max = B/T_mem` |

**验证结论（V100，10 类 workload × 5 模型，功率 cap 扫描）**：
- prefill 统一显式模型：9/10 R² ≥0.92，且 **10 个 workload 全部复原 `p≈1`**（吞吐∝频率的
  计算受限机制指数，与 DVFS 直测 `T∝f^0.90` 一致）；等价的 V²f 参数化 R² 0.85–0.995；
- decode 三阶段可加模型：9/10 的 R² = **0.90–0.997**（相对 RMSE 0.6–3.2%），全面优于旧的
  `min(V²f, T_max)` 分段近似（R² −0.21–0.93）；
- 平台量级 `T_max = B·BW_eff/(权重 + B·C·kv/tok)` 跨 **~140×**（6.4→900 tok/s）成立。

**能效直觉**：同功率下 prefill 每焦耳产出 ≈ **10×** decode（prefill 权重复用、decode 每步重读全部权重）；
prefill 能效峰在中低功率（V100 上 ≈40 tok/J @ ~155 W），decode 过拐点后能效单调下降 —— 这是功率
封顶（power capping）对 decode 几乎免费、对 prefill 昂贵的根源，也是机架级规划的物理基础。

---

## 1. 统一框架：两条功率物理原语 + roofline

### 1.1 两条功率原语（一切的来源）

**(a) 逻辑/计算动态功率**。开关的 CMOS 阵列耗散 `P_logic = α·C·V²·f`。可靠翻转要求电压随频率上升
（活跃区 `V ≈ V₀ + γf`），故 **每操作能量 `E_op ∝ V²` 随速度上升** —— 算得越快、每次运算越贵。

**(b) 访存/搬运功率**。搬 1 bit 耗散近似**固定**能量 `E_bit`（充放固定的线与单元电容；HBM 的数据
时钟不随核心 DVFS 调压），故 `P_mem = E_bit·BW`，**每 bit 能量与速度无关**。

> **一切不对称的根源**：计算的单位能量随速度涨（`∝V²`），访存的单位能量不涨。

### 1.2 roofline：两个阶段各卡在哪

每 token 两阶段 FLOPs 几乎相同，决定性差异在**权重复用**（算术强度 `I` 相对脊点 `I* = Φ/β`）：

| | FLOPs/步 | 字节/步 | 算术强度 | 判定 |
|---|---|---|---|---|
| prefill | `≈2N·(B·S)` | 每个权重 tile 读一次、被全部 `B·S` 个位置复用 | `I ≫ I*` | **计算受限** → `T ∝ f_sm` |
| decode | `≈2N·B` | `D_mem = W + B·C·kv`（每步重读全部权重 + B 条序列的 KV） | `I ≪ I*` | **访存受限** → `T ∝ BW` |

（`W`=权重字节，`kv`=每 token 的 KV 字节 `2L·n_kv·h·b`，`C`=上下文长，`B`=batch。）

### 1.3 理想极限（第一性原理的"渐近律"）

在 `V ∝ f` 的理想区（无功率墙、无热限）：
- **prefill**：`T∝f` 且 `P∝V²f∝f³` ⇒ **`P ≈ P₀ + k_c·T³`（立方）**，每 token 能量 `∝T²`；
- **decode**：`T∝BW` 且 `P∝BW` ⇒ **`P ≈ P₀ + k_m·T`（线性）**，每 token 能量 ≈ 常数。

**实测校验（V100 锁频 DVFS 510→1530 MHz）**：机制成立 —— prefill `T ∝ f^0.90`（R²=0.99）、
decode `T ∝ f^0.26`（频率×3 只换 ×1.37 吞吐，访存受限确认）。但**立方指数不干净**：V100 中段
V-f 曲线平坦（`V≈const ⇒ P_dyn∝f`），拟合指数对静态底 P₀ 简并（P₀=44 W 时 γ≈1.5，P₀=90 W 时
γ≈3.0）。decode 线性律经 **batch 旋钮**确认：`P = 111 + 0.190·T`，R²=0.996。
⇒ 理想律是渐近极限；**下面的实测形式才是用于拟合与预测的模型**。

---

## 2. Prefill 理论模型（计算受限 → 单段显式幂律）

> 完整推导（与 decode 严格同构）见 [pt_cap_gpu1/prefill_model_theory.md](pt_cap_gpu1/prefill_model_theory.md)。

### 2.1 构建（与 decode 同一条定律，`T_mem→0` 的退化）

每次前向耗时同样服从 `t = T_mem + T_comp`；prefill 权重复用（`I≫I*`）使 **`T_mem ≪ T_comp`
在整个 DVFS 范围成立**，故 `t ≈ O_comp/OPS(f_sm)`、`OPS ∝ f^p`：

$$T(x) = T_{f_{max}}\cdot x^{p},\qquad P(x) = P_s + \chi x^{\theta},\qquad x=f_{sm}/f_{max}$$

反解功率代回（与 decode 相同的合成步骤），得显式模型：

$$\boxed{\;\text{Throughput}(P) = T_{f_{max}}\left(\frac{P-P_s}{\chi}\right)^{p/\theta}\;}$$

**单段**幂律（指数 `p/θ<1` → 凹），无阶段、无带宽平台——唯一的饱和是频率顶 `x=1`
（通常在 cap 量程之外）。阶段结构的有无完全由 `T_mem` 地板决定：decode 有地板 → 三阶段；
prefill 没有 → 单段。旧文档的 V²f 形式 `P(T)=P₀+κT(1+ρT)²` 是同一物理在仿射电压下的
等价参数化，保留作基线。

### 2.2 实测验证（跨 10 类 workload）

两步时钟空间拟合（与 decode 同流程），V100 方法学 v3：

| workload | S×B | 统一 R² | V²f R² | p | | workload | S×B | 统一 R² | V²f R² | p |
|---|---|--:|--:|--:|---|---|---|--:|--:|--:|
| chat-phi3 | 512×8 | 0.923 | 0.972 | 1.32 | | translate-qwen3b | 512×8 | 0.962 | 0.984 | 1.02 |
| rag-phi3 | 4096×2 | 0.992 | 0.991 | 0.92 | | fastchat-qwen15b | 512×16 | 0.950 | 0.984 | 1.01 |
| code-phi3 | 2048×4 | 0.992 | 0.992 | 0.97 | | classify-qwen7b | 2048×4 | 0.986 | 0.991 | 0.91 |
| longform-phi3 | 256×16 | 0.957 | 0.956 | 1.19 | | qwen3chat-4b | 512×8 | 0.871 | 0.849 | 1.18 |
| summarize-qwen7b | 4096×2 | 0.986 | 0.995 | 0.94 | | qwen3think-4b | 2048×4 | 0.979 | 0.974 | 0.81 |

**核心验证是 p 列**：`p = 0.81–1.32`（中位 ≈0.99）——10 个 workload 从 cap 扫描**独立复原**
"计算受限 ⇒ 吞吐∝频率"的机制指数，与锁频 DVFS 直测的 `T∝f^0.90` 一致。两种参数化拟合能力
相当（统一模型胜在框架一致与参数可解释；V²f 样本内略优）；qwen3chat 两者皆偏低（0.85/0.87，
最低 cap 点落在电压地板区）。图：[fig_prefill_models.png](pt_cap_gpu1/portfolio/fig_prefill_models.png)。

### 2.3 能效推论（闭式）

`E(P) = T/P ∝ (P−P_s)^{p/θ}/P`，令 `a=p/θ`，能效峰在 **`P* = P_s/(1−a)`**。
V100+Phi-3 实测峰 ≈**40 tok/J @ ~155 W**（62% TDP），与 `P_s≈70–90 W、a≈0.4–0.5` 一致。
能效敏感的部署把 prefill 压到峰附近，代价是 TTFT——交互应用的 cap 下界由延迟 SLO 决定。

---

## 3. Decode 理论模型（访存受限 → 三阶段可加）

### 3.1 物理设定：时间相加，而非取大

单步耗时由**访存**与**计算**两部分**相加**（串行/不重叠极限；旧模型的 `min()`/roofline 是完美重叠
极限，见 §3.5）：

$$t_{\text{step}} = T_{mem} + T_{comp}(x),\qquad T = \frac{B}{t_{\text{step}}},\qquad x=\frac{f_{sm}}{f_{max}}$$

两条关键约束：
1. **`T_mem` 是常数地板**：`T_mem = D_mem/BW_{eff}`，`D_mem = W + B·C·kv`。显存频率不随核心 DVFS
  （V100 焊死 877 MHz；Hopper 亦为运行内固定）⇒ 带宽与功率无关 ⇒ **吞吐天花板 `T_max = B/T_mem` 固定**。
2. **计算时间随核心降频膨胀**：`T_comp(x) = C_c·(x^{-p} − 1)`（`x=1` 时恰好为 0，平台精确成立；
  `p` 为**有效**指数，含低频下占用率/延迟隐藏塌缩，故 `p>1` 常见）。

### 3.2 功率侧与显式 `Throughput(P)`

核心频率经 DVFS 决定功率：`P(x) = P_s + χ·x^θ`（θ∈[2,4]，电压随频率上升的幂律合并）。
反解 `x(P) = ((P−P_s)/χ)^{1/θ}`（钳到 ≤1）代回，得**显式解析模型**：

$$\boxed{\;\text{Throughput}(P) \;=\; \frac{B}{\;T_{mem} + C_c\!\left[\left(\dfrac{P-P_s}{\chi}\right)^{-p/\theta}\!\!-\,1\right]}\;}$$

适用域 `P_s < P ≤ P_s + χ`；超过后 `x=1`、吞吐恒为 `B/T_mem`。

### 3.3 三阶段与边界

| 阶段 | 条件 | 机理 | 行为 |
|---|---|---|---|
| **I 计算主导** | `T_comp > T_mem`（P 极低） | 核心被压到标称 20–30%，访存受限任务被挤成计算受限 | 幂律上升 `∝(P−P_s)^{p/θ}` |
| **II 计算-访存交替** | `T_comp ∼ T_mem` | 两项相加、此消彼长；提频的边际收益递减 | 上升但斜率变缓（**旧 min() 在此必然低估**） |
| **III 访存平台** | `T_comp < 5%·T_mem` | 耗时≈`T_mem` 且频率到顶；再加功率只发热 | 平台 `T_max=B/T_mem`，与 P 脱钩 |

边界（解析）：I/II 在 `x₁ = (C_c/(T_mem+C_c))^{1/p}`；II/III 在 `x₂ = (C_c/(0.05·T_mem+C_c))^{1/p}`，
代回 `P(x)` 得边界功率。V100 实测：`P₁ ≈ 87–97 W`（阶段 I 几乎全在 100 W 量程下限以下）、
`P₂ ≈ 109–177 W`（**越访存受限，越低功率就饱和** —— 32k 摘要在 114 W 即进平台）。

### 3.4 天花板定律（可定量预测的部分）

$$\boxed{\;T_{\max} \;=\; \frac{B\cdot BW_{eff}}{\,W + B\cdot C_{eff}\cdot kv\,}\;}$$

- **可加访存量**：权重项 + KV 项。KV 主导时 `T_max → BW_eff/(C·kv)`：**上下文翻倍、平台减半（1/C 律）**；
  权重主导时（GQA 小 KV）平台近似 `∝B`。
- **MHA vs GQA 是主要驱动**：Phi-3（MHA，384 KB/tok）在 C=4096 平台已塌到 109 tok/s，
  Qwen2.5（GQA，28–56 KB/tok）同级上下文平台高一个量级。
- `C_eff = C + steps/2`：测量窗口内 KV 会生长，用流量加权的有效上下文（方法学 v3）。
- `BW_eff` 是**模型×硬件属性**而非常数（详见 §4.4）。

### 3.5 与旧模型的关系（为什么要精化）

旧模型 `T(P) = min(T_{V²f}(P), T_max)` 假设计算与访存**完美重叠**（每步耗时 = 取大）。它抓住了
上升段与平台，但在阶段 II（两者相当、相加才对）**系统性低估吞吐** —— 这正是实测中段残差的来源。
可加模型把 min() 的锐拐点换成物理正确的平滑过渡；两个极端访存受限的 workload（32k 摘要、B=8 分类）
在旧模型下要"饱和特判"，在可加模型里只是 **`C_c` 很小的自然极限**，无需特判。

### 3.6 拟合方法（两步，时钟空间）

利用每个 cap 点随手记录的 `sm_clk` 遥测：
1. 功率侧 `(clk, P)`：网格 `P_s×θ`、最小二乘 `χ`；
2. 吞吐侧 `(clk, T)`：`T_mem` 锚定平台（最高功率 3 点均值的 `B/T_plateau`），网格 `p`、最小二乘 `C_c`；
3. 合成显式 `Throughput(P)`，在功率空间对**全部点**评 R²/相对 RMSE，并做**留一交叉验证**防过拟合。

诚实说明：`P_s↔θ` 部分简并、近平坦曲线上 `p` 弱可辨识（CSV 的 `exponents_railed` 列标记，
对应阶段边界抑制不报）；两步拟合的时钟空间最优不等于功率空间最优（差异 <0.01 R²）。

---

## 4. 实验结果汇总（V100-DGXS-32GB，10 workload × 5 模型）

### 4.1 设定

单卡 V100（250 W cap、HBM2 877 MHz 固定、f_max=1530 MHz）。每 workload 固定 `(S,B)/(C,B)`，
只扫功率 cap [100..250] W（8 点）。**测量方法学 v3**（伪影修复的完整证据链见
[DATA_QUALITY.zh.md](pt_cap_gpu1/portfolio/DATA_QUALITY.zh.md)）：KV 用整段 prefill 建
（chunk=ctx；小块 seeding 让分配器碎片化、GPU 饿死，chunk32 只有 225–233 vs chunk256 的 ~827 tok/s）、
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（消除 ±9% 双模并快 15%）、步数目标窗口
（≥32 步）+ 重复取中位、记录 `ctx_eff`。跨 4 卡参考点一致性 <2%。

### 4.2 主表：10 类 workload 与 decode 两代模型对比

| workload | 应用 | 模型 | decode C×B | 平台 tok/s | 旧 min() R² | **可加 R²** | 相对RMSE | LOO 可加/旧 |
|---|---|---|---|--:|--:|--:|--:|--:|
| chat-phi3 | 对话 | Phi-3-mini (MHA'24) | 256×64 | 825 | 0.928 | **0.986** | 1.7% | 6.8 / 16.0 |
| rag-phi3 | RAG | Phi-3-mini | 1024×32 | 277 | 0.898 | **0.991** | 1.4% | 11.8 / 15.8 |
| code-phi3 | 代码 | Phi-3-mini | 2048×16 | 128 | 0.906 | **0.987** | 1.5% | 24.7 / 14.9 |
| longform-phi3 | 长文 | Phi-3-mini | 4096×8 | 109 | 0.917 | **0.997** | 0.6% | 2.1 / 12.6 |
| summarize-qwen7b | 摘要32k | Qwen2.5-7B (GQA) | 32768×4 | 6.4 | −0.21 | **0.903** | 2.6% | 27.5 / 9.3 |
| translate-qwen3b | 翻译 | Qwen2.5-3B (GQA) | 512×64 | 559 | 0.910 | **0.997** | 0.7% | 10.5 / 14.0 |
| fastchat-qwen15b | 轻对话 | Qwen2.5-1.5B (GQA) | 512×64 | 900 | 0.891 | **0.993** | 0.9% | 26.1 / 12.6 |
| classify-qwen7b | 分类 | Qwen2.5-7B | 256×8 | 149 | 0.440 | 0.632 | 9.8% | 18.2 / 21.2 |
| qwen3chat-4b | 对话'25 | Qwen3-4B-2507 (GQA+QKNorm) | 1024×32 | 129 | 0.719 | **0.987** | 1.1% | 16.6 / 12.6 |
| qwen3think-4b | **推理CoT** | Qwen3-4B-2507 | 8192×8 | 16.0 | 0.712 | **0.916** | 3.2% | 25.4 / 13.9 |

- 对比是**公平的**：旧模型网格已放开收敛（未放开时它被低估最多 −0.9 R²）、含 v1 的饱和特判分支。
- LOO 列为「可加 / 旧」两模型的留一样本外误差：可加模型 5/10 胜。混合的原因是**边界外推伪影**——
  留出最低功率点时可加模型须外推到 `P_s` 以下（钳到近零吞吐 → 巨误差），旧模型的平推插值反而占便宜；
  在曲线弯曲、信息量大的 workload 上（chat/rag/longform/translate/classify）可加模型 LOO 全部占优。
- classify（B=8）是诚实的例外：cap ≥180 W 后实际功耗只有 ~150 W，**cap 不再咬合**，DVFS 调速器
  对欠饱和负载自行浮动，吞吐在平台附近散动（两张卡复现）——属"cap 失效区"的真实行为。
- 图：[fig_decode_models.png](pt_cap_gpu1/portfolio/fig_decode_models.png)（主图，10 面板双模型对比）、
  [fig_portfolio_grid.png](pt_cap_gpu1/portfolio/fig_portfolio_grid.png)（prefill+decode 总览）。

### 4.3 三阶段参数（可加模型，V100）

| workload | T_mem (ms) | C_c (ms) | p | P₁ (W) | P₂ (W) |
|---|--:|--:|--:|--:|--:|
| chat-phi3 | 77.5 | 8.8 | 2.8 | 96 | 177 |
| rag-phi3 | 115.4 | 6.3 | 3.7 | 95 | 167 |
| code-phi3 | 125.1 | 12.4 | 2.9 | 97 | 169 |
| longform-phi3 | 73.7 | 12.6 | 2.2 | 94 | 163 |
| summarize-qwen7b | 625.0 | 0.6 | 8.7† | 97 | 114 |
| translate-qwen3b | 114.5 | 13.6 | 2.6 | 94 | 165 |
| fastchat-qwen15b | 71.1 | 5.6 | 3.0 | 95 | 154 |
| qwen3chat-4b | 247.7 | 1.5 | 6.6† | 95 | 126 |
| qwen3think-4b | 500.0 | 2.9 | 6.3† | 97 | 129 |

（† 近平坦曲线（`C_c/T_mem` 极小）上 p 弱可辨识：数值为有效指数、非物理常数，只有 `T_mem`/平台可信 —— 见 §3.6。）
规律清晰：**访存越重（`C_c/T_mem` 越小），P₂ 越靠前** —— 摘要 114 W 就饱和，对话要 177 W。

### 4.4 天花板验证与有效带宽

拟合平台 vs `B·BW_eff/(W+B·C_eff·kv)` 预测，按模型标定一个 `BW_eff`：**10 点跨 ~140×
（6.4→900 tok/s）贴 y=x**（[fig_tmax_validation.png](pt_cap_gpu1/portfolio/fig_tmax_validation.png)）。

| 模型 | BW_eff（实测标定） | 说明 |
|---|--:|---|
| Phi-3-mini | ~186 GB/s | chat/rag/code 隐含带宽 165–187 一致 → 验证可加 `D_mem` 分解 |
| Qwen2.5-7B | ~260 GB/s | 大权重流带宽利用率最高 |
| Qwen2.5-3B / 1.5B | 65 / 57 GB/s | 小 hidden、多 launch，利用率低 |
| Qwen3-4B-2507 | ~52 GB/s | 36 层小矩阵，同小 Qwen 一档 |

`BW_eff` 是**模型×硬件×引擎**属性（eager HF 的逐步 kernel 启动 + DynamicCache 每步全 KV
cat 拷贝——每步实际访存 ≈ 权重+~3×KV——都被标定吸收）。显著残差：32k 超长上下文的稀疏 KV 读
再打对折（summarize 落在预测线下方）。

### 4.5 能效与容量规划要点

- prefill 能效单峰（V100 峰 ≈40 tok/J @155 W）；decode 峰在拐点附近（≈4–5 tok/J），过拐点单调降。
- **同功率 prefill:decode 能效 ≈ 10:1**（权重复用 vs 每步重读）。
- 功率封顶的操作含义：**decode 卡可以放心压到 P₂ 附近**（平台内免费省电）；prefill 卡压 cap 直接
  换吞吐，交互应用受 TTFT SLO 约束。机架级劈分/回本分析见 [rack_power_capping/](rack_power_capping/)，
  按应用的 P:D 比例见 [workload_analysis/](workload_analysis/)。
- 推理型负载（长 CoT，qwen3think）是 2025 年的关键新形态：同一模型上下文 1024→8192，平台
  129→16 tok/s —— **decode 极重 + 长 KV 在功率规划里代价极高**。

### 4.6 诚实的偏差清单

1. `∝B` 偏弱：同 `D_mem` 的 rag/code/longform 平台比 2.2:1.2:1（理论 4:2:1）——低 batch 下并发
   流不足以打满带宽，decode 偏延迟受限。
2. `BW_eff` 随负载浮动 52–260 GB/s（§4.4）——公式抓准访存**量**，流过字节的有效**带宽**本身可变。
3. classify 的"cap 失效区"散动（§4.2）。
4. p/θ 是有效指数，近平坦曲线上弱可辨识（已在 CSV 标记、边界抑制）。
5. 全部为未热降频（冷启短测 + 热门控）口径；持续重载另见 [schedule_lab/thermal_throttle/](schedule_lab/thermal_throttle/)。

---

## 5. 跨硬件验证：H200（下一步）

一键测量包已就绪并在 V100 上端到端验证（[H200_操作手册.md](H200_操作手册.md) /
[pt_cap_gpu1/portfolio/OFFLINE_H200.zh.md](pt_cap_gpu1/portfolio/OFFLINE_H200.zh.md)）。预期：
- **形状复现**：prefill 凸 V²f、decode 三阶段 + 平台（HBM3e 运行内同样固定频率 → 平台依旧存在）；
- **量级平移**：平台 ∝ 带宽（4.8 TB/s vs 0.9），约高一个量级；阶段边界整体右移（P_s ~200 W 量级）；
- 能量法窗口功率 + 降频门控已内建（Hopper 的 `power.draw` 是 ~1s 滑动平均，不可直接用）。

## 6. 文件地图

| 内容 | 文件 |
|---|---|
| **本总纲**（理论+结果唯一入口） | `MODEL_AND_RESULTS.zh.md` |
| prefill 显式模型推导（与 decode 同构） | [pt_cap_gpu1/prefill_model_theory.md](pt_cap_gpu1/prefill_model_theory.md) |
| decode 三阶段推导细节 + 单卡标定 | [pt_cap_gpu1/decode_model_theory.md](pt_cap_gpu1/decode_model_theory.md) |
| 测量方法学 v3 证据链 | [pt_cap_gpu1/portfolio/DATA_QUALITY.zh.md](pt_cap_gpu1/portfolio/DATA_QUALITY.zh.md) |
| portfolio 结果细节 | [pt_cap_gpu1/portfolio/RESULTS.zh.md](pt_cap_gpu1/portfolio/RESULTS.zh.md) |
| 10 workload 配置 | [pt_cap_gpu1/portfolio/portfolio.py](pt_cap_gpu1/portfolio/portfolio.py) |
| 原始数据（V100 v3） | `pt_cap_gpu1/portfolio/data/*.csv` + `meta.json`（元数据为事后补记：v3 采集早于 meta 功能） |
| 主图：decode 双模型对比 | `pt_cap_gpu1/portfolio/fig_decode_models.png` |
| prefill 统一模型 vs V²f | `pt_cap_gpu1/portfolio/fig_prefill_models.png` + `prefill_model_compare.csv` |
| 总览（统一模型 vs 全部实测点） / 天花板验证 | `fig_portfolio_grid.png` / `fig_tmax_validation.png` |
| 拟合参数表 | `decode_model_compare.csv` / `portfolio_fits.csv` |
| 一键测量（H200/任意卡） | `pt_cap_gpu1/portfolio/run_all.sh`（`--smoke` 验机） |
| 单模型早期基线（**legacy**：变-batch frontier + 旧 min() 模型，仅作历史对照） | `pt_cap_gpu1/fig_theory_vs_measured.png` |
| 机架级规划 / 负载分类 | [rack_power_capping/](rack_power_capping/) / [workload_analysis/](workload_analysis/) |
