# LLM 推理的功率↔吞吐模型与全部实验结果 —— 总纲

> **本文件是唯一权威入口，自成体系**：完整给出 prefill 与 decode 两个阶段的理论模型
>（统一构建、完整推导、标定数值），并汇总全部实验结果（V100 上 10 类 workload × 5 个模型）。
> 子文档：[pt_cap_gpu1/prefill_model_theory.md](pt_cap_gpu1/prefill_model_theory.md) /
> [pt_cap_gpu1/decode_model_theory.md](pt_cap_gpu1/decode_model_theory.md)（单阶段独立成篇）、
> [pt_cap_gpu1/portfolio/DATA_QUALITY.zh.md](pt_cap_gpu1/portfolio/DATA_QUALITY.zh.md)（测量方法学）、
> [H200_操作手册.md](H200_操作手册.md)（跨硬件验证的执行手册）。

---

## 0. 一页速览

**建模对象是 `P ↔ T`**（功率 ↔ token 吞吐）。旋钮 = **功率上限**（`nvidia-smi -pl`）：设定后 GPU 自选
能维持的核心频率，功率与吞吐同时移动；频率只是中间机制，不出现在最终模型里。

| 阶段 | 瓶颈（roofline） | 显式模型 `Throughput(P)` | 形状 |
|---|---|---|---|
| **prefill** | 计算受限（`I ≫ I*`） | $T_{f_{max}}\left(\frac{P-P_{static}}{\chi}\right)^{p/\theta}$ | **单段**幂律、凹，量程内无平台 |
| **decode** | 访存受限（`I ≪ I*`） | $\dfrac{B}{T_{mem}+C\left[\left(\frac{P-P_{static}}{\chi}\right)^{-p/\theta}-1\right]}$ | **三阶段**：幂律升 → 边际递减 → 带宽平台 $B/T_{mem}$ |

两式来自**同一条每 token 耗时定律 + 同一个 DVFS 功率模型**（§2）；唯一差别是 decode 有
与频率无关的访存地板 $T_{mem}$，prefill 没有——**阶段结构完全由这个地板的有无决定**。

**验证结论（V100，10 类 workload × 5 模型，功率 cap 扫描）**：
- prefill：9/10 R² ≥0.92，且 **10 个 workload 全部拟出 `p≈1`**（吞吐∝频率的计算受限机制指数，
  与锁频 DVFS 直测 `T∝f^0.90` 一致）；
- decode 三阶段：9/10 R² = **0.90–0.997**（相对 RMSE 0.6–3.2%），全面优于旧的完美重叠
  近似 `min(V²f, T_max)`（R² −0.21–0.93）；
- 平台量级 `T_max = B·BW_eff/(权重 + B·C_eff·kv/tok)` 跨 **~140×**（6.4→900 tok/s）成立。

**能效直觉**：同功率下 prefill 每焦耳产出 ≈ **10×** decode（权重复用 vs 每步重读全部权重）；
prefill 能效峰在中低功率（V100 ≈40 tok/J @ ~155 W），decode 过拐点后能效单调降——这是功率
封顶对 decode 几乎免费、对 prefill 昂贵的根源，也是机架级规划的物理基础。

---

## 1. 物理基础：两条功率原语 + roofline

### 1.1 两条功率原语（一切的来源）

**(a) 逻辑/计算动态功率**。开关的 CMOS 阵列耗散 `P_logic = α·C·V²·f`。可靠翻转要求电压随频率上升
（活跃区 `V ≈ V₀ + γf`），故**每操作能量 `E_op ∝ V²` 随速度上升**——算得越快、每次运算越贵。

**(b) 访存/搬运功率**。搬 1 bit 耗散近似**固定**能量 `E_bit`（充放固定的线与单元电容；HBM 的数据
时钟不随核心 DVFS 调压），故 `P_mem = E_bit·BW`，**每 bit 能量与速度无关**。

> **一切不对称的根源**：计算的单位能量随速度涨（∝V²），访存的单位能量不涨。

### 1.2 roofline：两个阶段各卡在哪

每 token 两阶段 FLOPs 几乎相同，决定性差异在**权重复用**（算术强度 `I` 相对脊点 `I* = Φ/β`）：

| | FLOPs/步 | 字节/步 | 算术强度 | 判定 |
|---|---|---|---|---|
| prefill | `≈2N·(B·S)` | 每个权重 tile 读一次、被全部 `B·S` 个位置复用 | `I ≫ I*` | **计算受限** → `T ∝ f_sm` |
| decode | `≈2N·B` | `D_mem = W + B·C·kv`（每步重读全部权重 + B 条序列的 KV） | `I ≪ I*` | **访存受限** → `T ∝ BW` |

（`W`=权重字节，`kv`=每 token KV 字节 `2L·n_kv·h·b`，`C`=上下文长，`B`=batch，`S`=prompt 长。）

### 1.3 理想极限（第一性原理的"渐近律"）

在 `V ∝ f` 的理想区（无功率墙、无热限）：prefill `P ≈ P₀ + k_c·T³`（立方，每 token 能量∝T²）；
decode `P ≈ P₀ + k_m·T`（线性，每 token 能量≈常数）。
**实测校验（V100 锁频 DVFS 510→1530 MHz）**：机制成立——prefill `T∝f^0.90`（R²=0.99）、
decode `T∝f^0.26`（频率×3 只换 ×1.37 吞吐）；但立方指数不干净（V100 中段 V-f 平坦，指数对
P₀ 简并：P₀=44 W 时 γ≈1.5，90 W 时 γ≈3.0）；decode 线性律经 batch 旋钮确认
（`P=111+0.190T`，R²=0.996）。⇒ 理想律是渐近极限；**§2–4 的实测形式才是用于拟合与预测的模型**。

---

## 2. 统一构建：一条耗时定律 + 一个 DVFS 功率模型

### 2.1 每 token 耗时定律

单 token（步）耗时由**访存**与**计算**两部分**相加**（不是取 max）：

$$
\text{Time per Token} = T_{mem} + T_{comp}
= \frac{D_{mem}}{BW(f_{mem})} + \frac{O_{comp}}{OPS(f_{sm})}
$$

| 符号 | 含义 |
|---|---|
| $D_{mem}$ | 搬运的数据量（模型权重 + KV Cache / 激活） |
| $BW(f_{mem})$ | 显存带宽，取决于显存控制器频率 $f_{mem}$ |
| $O_{comp}$ | 单步计算量（FLOPs） |
| $OPS(f_{sm})$ | 实际算力，$OPS \propto f_{sm}^{\,p}$（$p$ 为有效指数，含低频占用率塌缩） |

**关键约束**：DVFS/功率 cap 调的是 **SM 频率 $f_{sm}$**，而**显存控制器频率 $f_{mem}$ 基本固定**
（V100 焊死 877 MHz；Hopper 运行内固定）。于是：

$$
\boxed{\,BW(f_{mem})=\text{const}\;\Rightarrow\;T_{mem}=\text{const}\,}
$$

访存时间是一条**与 SM 频率无关的常数地板**；SM 频率只能改变 $T_{comp}$。
两个阶段的全部差别就在这条地板的大小：decode 的 $T_{mem}$ 是主项（§4），
prefill 的 $T_{mem}\ll T_{comp}$ 全程可忽略（§3）。

### 2.2 DVFS 功率模型与反解

SM 频率经 DVFS 决定功率（电压随频率非线性上升，动态功耗并为幂律）：

$$
P(f_{sm}) = P_{static} + \chi\left(\frac{f_{sm}}{f_{max}}\right)^{\theta},\qquad \theta\in[2,4]
$$

吞吐与功率都只由 $f_{sm}$ 驱动，故「吞吐 vs 功率」是一条以 $f_{sm}$ 为参数的曲线。
把功率**反解出频率**再代入吞吐，即得显式模型。记 $x=f_{sm}/f_{max}$：

$$
x(P)=\left(\frac{P-P_{static}}{\chi}\right)^{1/\theta},\qquad P_{static}<P\le P_{static}+\chi=P(f_{max})
$$

**适用域**：功率上限 $\ge P(f_{max})$ 时频率已顶到 $f_{max}$（$x{=}1$），须把 $x$ 钳到 1
（$x>1$ 分支非物理，不可代入）。$P_{static}\leftrightarrow\theta$ 部分简并——单独数值仅供参考，
合成出的 $\text{Throughput}(P)$ 不受影响。

---

## 3. Prefill 理论模型：$T_{mem}\to 0$ → 单段显式幂律

### 3.1 推导

prefill 权重复用（`I≫I*`）使 $T_{mem}\ll T_{comp}$ **在整个 DVFS 范围成立**：

$$
\text{Time}\approx \frac{O_{comp}}{OPS(f_{sm})}\;\Rightarrow\;\text{Throughput}(x)= T_{f_{max}}\cdot x^{\,p}
$$

代入 §2.2 的 $x(P)$，得显式模型：

$$
\boxed{\;\text{Throughput}(P) = T_{f_{max}}\left(\frac{P-P_{static}}{\chi}\right)^{p/\theta},
\qquad P_{static}<P\le P_{static}+\chi\;}
$$

- **单段**幂律，指数 $p/\theta\in(0,1)$（实测 0.33–0.99）⇒ 曲线**凹**：低功率端每瓦换得多、
  高功率端边际递减；
- **无阶段、无带宽平台**——没有 $T_{mem}$ 地板，就没有转折。唯一的饱和是频率顶 $x=1$
  （吞吐到 $T_{f_{max}}$），且重载 prefill 的 cap 量程通常在 $P(f_{max})$ 之下，量程内不出现；
- 等价反函数是纯幂律 $P(T)=P_{static}+\chi (T/T_{f_{max}})^{\theta/p}$；旧文档的 V²f 形式
  $P(T)=P_0+\kappa T(1+\rho T)^2$ 是同一物理在仿射电压 $V=V_0+\gamma f$ 下的另一种参数化，
  保留作基线（拟合能力相当，见 §3.2）。

**能效推论（闭式）**：$E(P)=T/P\propto (P-P_{static})^{a}/P$（$a=p/\theta$），峰在

$$
P^* = \frac{P_{static}}{1-a}
$$

V100+Phi-3 实测峰 ≈**40 tok/J @ ~155 W**（62% TDP），与 $P_{static}\approx70{-}90$ W、
$a\approx0.4{-}0.5$ 一致。能效敏感的部署把 prefill 压到峰附近，代价是 TTFT——交互应用的
cap 下界由延迟 SLO 决定。

### 3.2 标定与验证（V100，10 类 workload，v3 数据）

两步时钟空间拟合（功率侧 $P_{static},\chi,\theta$；吞吐侧对数最小二乘出 $p,T_{f_{max}}$；
合成 $\text{Throughput}(P)$ 在功率空间打分），与 V²f 基线同口径对比：

| workload | S×B | 统一 R² | V²f R² | $p$ | | workload | S×B | 统一 R² | V²f R² | $p$ |
|---|---|--:|--:|--:|---|---|---|--:|--:|--:|
| chat-phi3 | 512×8 | 0.923 | 0.972 | 1.32 | | translate-qwen3b | 512×8 | 0.962 | 0.984 | 1.02 |
| rag-phi3 | 4096×2 | 0.992 | 0.991 | 0.92 | | fastchat-qwen15b | 512×16 | 0.950 | 0.984 | 1.01 |
| code-phi3 | 2048×4 | 0.992 | 0.992 | 0.97 | | classify-qwen7b | 2048×4 | 0.986 | 0.991 | 0.91 |
| longform-phi3 | 256×16 | 0.957 | 0.956 | 1.19 | | qwen3chat-4b | 512×8 | 0.871 | 0.849 | 1.18 |
| summarize-qwen7b | 4096×2 | 0.986 | 0.995 | 0.94 | | qwen3think-4b | 2048×4 | 0.979 | 0.974 | 0.81 |

**核心验证是 p 列**：$p=0.81$–$1.32$（中位 ≈0.99）——10 个 workload 从 cap 扫描**独立复原**
"计算受限 ⇒ 吞吐∝频率"的机制指数，与锁频 DVFS 直测的 $T\propto f^{0.90}$ 一致。
两种参数化拟合能力相当（统一模型胜在框架一致与参数可解释；V²f 样本内略优）。
qwen3chat 两者皆偏低（0.85/0.87）：最低 cap 点实抽 72 W、时钟 441 MHz，落在电压地板区之外。
图：[fig_prefill_models.png](pt_cap_gpu1/portfolio/fig_prefill_models.png)。

---

## 4. Decode 理论模型：$T_{mem}$ 地板 → 三阶段

### 4.1 核心物理逻辑

$$
\text{Throughput} = \frac{\text{Batch Size}}{\text{Time per Token}},\qquad
\text{Time per Token} = T_{mem} + T_{comp}
$$

decode 每步重读全部权重 + KV（`I≪I*`），$T_{mem}$ 是主项。由 §2.1 的关键约束：

$$
\boxed{\,BW(f_{mem})=\text{const}\;\Rightarrow\;T_{mem}=\text{const}\;\Rightarrow\;
\text{吞吐天花板}=\frac{\text{Batch Size}}{T_{mem}}\,}
$$

访存地板与 SM 频率无关，**唯一决定吞吐上限**；SM 频率只能改变 $T_{comp}$。

### 4.2 显式的吞吐–功率关系

代入 $\text{Throughput}=B/(T_{mem}+T_{comp})$，其中 $T_{comp}=C\,(x^{-p}-1)$
（$p$ 源自算力降级 $OPS\propto f_{sm}^{\,p}$；$x{=}1$ 时 $T_{comp}{=}0$，平台精确成立）：

$$
\boxed{\;\text{Throughput}(P)=\frac{B}{\,T_{mem}+C\!\left[\left(\dfrac{P-P_{static}}{\chi}\right)^{-p/\theta}-1\right]}
=\frac{B}{\,(T_{mem}-C)+C\left(\dfrac{\chi}{P-P_{static}}\right)^{p/\theta}}\;}
$$

**两端渐近**（对应下文三阶段）：

$$
\underbrace{\text{Throughput}\approx\frac{B}{C}\left(\frac{P-P_{static}}{\chi}\right)^{p/\theta}\propto (P-P_{static})^{p/\theta}}_{\text{低功率：}T_{comp}\gg T_{mem}\text{，指数 }p/\theta\lesssim1\text{，近似线性}}
\qquad
\underbrace{\text{Throughput}\to \frac{B}{T_{mem}}}_{\text{高功率：}T_{comp}\to0\text{，平台}}
$$

### 4.3 吞吐随功率从低到高的三个阶段

| 阶段 | 功率/频率条件 | 机理 | 吞吐–功率行为 |
|---|---|---|---|
| **1 伪计算受限** | $P$ 极低（$T_{comp}>T_{mem}$，$f_{sm}$ 被压到标称 20–30%） | $OPS(f_{sm})$ 急剧下降（频率↓且占用率塌缩），$T_{comp}$ 大幅上升、超过 $T_{mem}$，访存受限任务被挤成计算受限 | 幂律上升 $\propto(P-P_{static})^{p/\theta}$，指数 $\lesssim1$，**近似线性陡升** |
| **2 边际递减** | 中等功率（$T_{comp}\sim T_{mem}$） | $f_{sm}$ 越过临界点后 $T_{comp}$ 收缩，任务回归访存受限；提频对缩短 $T_{mem}+T_{comp}$ 贡献渐小，功率主要喂给 $f_{sm}^{\theta}$ 动态功耗 | 持续上升但**斜率明显变缓**（旧 min() 在此系统性低估——"完美重叠"假设的必然） |
| **3 访存平台** | 高功率/满载（$T_{comp}<5\%\,T_{mem}$，$f_{sm}\to f_{max}$） | $T_{comp}$ 可忽略，总耗时 $\approx T_{mem}$；且 $f_{sm}$ 已顶 $f_{max}$，再加功率只升压发热不升频 | **平台** $=B/T_{mem}$，与 $P$ 脱钩 |

边界（解析）：I/II 在 $x_1=(C/(T_{mem}+C))^{1/p}$，II/III 在 $x_2=(C/(0.05\,T_{mem}+C))^{1/p}$，
代回 $P(x)$ 得边界功率。

### 4.4 天花板定律（可定量预测的部分）

$$
\boxed{\;T_{\max} = \frac{B\cdot BW_{eff}}{\,W + B\cdot C_{eff}\cdot kv\,}\;}
$$

- **可加访存量**：KV 主导时 $T_{\max}\to BW_{eff}/(C\cdot kv)$——**上下文翻倍、平台减半（1/C 律）**；
  权重主导时（GQA 小 KV）平台近似 ∝B。
- **MHA vs GQA 是主要驱动**：Phi-3（MHA，384 KB/tok）在 C=4096 平台已塌到 109 tok/s，
  Qwen2.5（GQA，28–56 KB/tok）同级上下文高一个量级。
- $C_{eff}=C+\text{steps}/2$：测量窗口内 KV 生长，用流量加权的有效上下文（方法学 v3）。
- $BW_{eff}$ 是**模型×硬件×引擎**属性而非常数（§5.4）。

### 4.5 单卡标定（V100 + Phi-3，13 点锁频扫描，batch=96）

| 量 | 值 |
|---|---|
| $T_{mem}$（常数访存地板） | 149.3 ms（有效带宽 ~116 GB/s ≈峰值 13%：访存受限且延迟受限） |
| $C,\ p$（$T_{comp}=C(x^{-p}{-}1)$） | 30.8 ms，$p=1.84$ |
| $P_{static},\ \chi,\ \theta$ | 50 W，155.5 W，2.15 |
| **天花板** $B/T_{mem}$ | **643 tok/s**（= 实测最大，精确命中） |
| 阶段分界 | I/II：$P_1\approx70$ W（585 MHz）；II/III：$P_2\approx171$ W（1359 MHz） |
| 阶段 1 指数 $p/\theta$ | 0.855（近线性） |
| 拟合优度 | 频率空间 R²=0.97；功率 R²=0.99；功率空间 R²=0.956 |

双路独立推导结果一致，且 $\text{Throughput}(P)$ 是参数模型的精确解析逆——把 $P(f)$ 回代可
机器精度复现 $\text{Throughput}(f)$（差 $10^{-13}$）。细节与注意事项见
[decode_model_theory.md](pt_cap_gpu1/decode_model_theory.md) §四。

### 4.6 跨 workload 验证与旧模型对比

与旧的完美重叠 roofline 近似 $T=\min(T_{V^2f},\,T_{max})$ 对比（公平基线：旧模型网格已放开
收敛、含饱和特判分支）：可加三阶段在 9/10 上 R²=0.90–0.997（相对 RMSE 0.6–3.2%），旧模型
−0.21–0.93（3.3–12.1%）；留一交叉验证 5/10 胜（败点均为最低功率点的域外外推伪影，曲线弯曲
的 workload 上全部占优）。两个极端访存受限的 workload（32k 摘要、B=8 分类）是同一定律的
小 $C$ 极限——无需特判。注意 $p,\theta$ 为**有效**指数（近平坦曲线上弱可辨识，CSV
`exponents_railed` 列标记）；且 transformers 的 DynamicCache 每步对全 KV 做 cat（读+写拷贝），
每步实际访存 ≈ 权重+~3×KV，被 $T_{mem}$ 标定自然吸收。
图：[fig_decode_models.png](pt_cap_gpu1/portfolio/fig_decode_models.png)。

---

## 5. 实验结果汇总（V100-DGXS-32GB，10 workload × 5 模型）

### 5.1 设定

单卡 V100（250 W cap、HBM2 877 MHz 固定、f_max=1530 MHz）。每 workload 固定 `(S,B)/(C,B)`，
只扫功率 cap [100..250] W（8 点）。**测量方法学 v3**（伪影修复的完整证据链见
[DATA_QUALITY.zh.md](pt_cap_gpu1/portfolio/DATA_QUALITY.zh.md)）：KV 用整段 prefill 建
（chunk=ctx；小块 seeding 让分配器碎片化、GPU 饿死，chunk32 只有 225–233 vs chunk256 的
~827 tok/s）、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（消除 ±9% 双模并快 15%）、
步数目标窗口（≥32 步）+ 重复取中位、记录 `ctx_eff`。跨 4 卡参考点一致性 <2%。

### 5.2 主表：10 类 workload 与 decode 两代模型对比

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
  留出最低功率点时可加模型须外推到 $P_{static}$ 以下（钳到近零吞吐 → 巨误差），旧模型的平推
  插值反而占便宜；曲线弯曲、信息量大的 workload 上可加模型 LOO 全部占优。
- classify（B=8 欠饱和）：cap≥180 W 后实际功耗仅 ~150 W，**cap 不再咬合**，调速器自行浮动——
  "cap 失效区"的真实行为（两张卡复现），如实呈现。
- 总览图（统一模型 vs 全部实测点）：[fig_portfolio_grid.png](pt_cap_gpu1/portfolio/fig_portfolio_grid.png)。

### 5.3 三阶段参数（可加模型，V100）

| workload | $T_{mem}$ (ms) | $C$ (ms) | $p$ | $P_1$ (W) | $P_2$ (W) |
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

（† 近平坦曲线（$C/T_{mem}$ 极小）上 $p$ 弱可辨识：数值为有效指数、非物理常数，只有
$T_{mem}$/平台可信——见 §4.6。）规律清晰：**访存越重（$C/T_{mem}$ 越小），$P_2$ 越靠前**——
摘要 114 W 就饱和，对话要 177 W。

### 5.4 天花板验证与有效带宽

拟合平台 vs $B\cdot BW_{eff}/(W+B\cdot C_{eff}\cdot kv)$ 预测，按模型标定一个 $BW_{eff}$：
**10 点跨 ~140×（6.4→900 tok/s）贴 y=x**
（[fig_tmax_validation.png](pt_cap_gpu1/portfolio/fig_tmax_validation.png)）。

| 模型 | BW_eff（实测标定） | 说明 |
|---|--:|---|
| Phi-3-mini | ~186 GB/s | chat/rag/code 隐含带宽 165–187 一致 → 验证可加 $D_{mem}$ 分解 |
| Qwen2.5-7B | ~260 GB/s | 大权重流带宽利用率最高 |
| Qwen2.5-3B / 1.5B | 65 / 57 GB/s | 小 hidden、多 launch，利用率低 |
| Qwen3-4B-2507 | ~52 GB/s | 36 层小矩阵，同小 Qwen 一档 |

$BW_{eff}$ 是**模型×硬件×引擎**属性（eager HF 的逐步 kernel 启动 + DynamicCache 每步全 KV
cat 都被标定吸收）。显著残差：32k 超长上下文的稀疏 KV 读再打对折（summarize 落在预测线下方）。

### 5.5 能效与容量规划要点

- prefill 能效单峰（闭式 $P^*=P_{static}/(1-p/\theta)$；V100 峰 ≈40 tok/J @155 W）；
  decode 峰在拐点 $P_2$ 附近（≈4–5 tok/J），过拐点单调降。
- **同功率 prefill:decode 能效 ≈ 10:1**。
- 功率封顶的操作含义：**decode 卡放心压到 $P_2$ 附近**（平台内免费省电）；prefill 卡压 cap
  直接换吞吐，受 TTFT SLO 约束。机架级规划见 [rack_power_capping/](rack_power_capping/)：
  **10 个实测 workload 归入 6 个应用类，每类一张机架配方**（含物理插槽上限约束）在
  [rack_power_capping/v100/WORKLOADS.zh.md](rack_power_capping/v100/WORKLOADS.zh.md)；
  真实 trace 的 P:D 统计见 [workload_analysis/](workload_analysis/)。
- 推理型负载（长 CoT）：同一模型 C 1024→8192，平台 129→16 tok/s——**decode 极重 + 长 KV
  的 2025 负载形态在功率规划里代价极高**。

### 5.6 诚实的偏差清单

1. `∝B` 偏弱：同 $D_{mem}$ 的 rag/code/longform 平台比 2.2:1.2:1（理论 4:2:1）——低 batch
   下并发流不足以打满带宽，decode 偏延迟受限。
2. $BW_{eff}$ 随负载浮动 52–260 GB/s（§5.4）——公式抓准访存**量**，有效**带宽**本身可变。
3. classify 的"cap 失效区"散动（§5.2）。
4. $p,\theta$ 是有效指数，近平坦曲线上弱可辨识（CSV 标记、边界抑制）。
5. 全部为未热降频（冷启短测 + 热门控）口径；持续重载见 [schedule_lab/thermal_throttle/](schedule_lab/thermal_throttle/)。

---

## 6. 跨硬件验证：H200（下一步）

一键测量包已就绪并在 V100 上端到端验证（[H200_操作手册.md](H200_操作手册.md)）。预期：
- **形状复现**：prefill 单段凹幂律、decode 三阶段 + 平台（HBM3e 运行内同样固定频率）；
- **量级平移**：平台 ∝ 带宽（4.8 TB/s vs 0.9），约高一个量级；$P_{static}$ ~200 W 量级、
  阶段边界整体右移；
- 能量法窗口功率 + 降频门控已内建（Hopper 的 `power.draw` 是 ~1s 滑动平均，不可直接用）。

## 7. 文件地图

| 内容 | 文件 |
|---|---|
| **本总纲**（完整理论+结果） | `MODEL_AND_RESULTS.zh.md` |
| prefill / decode 独立成篇（同一构建） | [prefill_model_theory.md](pt_cap_gpu1/prefill_model_theory.md) / [decode_model_theory.md](pt_cap_gpu1/decode_model_theory.md) |
| 测量方法学 v3 证据链 | [DATA_QUALITY.zh.md](pt_cap_gpu1/portfolio/DATA_QUALITY.zh.md) |
| portfolio 结果细节 | [RESULTS.zh.md](pt_cap_gpu1/portfolio/RESULTS.zh.md) |
| 10 workload 配置 / 共享拟合库 | `portfolio/portfolio.py` / `portfolio/fitlib.py` |
| 原始数据（V100 v3） | `portfolio/data/*.csv` + `meta.json`（元数据为事后补记） |
| 总览（统一模型 vs 全部实测点） | `portfolio/fig_portfolio_grid.png` |
| prefill / decode 新旧模型对比 | `fig_prefill_models.png` / `fig_decode_models.png` + 对应 CSV |
| 天花板验证 | `portfolio/fig_tmax_validation.png` |
| 单模型早期基线（**legacy**：变-batch frontier + 旧 min()） | `pt_cap_gpu1/fig_theory_vs_measured.png` |
| 一键测量（H200/任意卡） | `portfolio/run_all.sh`（`--smoke` 验机） |
| 机架级规划（V100，按 workload 类别的配方 + 经济性） | [rack_power_capping/v100/](rack_power_capping/v100/)，主文档 [WORKLOADS.zh.md](rack_power_capping/v100/WORKLOADS.zh.md) |
| 真实 trace 的负载分类统计 | [workload_analysis/](workload_analysis/) |
