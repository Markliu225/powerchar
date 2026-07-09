# 预填充阶段：功率 → 吞吐 的显式理论模型

> 总纲（prefill+decode 理论与全部结果）见 [../MODEL_AND_RESULTS.zh.md](../MODEL_AND_RESULTS.zh.md)；
> 本文件与 [decode_model_theory.md](decode_model_theory.md) **同构**：同一条每 token 耗时定律、
> 同一个 DVFS 功率模型、同样反解功率得显式模型——prefill 是 $T_{mem}\to 0$ 的退化情形，
> **单段、无阶段划分**。

> **对象**：预填充（prefill），固定 batch $B$ 与提示长度 $S$；通过功率上限压 SM 频率 $f_{sm}$。
> **目标**：给出**吞吐量关于功率的显式解析模型**，并解释其单段凹曲线机理。

配套拟合脚本 [`portfolio/plot_prefill_models.py`](portfolio/plot_prefill_models.py) /
图 [`portfolio/fig_prefill_models.png`](portfolio/fig_prefill_models.png)。

---

## 一、核心物理逻辑与决定因素

Token 吞吐量：

$$
\text{Throughput} = \frac{\text{每次前向的 token 数 } B\cdot S}{\text{每次前向耗时}}
$$

每次前向耗时与 decode 同一条定律——**访存与计算相加**：

$$
\text{Time} = T_{mem} + T_{comp}
= \frac{D_{mem}}{BW(f_{mem})} + \frac{O_{comp}}{OPS(f_{sm})}
$$

| 符号 | 含义 |
|---|---|
| $D_{mem}$ | 搬运数据量（权重读一次 + 激活） |
| $O_{comp}$ | 前向计算量 $\approx 2N\cdot B S$ FLOPs |
| $OPS(f_{sm})$ | 实际算力，$OPS \propto f_{sm}^{\,p}$（$p$ 为有效指数） |

**与 decode 的唯一差别是算术强度**：prefill 每个权重 tile 读一次、被全部 $B\cdot S$ 个位置复用，
$I \gg I^*$（脊点），故在**整个 DVFS 范围内** $T_{comp} \gg T_{mem}$。由此本模型的核心：

$$
\boxed{\,T_{mem}\ll T_{comp}\ \forall f_{sm}\;\Rightarrow\;\text{Time}\approx \frac{O_{comp}}{OPS(f_{sm})}\;\Rightarrow\;\text{Throughput}= T_{f_{max}}\cdot x^{\,p},\quad x=\frac{f_{sm}}{f_{max}}\,}
$$

吞吐**全程跟随核心频率**（$T\propto f^p$，$p\approx 1$）；没有与频率无关的常数地板，
所以**没有平台、没有阶段划分**——这正是它与 decode 三阶段曲线的全部区别。

---

## 二、DVFS 功率模型与显式的吞吐–功率关系

功率模型与 decode **完全相同**：

$$
P(f_{sm}) = P_{static} + \chi\left(\frac{f_{sm}}{f_{max}}\right)^{\theta},\qquad
x(P)=\left(\frac{P-P_{static}}{\chi}\right)^{1/\theta}
$$

反解代入吞吐，得显式模型：

$$
\boxed{\;\text{Throughput}(P) = T_{f_{max}}\left(\frac{P-P_{static}}{\chi}\right)^{p/\theta},
\qquad P_{static}<P\le P_{static}+\chi\;}
$$

- **单段幂律**，指数 $p/\theta \in (0,1)$（实测 0.33–0.99）$\Rightarrow$ 曲线**凹**：低功率端每瓦
  换来的吞吐多，高功率端边际递减——与实测形状一致。
- **适用域**：$P > P_{static}+\chi$ 时频率已顶 $f_{max}$（$x$ 钳到 1），吞吐饱和于 $T_{f_{max}}$。
  这是 prefill 唯一的"天花板"，且是**频率顶**而非带宽顶；重载 prefill 的 cap 范围通常在
  $P(f_{max})$ 之下，故量程内不出现平台。
- 等价的反函数形式是纯幂律 $P(T) = P_{static} + \chi\,(T/T_{f_{max}})^{\theta/p}$。旧文档的
  V²f 形式 $P(T)=P_0+\kappa T(1+\rho T)^2$ 是同一物理在仿射电压 $V=V_0+\gamma f$ 下的另一种
  参数化——两者拟合能力相当（见 §四），本构建的价值在于**与 decode 共用同一套显式框架与参数含义**。

**能效推论**（闭式）：$E(P)=\text{Throughput}/P \propto (P-P_{static})^{p/\theta}/P$，
令 $a=p/\theta$，能效峰在

$$
P^* = \frac{P_{static}}{1-a}
$$

（$a\to1$ 时峰移出量程、能效近乎平坦；$a$ 小时峰紧贴 $P_{static}$ 之上。V100+Phi-3 实测峰
≈155 W，与 $P_{static}\approx 70{-}90$ W、$a\approx 0.4{-}0.5$ 一致。注意 $P_{static}$ 与
$\theta$ 部分简并，$P^*$ 只在 $a$ 可靠时定量可信。）

---

## 三、为什么没有阶段（与 decode 的对照）

| | prefill | decode |
|---|---|---|
| 每 token 耗时 | $t = T_{mem} + C x^{-p}$，$T_{mem}\approx 0$ | 同式，$T_{mem}$ 是**常数主项** |
| 低功率 | $t\approx Cx^{-p}$：幂律上升 | 同（阶段 1，被挤成计算受限） |
| 高功率 | 仍 $t\approx Cx^{-p}$：**继续上升** | $t\to T_{mem}$：**平台**（阶段 3） |
| 中段 | 无转折 | 两项相当，边际递减（阶段 2） |
| 天花板 | 频率顶 $T_{f_{max}}$（常在量程外） | 带宽顶 $B/T_{mem}$（量程内，固定） |

一句话：**阶段结构完全由 $T_{mem}$ 地板的有无决定**。prefill 没有地板，所以单段；
decode 有地板，所以三阶段。这就是"同一条定律、两个极限"。

---

## 四、标定与验证（V100，10 类 workload，v3 数据）

对每个 workload 的 prefill 功率-cap 扫描（8 点）做两步时钟空间拟合
（与 decode 相同流程：功率侧 $P_s,\chi,\theta$；吞吐侧对数最小二乘出 $p,T_{f_{max}}$；
合成 $\text{Throughput}(P)$ 在功率空间打分），并与 V²f 基线同口径对比：

| workload | 统一模型 R² | V²f 基线 R² | $p$ | $p/\theta$ |
|---|--:|--:|--:|--:|
| chat-phi3 | 0.923 | 0.972 | 1.32 | 0.88 |
| rag-phi3 | 0.992 | 0.991 | 0.92 | 0.33 |
| code-phi3 | 0.992 | 0.992 | 0.97 | 0.35 |
| longform-phi3 | 0.957 | 0.956 | 1.19 | 0.68 |
| summarize-qwen7b | 0.986 | 0.995 | 0.94 | 0.48 |
| translate-qwen3b | 0.962 | 0.984 | 1.02 | 0.52 |
| fastchat-qwen15b | 0.950 | 0.984 | 1.01 | 0.58 |
| classify-qwen7b | 0.986 | 0.991 | 0.91 | 0.43 |
| qwen3chat-4b | 0.871 | 0.849 | 1.18 | 0.99 |
| qwen3think-4b | 0.979 | 0.974 | 0.81 | 0.35 |

**核心验证**：$p = 0.81$–$1.32$（中位 ≈0.99）——**10 个 workload 从 cap 扫描独立复原了
"计算受限 ⇒ 吞吐∝频率"的机制指数**，与锁频 DVFS 直接实测的 $T\propto f^{0.90}$ 一致。
这是本构建方式（显式 $x(P)$ 合成）最强的机制证据。

**诚实说明**：
1. 两种参数化拟合能力相当（统一模型 9/10 的 R²≥0.92；V²f 在样本内/留一略优于统一模型，
   统一模型在 qwen3chat 上更好）。选择统一构建的理由是**框架一致性与参数可解释性**
   （$p$ 直接是计算指数），不是拟合优度。
2. $P_{static}\leftrightarrow\theta$ 部分简并（与 decode 相同），单独数值仅供参考，
   合成的 $\text{Throughput}(P)$ 不受影响。
3. qwen3chat 两种模型都偏低（0.85/0.87）：最低 cap 点（实抽 72 W、时钟 441 MHz）落在电压
   地板区，任何光滑单段模型都难以同时吃下该点与主段——属低功率角的已知偏离。
