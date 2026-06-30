# 解码阶段：功率 → 吞吐 的三阶段理论模型

> **对象**：自回归解码（decode），固定 batch；通过 DVFS 调节 SM 频率 $f_{sm}$ 来限制功耗。
> **目标**：给出**吞吐量关于功率的显式解析模型**，并解释其三阶段曲线机理。

配套拟合脚本 [`plot_decode_fixedbatch.py`](plot_decode_fixedbatch.py) / 图 [`fig_decode_fixedbatch.png`](fig_decode_fixedbatch.png)。

---

## 一、核心物理逻辑与决定因素

Token 吞吐量：

$$
\text{Throughput} = \frac{\text{Batch Size}}{\text{Time per Token}}
$$

单 Token 耗时由**访存**与**计算**两部分**相加**（不是取 max）：

$$
\text{Time per Token} = T_{mem} + T_{comp}
= \frac{D_{mem}}{BW(f_{mem})} + \frac{O_{comp}}{OPS(f_{sm})}
$$

| 符号 | 含义 |
|---|---|
| $D_{mem}$ | 搬运的数据量（模型权重 + KV Cache） |
| $BW(f_{mem})$ | 显存带宽，取决于显存控制器（MC）频率 $f_{mem}$ |
| $O_{comp}$ | 单步计算量（FLOPs） |
| $OPS(f_{sm})$ | GPU 实际运算速度，与 SM 频率 $f_{sm}$ 正相关 |

**两条关键约束**：① Decode 阶段 $T_{mem}\gg T_{comp}$（访存受限）；② DVFS 调的是 **SM 频率 $f_{sm}$**，而**显存控制器频率 $f_{mem}$ 基本固定**。由约束 ② 得本模型的核心：

$$
\boxed{\,BW(f_{mem})=\text{const}\;\Rightarrow\;T_{mem}=\text{const}\;\Rightarrow\;\text{吞吐天花板}=\frac{\text{Batch Size}}{T_{mem}}\,}
$$

访存时间是一条**与 SM 频率无关的常数地板**，唯一决定吞吐上限；SM 频率只能改变 $T_{comp}$。

---

## 二、DVFS 功率模型与显式的吞吐–功率关系

SM 频率经 DVFS 决定功率（电压随频率非线性上升，动态功耗并为幂律）：

$$
P(f_{sm}) = P_{static} + \chi\left(\frac{f_{sm}}{f_{max}}\right)^{\theta},\qquad \theta\in[2,3]
$$

吞吐与功率都只由 $f_{sm}$ 驱动，故「吞吐 vs 功率」是一条以 $f_{sm}$ 为参数的曲线。把功率**反解出频率**再代入吞吐，即得显式模型。记 $x=f_{sm}/f_{max}$：

$$
x(P)=\left(\frac{P-P_{static}}{\chi}\right)^{1/\theta},\qquad P_{static}<P\le P_{static}+\chi=P(f_{max})
$$

代入 $\text{Throughput}=B/(T_{mem}+T_{comp})$，其中 $T_{comp}=C\,(x^{-p}-1)$（$p$ 源自算力降级 $OPS\propto f_{sm}^{\,p}$，$x{=}1$ 时 $T_{comp}{=}0$）：

$$
\boxed{\;\text{Throughput}(P)=\frac{B}{\,T_{mem}+C\!\left[\left(\dfrac{P-P_{static}}{\chi}\right)^{-p/\theta}-1\right]}
=\frac{B}{\,(T_{mem}-C)+C\left(\dfrac{\chi}{P-P_{static}}\right)^{p/\theta}}\;}
$$

**适用域**：$P_{static}<P\le P(f_{max})$。当功率上限 $\ge P(f_{max})$ 时频率已顶到 $f_{max}$（$x{=}1$），须把 $x$ 钳到 1，吞吐恒为天花板 $B/T_{mem}$（$x>1$ 分支非物理，不可代入）。

**两端渐近**（对应下文三阶段）：

$$
\underbrace{\text{Throughput}\approx\frac{B}{C}\left(\frac{P-P_{static}}{\chi}\right)^{p/\theta}\propto (P-P_{static})^{p/\theta}}_{\text{低功率：}T_{comp}\gg T_{mem}\text{，指数 }p/\theta\lesssim1\text{，近似线性}}
\qquad
\underbrace{\text{Throughput}\to \frac{B}{T_{mem}}}_{\text{高功率：}T_{comp}\to0\text{，平台}}
$$

---

## 三、吞吐随功率从低到高的三个阶段

| 阶段 | 功率/频率条件 | 机理 | 吞吐–功率行为 |
|---|---|---|---|
| **1 伪计算受限** | $P$ 极低（$T_{comp}>T_{mem}$，$f_{sm}$ 被压到标称 20–30%） | $OPS(f_{sm})$ 急剧下降（频率↓且占用率塌缩），$T_{comp}$ 大幅上升、超过 $T_{mem}$，访存受限任务被挤成计算受限 | 幂律上升 $\propto(P-P_{static})^{p/\theta}$，指数 $\lesssim1$，**近似线性陡升** |
| **2 边际递减** | 中等功率（$T_{comp}\sim T_{mem}$） | $f_{sm}$ 越过临界点后 $T_{comp}$ 收缩，任务回归访存受限；提频对缩短 $T_{mem}+T_{comp}$ 贡献渐小，功率主要喂给 $f_{sm}^{\theta}$ 动态功耗 | 持续上升但**斜率明显变缓** |
| **3 访存平台** | 高功率/满载（$T_{comp}<5\%\,T_{mem}$，$f_{sm}\to f_{max}$） | $T_{comp}$ 可忽略，总耗时 $\approx T_{mem}$；且 $f_{sm}$ 已顶 $f_{max}$，再加功率只升压发热不升频 | **平台** $=B/T_{mem}$，与 $P$ 脱钩 |

---

## 四、拟合结果与验证（实测 13 点）

按上式拟合 V100 + Phi-3-mini 实测扫频（batch=96，$f_{max}{=}1530$ MHz），并把 $T_{mem}$ 锚定到实测平台（最高吞吐）使天花板自洽：

| 量 | 值 |
|---|---|
| $T_{mem}$（常数访存地板） | 149.3 ms（有效带宽 ~116 GB/s，≈峰值 13%：访存受限但延迟受限） |
| $C,\ p$（$T_{comp}=C(x^{-p}{-}1)$） | 30.8 ms，$p=1.84$ |
| $P_{static},\ \chi,\ \theta$（功率） | 50 W，155.5 W，2.15 |
| **天花板** $B/T_{mem}$ | **643 tok/s**（= 实测最大，精确命中） |
| 阶段分界 | I/II：$P_1\approx70$ W（585 MHz）；II/III：$P_2\approx171$ W（1359 MHz） |
| 阶段 1 指数 $p/\theta$ | 0.855（近线性） |
| 拟合优度 | 吞吐（频率空间）$R^2{=}0.97$；功率 $R^2{=}0.99$；吞吐（功率空间）$R^2{=}0.956$ |

**验证**（双路独立推导 + 数值复核）：两路代数推导结果完全一致，且 $\text{Throughput}(P)$ 是参数模型的精确解析逆——把 $P(f)$ 回代可机器精度复现 $\text{Throughput}(f)$（差 $10^{-13}$）。

**说明/注意**：
1. 功率空间 $R^2$（0.956）略低于频率空间（0.97），是因为反解功率把功率拟合的残差也叠加进来——属正常，非建模错误。
2. 最差点在 74.4 W / 487 MHz（模型 358 vs 实测 289，误差 24%）：该点实测功率偏高于功率拟合曲线，反解出的 $x$ 偏大；其余 $P\ge153$ W 各点误差 <2%。
3. 阶段 1 的幂律只给**形状（指数 0.855）**可靠，裸前置系数在该区间不可靠（少数点）；$P_{static}$ 触下界、真值可能更低，会平移 $(P-P_{static})$ 原点。

> 理论部分（一~三）为纯符号；本节为在实测上的标定数值，与图一一对应（图 (a) 即上式的 $\text{Throughput}(P)$ 曲线 + 三阶段 + 天花板）。
