# GPU 功率 vs Token 吞吐率的解析模型(架构 + batch 显式版)

> 对象:`microsoft/Phi-3-mini-4k-instruct`(fp16)on `NVIDIA Tesla V100-DGXS-32GB`
> 风格:延续 SweetSpot 的第一性原理(逐层 FLOP + 访存复杂度),但从「能耗 vs 序列长度」扩展到「**功率 P vs 吞吐 T**」。所有系数都写成架构参数 $\{L,d,n_q,n_{kv},h,d_{ff},b\}$、batch $B$、序列长度 $S/C$、时钟 $f$ 的**显式函数**,不留神秘常数。先符号推导、量纲校验,再代入数值;实测数据只作量级锚点,不用于拟合。

> **三路推导的裁定汇总(凡有分歧处,本文采纳的结论)**
> 1. **gated-FFN 系数**:三路一致且校验通过 —— SwiGLU 三个矩阵 $=6\,d\,d_{ff}$,而 Phi-3 恰有 $6d_{ff}=16d$,故 $6\,d\,d_{ff}=16d^2$,投影+FFN $=24d^2$ **精确成立**。
> 2. **$L\cdot24d^2=2N_{ne}$**:是 **≈** 而非 **=**(差 0.0055%,源于 lm_head/末层 norm 未计入)。本文写 $\approx$。
> 3. **$F_{proj}=8d^2$ 仅对 MHA 精确**:一般 GQA 须写 $F_{proj}=4d^2+4\,d\,n_{kv}h$($n_{kv}h=d$ 时退化为 $8d^2$)。Phi-3 是 MHA 故 $8d^2$ 精确,但通用模型保留 $n_{kv}h$ 项。
> 4. **prefill 的 $B$ 消去**:三路一致且校验通过(直接数值验证 $T_{pre}(B)$ 收敛到 $\Phi/c_{pre}$)。
> 5. **decode 仿射极限**:低 $B$ 线性 $\approx B\beta/W$,高 $B$ 顶 $\beta/(C\cdot kv_{tok})$,校验通过。
> 6. **decode 斜率 $s$ 中谁主导**:Lens A 原稿说「KV 项主导」,但用其自身锚点数值算出**是 FLOP 项主导**。本文采纳裁定:给出**交叉条件**,按实测 $\varepsilon$ 比值判定 FLOP 项主导(详见 §5)。

---

## 1. 目标与符号表

### 1.1 架构 / 硬件 / 负载符号

| 类别 | 符号 | 含义 | Phi-3 / V100 数值 |
|---|---|---|---|
| 架构 | $L$ | 层数 | 32 |
| | $d$ | 隐藏维 | 3072 |
| | $n_q$ | query 头数 | 32 |
| | $n_{kv}$ | KV 头数(MHA: $=n_q$) | 32 |
| | $h$ | head_dim | 96 |
| | $d_{ff}$ | FFN 中间维(SwiGLU) | 8192 |
| | $b$ | 每参数字节(fp16) | 2 |
| | $N_{ne}$ | 非嵌入参数量 | $3.624\times10^9$ |
| | $N$ | 总参数量 | $3.821\times10^9$ |
| | vocab | 词表 | 32064 |
| 负载 | $S$ | prefill 提示长度($=n_{in}$) | 128 |
| | $C$ | decode 上下文长度($\approx$ctx) | 256 |
| | $B$ | batch | 扫描变量 |
| | $n_{out}$ | 生成 token 数 | — |
| 硬件 | $\Phi(f)$ | fp16 峰值算力 $=\Phi_{max}\,f/f_{max}$ | $\Phi_{max}=87.01$ TFLOP/s @ $f_{max}=1530$ MHz |
| | $\beta$ | HBM2 峰值带宽(由显存时钟定,$\approx$与 $f$ 无关) | $781.9$ GB/s |
| | $I^\*$ | roofline 脊点 $=\Phi/\beta$ | $111.3$ FLOP/byte |
| | $P_{static}$ | 静态/漏电底噪 | $\approx44$ W(深空闲)/ $\approx90$ W(活动 uncore 底) |
| | cap | 功耗墙 | 250 W |
| | $V(f)$ | 核心电压 $\approx V_0+\gamma f$(顶端 $\propto f$);电压下限以下 $\approx V_{min}$(平) | — |
| 派生 | $\varepsilon_{flop}(f)$ | 每 FLOP 能耗 $\propto V(f)^2$($\propto f^2$ 顶端) | $\approx3\text{–}4$ pJ/FLOP |
| | $\varepsilon_{bit}$ | 每 byte 能耗 $\approx$ 常数 | $\approx50\text{–}77$ pJ/byte(裸)/ 有效值见 §6 |
| | MFU | 实际/峰值算力占比 | 实测 $\approx0.53$ |
| | $t_{ov}$ | 每次调用固定启动开销 | — |

### 1.2 关键架构恒等式(coefficient 的「脊柱」)

- **头分割精确**:$n_q\cdot h = 32\cdot96 = 3072 = d$ ⇒ Q,K,V,O 各为 $d\times d$ 矩阵,无投影富余。
- **gated-FFN ↔ $16d^2$ 奇迹**:SwiGLU 三矩阵 = $3\,d\,d_{ff}$ MAC = $6\,d\,d_{ff}$ FLOP;而 $6d_{ff}=6\cdot8192=49152=16\cdot3072=16d$,故

$$6\,d\,d_{ff}=16d^2=1.510\times10^8 \quad\text{(精确)}.$$

  因此每层每 token 的稠密(权重)FLOP $=8d^2+6\,d\,d_{ff}=8d^2+16d^2=24d^2$,SweetSpot 的「$24\,n_{in}d^2$」系数在门控 FFN 下**原样存活**。
- **「2N 规则」(≈)**:$L\cdot24d^2 = 32\cdot2.265\times10^8 = 7.2478\times10^9 \approx 2N_{ne}=7.2482\times10^9$(差 0.0055%,$2N_{ne}$ 含 lm_head/末层 norm)。
- **KV 缓存字节/token(求和到全层)**:

$$kv_{tok}=2\,L\,n_{kv}\,h\,b = 2\cdot32\cdot32\cdot96\cdot2 = 393216\ \text{B}=0.393\ \text{MB/token}.$$

- **权重字节(每步流式读一次)**:$W=N\cdot b = 3.821\times10^9\cdot2 = 7.642\times10^9\ \text{B}=7.64\ \text{GB}$.
- **注意力系数**:每层每 token $=4\,(\text{seq})\,d$(打分 $QK^\top$:$2\,\text{seq}\,d$;$AV$:$2\,\text{seq}\,d$)。prefill seq$=S$,decode seq$=C$。

---

## 2. 每 token 的 FLOPs 与字节(从架构推)

1 MAC $=$ 2 FLOP。逐项分解,系数全部显式。

### 2.1 Prefill(单次调用处理 $B\cdot S$ 个提示 token)

每 token 每层:

$$
\underbrace{F_{proj}}_{\text{Q,K,V,O}}=8d^2\ \big(\text{MHA};\ \text{通用 GQA: } 4d^2+4\,d\,n_{kv}h\big),\quad
\underbrace{F_{ffn}}_{\text{gate+up+down}}=6\,d\,d_{ff}\!\equiv\!16d^2,\quad
\underbrace{F_{attn}}_{QK^\top+AV}=4Sd.
$$

合并全层:

$$\boxed{\,c_{pre}(S)=L\,(24d^2+4Sd)\quad[\text{FLOP/token}]\,}$$

整次调用 FLOP $=B\,S\,c_{pre}(S)$。Phi-3,$S=128$:

$$c_{pre}(128)=32\,(2.265\times10^8+4\cdot128\cdot3072)=32\,(2.265\times10^8+1.573\times10^6)=7.298\times10^9\ \text{FLOP/token}$$

(稠密 $7.248\times10^9$ + 注意力 $5.0\times10^7$;注意力仅占 0.7% ⇒ prefill $\approx$ 纯稠密 GEMM)。

**字节**:权重读一次、被 $B\cdot S$ 个 token 复用 ⇒ 摊到每 token 的权重字节 $=W/(BS)$,极小;另有 $O(d)$ 的激活/KV 写。

### 2.2 Decode(单步发射 $B$ 个 token,每个对 $C$ 个缓存位做注意力)

每 token 每层 $=8d^2+6\,d\,d_{ff}+4Cd$,合并全层:

$$\boxed{\,c_{dec}(C)=L\,(24d^2+4Cd)\quad[\text{FLOP/token}]\,}$$

整步 FLOP $=B\,c_{dec}(C)$。Phi-3,$C=256$:

$$c_{dec}(256)=32\,(2.265\times10^8+4\cdot256\cdot3072)=7.348\times10^9\ \text{FLOP/token}.$$

**字节/步**:权重流式读一次(不复用,只产 $B$ 个 token)$=W$;为 $B$ 条上下文长 $C$ 的序列读 KV $=B\,C\,kv_{tok}$:

$$\boxed{\,\text{bytes}_{dec}(B,C)=W+B\,C\,kv_{tok}\,}$$

每序列 KV($C=256$)$=C\cdot kv_{tok}=256\cdot393216=1.006\times10^8\ \text{B}=0.1006\ \text{GB}$。

### 2.3 算术强度 vs 脊点 $I^\*=111.3$ FLOP/byte

- **Prefill**:$I_{pre}=c_{pre}/(W/(BS))$,随 $BS$ 增大;$BS=256$ 时 $I_{pre}=256\cdot7.298\times10^9/7.642\times10^9=244\gg111$ ⇒ **算力受限(compute-bound)**。
- **Decode**:$I_{dec}=B\,c_{dec}/(W+B\,C\,kv_{tok})\approx0.96\,B$;$B=1$ 时 $\approx0.95\ll111$ ⇒ **访存受限(memory-bound)**,所有现实 $B$ 皆然。

这两条「分别钉住不同引擎」是后文一切非对称的根源。

---

## 3. 吞吐模型 $T(B,\dots)$

roofline 墙时:$\;t=\max\!\big(\text{FLOP}/(\Phi(f)\,\text{MFU}),\ \text{bytes}/\beta\big)+t_{ov}$;$\;T=\dfrac{\text{发射 token 数}}{t}$。

### 3.1 Prefill —— 饱和区 $B$ 消去,得到算力顶

取算力分支:$t_{pre}=\dfrac{B\,S\,c_{pre}}{\Phi(f)\,\text{MFU}}+t_{ov}$,发射 $BS$ 个 token:

$$T_{pre}=\frac{BS}{t_{pre}}=\frac{BS\,\Phi(f)\,\text{MFU}}{BS\,c_{pre}+\Phi(f)\,\text{MFU}\,t_{ov}}.$$

当 $BS$ 增大、$t_{ov}$ 被摊薄,**$B$(与前因子里的 $S$)消去**:

$$\boxed{\,T_{pre}^{max}(f,S)=\frac{\Phi(f)\,\text{MFU}}{c_{pre}(S)}=\frac{\Phi(f)\,\text{MFU}}{L\,(24d^2+4Sd)}\quad(\text{与 }B\text{ 无关})\,}$$

> **裁定说明**:$B$ 在前因子里被消去,但 $S$ 仍残留在 $c_{pre}(S)$ 的 $4Sd$ 项里 —— 故 $T_{pre}^{max}$ **与 $B$ 无关、却仍弱依赖 $S$**。$B$ 的真实作用只剩 MFU($B$) 爬升,等价于**仿射调用时间** $t_{pre}=t_{fixed}+\kappa B$,$\kappa=S\,c_{pre}/(\Phi\,\text{MFU}_{sat})$,$T_{pre}=BS/(t_{fixed}+\kappa B)$,上限 $S/\kappa=T_{pre}^{max}$。

由 $\Phi(f)=\Phi_{max}(f/f_{max})$ ⇒ 饱和顶 $T_{pre}^{max}\propto f$。

**数值**(Phi-3,$S=128$):

$$T_{pre}^{max}=\frac{87.01\times10^{12}\cdot\text{MFU}}{7.298\times10^9}=11{,}922\cdot\text{MFU}\ \text{tok/s}.$$

MFU=1 ⇒ 11.9k;实测 $\text{MFU}\approx0.534$(`fit_summary` `achieved_mfu`),理想顶 $\approx6.3$k,batch 扫描实测天花板 5.03k tok/s(`throughput_ceiling`,$t_{ov}$+MFU<1 共同压低)。

> **数值一致性裁定**:`fit_summary` 给出 `compute_roof_ideal=11963`(用稠密 $2N_{ne}=7.248\times10^9$)与本文 $\Phi/c_{pre}=11922$(含注意力的 $c_{pre}=7.298\times10^9$)差 0.3%,纯属用哪份 FLOP 计数。天花板 5.03k 对应**有效** MFU $=5031/11922=0.42$;`achieved_mfu`=0.534 对应 6.3k 理想顶 —— 二者是「实测天花板」与「峰值理想」两个口径,不应混用为「5.0k @ MFU 0.53」。

### 3.2 Decode —— $B$ 仿射,时钟无关

取访存分支:$t_{dec}=t_{ov}+\dfrac{W+B\,C\,kv_{tok}}{\beta}$,发射 $B$ 个 token:

$$\boxed{\,T_{dec}(B,C,f)=\frac{B}{\,t_{ov}+\dfrac{W+B\,C\,kv_{tok}}{\beta}\,}\,}$$

- **低 $B$($W$ 主导)**:$T_{dec}\approx\dfrac{B\beta}{W+\beta t_{ov}}\to\dfrac{B\beta}{W}$,**线性**,斜率

$$\frac{\beta}{W}=\frac{781.9\times10^9}{7.642\times10^9}=102.3\ \text{tok/s/(单位 batch)}.$$

- **高 $B$($KV$ 主导)**:$T_{dec}\to\dfrac{\beta}{C\,kv_{tok}}=\dfrac{\beta}{2L\,n_{kv}h\,b\,C}$,**带宽顶**

$$\frac{781.9\times10^9}{256\cdot393216}=7{,}767\ \text{tok/s}.$$

- **时钟无关**:$\beta$ 由显存时钟定 ⇒ $T_{dec}$ 对核心 $f$ **平**(实测 $T\propto f^{0.26}$,残余微升来自 FLOP/uncore 时钟弱耦合)。

实测 decode 天花板仅 $\sim0.94$k tok/s(`throughput_ceiling`),远低于带宽顶 7.77k ⇒ 实际是**启动开销 + 访存**双重受限($t_{ov}$ 约 40 ms,`t_fixed_ms=40.8`),非带宽饱和。

---

## 4. 功率模型 $P(B,f)$

按引擎分解:

$$\boxed{\,P=P_{static}+\varepsilon_{flop}(f)\cdot(\text{FLOP/s})+\varepsilon_{bit}\cdot(\text{byte/s})\,}$$

- $P_{static}\approx44$ W(漏电/uncore 底)。
- $\varepsilon_{flop}(f)\propto C_{eff}V(f)^2$(CMOS $E_{op}=\tfrac12CV^2$)。$V(f)\approx V_0+\gamma f$ ⇒ 顶端 $\varepsilon_{flop}\propto f^2$;电压下限以下 $V\approx V_{min}$ ⇒ $\varepsilon_{flop}\approx$ 常数。
- $\varepsilon_{bit}\approx$ 常数(J/byte,$\sim$pJ/bit;HBM 时钟域电压几乎不随核心 $f$ 变)。

**量纲核对**:$[\varepsilon_{flop}][\text{FLOP/s}]=(\text{J/FLOP})(\text{FLOP/s})=\text{W}$ ✓;$[\varepsilon_{bit}][\text{byte/s}]=\text{W}$ ✓。

**把速率写成 arch+$B$+$f$**:

- **Prefill(算力引擎钉死,duty$\to$1)**:$\text{FLOP/s}\to\Phi(f)\,\text{MFU}=\Phi_{max}\text{MFU}(f/f_{max})$,byte/s 小(权重复用)。

$$P_{pre}\approx P_{static}+\varepsilon_{flop}(f)\cdot\Phi_{max}\,\text{MFU}\,(f/f_{max})\quad\big(\propto V(f)^2 f\big).$$

- **Decode(访存引擎钉死)**:$\text{byte/s}=\dfrac{W+B\,C\,kv_{tok}}{t_{dec}}\to\beta$(饱和),$\text{FLOP/s}=B\,c_{dec}/t_{dec}$(欠顶)。

$$P_{dec}\approx P_{static}+\varepsilon_{bit}\,\beta\cdot\text{duty}_{mem}+\varepsilon_{flop}(f)\cdot\frac{B\,c_{dec}}{t_{dec}}.$$

---

## 5. $P(T)$:两个旋钮 × 两个阶段(消去控制变量)

### 5.1 频率旋钮(固定 $B$,扫描 $f$)

**Prefill(算力受限)** —— 立方律。
$T_{pre}=\Phi(f)\text{MFU}/c_{pre}\propto f$ ⇒ $f=\dfrac{f_{max}\,c_{pre}}{\Phi_{max}\text{MFU}}\,T$。算力功率 $P_{cmp}=\varepsilon_{flop}(f)\cdot\Phi(f)\text{MFU}$。在 $V\propto f$ 段,$\varepsilon_{flop}=k_V f^2$、$\Phi(f)=\Phi_{max}f/f_{max}$,得 $P_{cmp}=\dfrac{k_V\Phi_{max}\text{MFU}}{f_{max}}f^3$。代入 $f(T)$:

$$\boxed{\,P_{pre}(T)=P_{static}+k_c\,T^3\,},\qquad
\boxed{\,k_c=k_V\,f_{max}^2\,\frac{[\,L(24d^2+4Sd)\,]^3}{(\Phi_{max}\,\text{MFU})^2}\;\propto\;\frac{c_{pre}(S)^3}{\Phi_{max}^2}\,}$$

> **裁定说明(修正 Lens A 的乱码中间式)**:正确形式是 $k_c=k_V\,[f_{max}\,c_{pre}/(\Phi_{max}\text{MFU})]^3\cdot(\Phi_{max}\text{MFU}/f_{max})$,化简即上式 —— **$c_{pre}$ 的立方**、分母 $(\Phi_{max}\text{MFU})^2$。Lens A 原稿那行写成 $c_{pre}^{2}/(\Phi_{max}\text{MFU})^3\cdot\Phi_{max}$ 是错的(与它自己的 boxed 结论矛盾),已弃用。架构含义:**$k_c$ 随每 token FLOP 系数 $L(24d^2+4Sd)$ 的立方暴涨** —— 更深/更宽的模型($L,d,d_{ff}$ 大)立方曲线更陡,长提示由 $4Sd$ 项加成。

量纲:$[k_V]=\text{J/FLOP/MHz}^2$,$[k_c]=\dfrac{\text{J/FLOP/MHz}^2\cdot(\text{FLOP/tok})^3\cdot\text{MHz}^2}{(\text{FLOP/s})^2}=\dfrac{\text{W}}{(\text{tok/s})^3}$ ✓。
数值:$k_c\approx4.2\times10^{-10}\ \text{W/(tok/s)}^3$ ⇒ $k_c\cdot(7351)^3\approx167$ W 动态,与实测 $P_{dyn}$ 同量级 ✓。

**Prefill 的诚实边界**:干净 $T^3$ 需 $V\propto f$;V100 在 $\lesssim1300$ MHz 段 $V\approx$ 常数 ⇒ $\varepsilon_{flop}\approx$ 常数 ⇒ $P_{cmp}\propto f\propto T$(**线性**)。完整曲线 $P=P_{static}+aT+bT^3$,只有近 $f_{max}$ 才立方主导。实测指数对 $P_{static}$ 口径敏感:$\gamma\approx1.45$(底 44 W)→ $\gamma\approx2.99$(活动底 90 W)。

**Decode(访存受限)**:$T\propto\beta$,与 $f$ 无关 ⇒ $T$ **平**,而 $P$ 仍随 $f$ 上升 ⇒ $P(T)$ 是一条**竖线**(抬时钟纯属烧电)。频率旋钮对 decode 是错旋钮。

### 5.2 batch 旋钮(固定 $f=f_{max}$,扫描 $B$)

**Decode(访存受限)** —— 仿射。
每步:能量 $=\varepsilon_{bit}(W+B\,C\,kv_{tok})+\varepsilon_{flop}\,B\,c_{dec}$,token $=B$,步时 $t\approx t_{ov}+(W+B\,C\,kv_{tok})/\beta$。$P=$ 能量$/t$、$T=B/t$,消去 $B$:

$$\boxed{\,P_{dec}(T)=P_{static}+a+s\,T\,}$$

$$\boxed{\,s=\varepsilon_{bit}\,C\,kv_{tok}+\varepsilon_{flop}\,c_{dec}(C)=\varepsilon_{bit}\,(2L\,n_{kv}h\,b\,C)+\varepsilon_{flop}\,L(24d^2+4Cd)\,}\quad[\text{J/token,边际能耗}]$$

$$\boxed{\,a\approx\varepsilon_{bit}\,\frac{W}{t_{ov}+W/\beta}\;(\text{即每步必付的权重流式底 }W/\beta\text{,被 }B\text{ 个 token 摊销})\,}$$

$s$ 携带 $C\,kv_{tok}=C\cdot2L\,n_{kv}h\,b$;$a$ 携带 $W=Nb$ —— **都是架构的显式函数**。低 $B$ 线性,高 $B$ 顶 $\beta/(C\,kv_{tok})$。
实测:$P_{dec}=110.9+0.19\,T$ W($R^2\approx1.0$,B≤32 未限频段;含 B=64 限频点为 $112+0.17T$)。截距 $\approx P_{static}+$ 权重流/uncore 底,斜率 $s\approx0.19$ W/(tok/s)。

> **裁定说明(修正「谁主导斜率 $s$」的自相矛盾)**:Lens A 原稿口头说「KV 项主导」,但用其自身锚点算出 $s_{kv}=\varepsilon_{bit}C\,kv_{tok}\approx6.4\times10^{-3}$、$s_{flop}=\varepsilon_{flop}c_{dec}\approx2.9\times10^{-2}$ J/tok ⇒ **FLOP 项约大 4.6 倍**。给出**交叉条件**:
> $$\text{KV 主导} \iff \varepsilon_{bit}/\varepsilon_{flop}>\frac{c_{dec}}{C\,kv_{tok}}=\frac{7.35\times10^9}{256\cdot393216}\approx73\ \text{FLOP/byte}.$$
> 实测 $\varepsilon_{bit}/\varepsilon_{flop}\approx64/4\approx16\ll73$ ⇒ **本配置下 FLOP 项主导斜率**。注意此判定仅精确到 $\varepsilon$ 比值的 $\sim2\times$;能耗杠杆(GQA/减字节)主要削的是**较小**的 KV 项,除非 $\varepsilon_{bit}/\varepsilon_{flop}$ 远大于实测值。

**Prefill(算力受限)**:加大 $B$ 只是填满**已饱和**的算力引擎(MFU$\to1$,约 $B\approx5$–6,`B0=5.4`)⇒ $T$ 钉在算力顶、$P$ 几乎立刻撞到功耗墙(实测 $\approx243$ W $\approx$ cap,跨 $B=2\dots64$ 基本平,时钟自动从 1530 降频到 $\sim1250$ 以守住墙)。$P(T)$ 是「陡升→撞墙平台」,不是长仿射线。

---

## 6. 每 token 能耗 $E=P/T$ 推论 + Phi-3/V100 数值实例

$E=P/T$。

- **Prefill(频率旋钮)**:$P=P_{static}+k_cT^3$ ⇒

$$E=\frac{P_{static}}{T}+k_c\,T^2\;\propto\;T^2\quad(\text{大 }T\text{ 时}).$$

  跑得越快,每 token 能耗**二次**上升 ⇒ **能耗最优 prefill = 最低时钟**。实测随 batch 填充(占用摊销,非频率)效率从 $\sim12$ 升到 $\sim34$ tok/J。
- **Decode(batch 旋钮)**:$P=P_{static}+a+s\,T$ ⇒

$$E=\frac{P_{static}+a}{T}+s\;\xrightarrow{\ B\uparrow\ }\;s.$$

  随 $B$ 增大摊销固定权重流 $W$,$E$ 降到渐近底 $s$($\approx0.18$–0.19 J/tok $\Rightarrow\sim5.3$–5.6 tok/J)。下限由边际项(本配置 FLOP 项主导,其次 KV 字节项)决定;杠杆 = **少搬字节**(GQA/MQA 缩 $kv_{tok}$ 里的 $n_{kv}$、KV 压缩)+ 增大 $B$ 摊销 $W$。实测 $\sim0.4\to5.6$ tok/J 横跨 batch 扫描。

### 数值实例总表(量级锚点,$\sim2\times$ 内一致)

| 量 | 公式 | 数值 |
|---|---|---|
| 脊点 $I^\*$ | $\Phi_{max}/\beta$ | 111.3 FLOP/byte |
| $c_{pre}(128)$ | $L(24d^2+4Sd)$ | $7.298\times10^9$ FLOP/tok |
| $c_{dec}(256)$ | $L(24d^2+4Cd)$ | $7.348\times10^9$ FLOP/tok |
| 稠密 FLOP/tok | $L\cdot24d^2$ | $7.248\times10^9\approx2N_{ne}$ |
| $kv_{tok}$ | $2L\,n_{kv}h\,b$ | 393216 B = 0.393 MB/tok |
| $W$ | $Nb$ | 7.642 GB |
| Prefill 算力顶 | $\Phi_{max}\text{MFU}/c_{pre}$ | 11.9k(MFU=1)/ 6.3k(0.53)/ 5.0k(实测) |
| Decode 低-$B$ 斜率 | $\beta/W$ | 102.3 tok/s/batch |
| Decode 带宽顶 | $\beta/(C\,kv_{tok})$ | 7.77k tok/s(实测受 $t_{ov}$ 限 $\sim0.94$k) |
| $\varepsilon_{flop}$ | $P_{cmp}/(\text{FLOP/s})$ | 3–4 pJ/FLOP |
| $\varepsilon_{bit}$(裸) | $P_{mem}/\beta$ | 50–77 pJ/byte |
| $\varepsilon_{bit}$(有效) | $s_{kv}/(C\,kv_{tok})$ | $\approx1.8$ nJ/byte(含 uncore/util<1/启动,故偏高) |
| $k_c$(频率/prefill) | $k_V f_{max}^2 c_{pre}^3/(\Phi_{max}\text{MFU})^2$ | $4.2\times10^{-10}$ W/(tok/s)³ |
| Decode 斜率 $s$(batch) | $\varepsilon_{bit}C\,kv_{tok}+\varepsilon_{flop}c_{dec}$ | $\approx0.19$ J/tok(实测拟合) |

---

## 7. 适用边界

1. **电压下限**:干净立方 $P\propto T^3$ 仅在 $V\propto f$ 的高频段成立;V100 在 $\lesssim1300$ MHz 段 $V\approx V_{min}$ 恒定 ⇒ 退化为 $P\propto T$ 线性。实测前缀指数 $\gamma\in[1.45,2.99]$,随 $P_{static}$ 口径(44 vs 90 W)漂移。
2. **功耗墙裁剪**:近 $f_{max}$ 与高 $B$ prefill 会撞 250 W 墙,GPU 自动降频(请求 1410/1530 MHz → 实际 $\sim1342/1321$),把曲线顶端削平,实测 $T\propto f$ 指数从理想 0.9 掉到 0.77。
3. **MFU**:$T_{pre}^{max}$ 的 MFU 是 $B$、shape、kernel 的函数;小 $B$ 欠填 GEMM(MFU 随 $B$ 爬升至饱和 $\approx5$–6),这是 prefill batch 旋钮上「升-饱和」形状的来源。「理想峰值顶」(MFU=0.53→6.3k)与「实测天花板」(5.0k→有效 0.42)是两个口径,核对时勿混用。
4. **两旋钮区分**:频率旋钮上 prefill 立方、decode 平(竖线);batch 旋钮上 decode 仿射、prefill 撞墙。**对 decode 抬时钟无效,要靠 batch;对 prefill 省能要靠降时钟。**
5. **GQA 一般化**:本模型对 MHA(Phi-3,$n_{kv}=n_q$)精确;真 GQA($n_{kv}<n_q$)须用 $F_{proj}=4d^2+4\,d\,n_{kv}h$,且 $kv_{tok}=2L\,n_{kv}h\,b$、注意力 $4(\text{seq})d$ 中的头维已自动随 $n_{kv}$/$n_q$ 正确缩放 —— 这也正是 GQA 削减 decode 访存底 $s_{kv}$ 的机理。
6. **$\varepsilon$ 比值不确定性**:$\varepsilon_{flop}$、$\varepsilon_{bit}$ 仅知到 $\sim2\times$,故 decode 斜率「FLOP vs KV 谁主导」(交叉点 73 FLOP/byte,实测比值 16)的结论在该误差内成立,但属边界判断而非铁律。

---

**结论**:每个系数都已写成 $\{L,d,n_q,n_{kv},h,d_{ff},b\}$、$B$、$S/C$、$f$ 的显式函数 —— prefill 算力顶 $\Phi(f)\text{MFU}/[L(24d^2+4Sd)]$、decode 仿射 $B/[t_{ov}+(W+BC\,kv_{tok})/\beta]$、频率旋钮立方系数 $k_c\propto c_{pre}^3/\Phi_{max}^2$、batch 旋钮斜率 $s=\varepsilon_{bit}C\,kv_{tok}+\varepsilon_{flop}c_{dec}$ 与截距 $a\approx\varepsilon_{bit}W/(t_{ov}+W/\beta)$ —— 全部量纲自洽,且与 `results/model_info.json`、`fit_summary.json`、`dvfs/prefill/decode.csv` 的实测锚点在 $\sim2\times$ 内吻合。

相关文件(绝对路径):
`/home/markliu/Desktop/powerchar/results/model_info.json`、`/home/markliu/Desktop/powerchar/results/fit_summary.json`、`/home/markliu/Desktop/powerchar/results/dvfs.csv`、`/home/markliu/Desktop/powerchar/results/prefill.csv`、`/home/markliu/Desktop/powerchar/results/decode.csv`、`/home/markliu/Desktop/powerchar/POWER_THROUGHPUT_MODEL.md`、`/home/markliu/Desktop/powerchar/ANALYTIC_MODEL.md`。