# LLM 推理的功率—吞吐解析模型

本文以 GPU 功率上限 $P$（单位 W，即通过驱动设置的 power cap）为自变量，以 token 吞吐（tok/s）为因变量，给定 GPU 规格与稠密 decoder-only 模型的结构参数及负载 $(B,s,n_o)$（批大小、prompt 长度、生成长度），建立 prefill 与 decode 两阶段的解析吞吐模型，并在时延约束下将其推广为 goodput。建模粒度为 roofline：将一次前向拆为若干部分，每个部分的耗时取其计算时间与访存时间的较大者，kernel 启动与调度等固定开销以加性常数吸收。

## 1 功率如何决定算力

功率上限通过 DVFS（动态电压频率调节）决定 GPU 的核心运行频率，进而决定有效算力。CMOS 电路的动态功耗正比于 $CV^2f$，而在工作区间内电压随频率近似线性上升，两者合并后功耗可写成频率的幂律。记峰值频率为 $f_{\max}$、静态功率（空闲时的漏电、显存刷新、风扇等功耗）为 $P_{\mathrm{stat}}$、额定功率上限为 $P_{\max}$，则

$$P(f)=P_{\mathrm{stat}}+(P_{\max}-P_{\mathrm{stat}})\left(\frac{f}{f_{\max}}\right)^{\gamma},\qquad \gamma\in[2,3],\tag{1}$$

其中指数 $\gamma$ 由硬件的电压—频率曲线决定，需实测标定。反解得到功率上限所允许的相对频率

$$\phi(P)=\frac{f(P)}{f_{\max}}=\left(\frac{P-P_{\mathrm{stat}}}{P_{\max}-P_{\mathrm{stat}}}\right)^{1/\gamma},\tag{2}$$

取值截断在 $[\phi_{\min},1]$ 之间（$\phi_{\min}$ 为硬件最低时钟对应的相对频率）。有效算力正比于频率，而 HBM 带宽通常不随功率上限调节，可近似视为常数：

$$F(P)=\eta_CF_{\mathrm{peak}}\phi(P),\qquad W=\eta_MW_{\mathrm{peak}},\tag{3}$$

其中 $F_{\mathrm{peak}}$ 与 $W_{\mathrm{peak}}$ 分别为给定精度下的峰值算力与峰值带宽，$\eta_C$、$\eta_M$ 为计算与带宽的达成效率（前者即通常所说的 MFU，因 GEMM 形状不同，prefill 与 decode 可取不同的值）。

## 2 每个 token 的计算量与访存量

设模型层数为 $L$、隐维为 $d$、查询头数为 $h$、KV 头数为 $h_{kv}$、头维为 $d_h=d/h$，则 KV 的有效维度为 $d_{kv}=h_{kv}d_h$，GQA 比为 $g=h/h_{kv}$；FFN 中间维为 $d_{ff}$、其矩阵个数为 $n_{ff}$（标准 FFN 取 2，SwiGLU 取 3），词表大小为 $V$。参与矩阵乘的参数总量按层累加，包含末端的 lm_head 而不含查表式的 embedding：

$$N=L\left(2d^{2}+2dd_{kv}+n_{ff},d,d_{ff}\right)+dV,\tag{5}$$

其中 $2d^2$ 来自 Q、O 投影，$2dd_{kv}$ 来自 K、V 投影，$n_{ff}dd_{ff}$ 来自 FFN，$dV$ 来自 lm_head。

在此基础上，计算量与访存量遵循三条计数规则（约定一次乘加为 2 FLOP）。其一，线性层每个 token 每个参数需 2 FLOP，故一个 token 通过全部线性层的计算量为 $2N$，且其权重访存量为 $b_wN$（$b_w$ 为权重字节宽度），与该 token 是否与其他 token 共享读取有关。其二，注意力打分部分与上下文长度成正比：当上下文长度为 $n$ 时，一个 token 的 $QK^{\top}$ 与 $AV$ 各需 $2Ldn$ FLOP，合计 $4Ldn$，这一计算量与 GQA 无关，因为查询侧仍保有全部 $h$ 个头。其三，KV cache 每 token 每层写入 K、V 各 $d_{kv}$ 个元素，即 $2Ld_{kv}b_{kv}$ 字节（$b_{kv}$ 为 KV 字节宽度），读取时其访存量随上下文长度线性放大，GQA 的收益正体现在此处按 $d_{kv}$ 而非 $d$ 计量。此外，每层每 token 还需从 HBM 读写约 $k_{\mathrm{act}},d,b_a$ 字节的激活（$b_a$ 为激活字节宽度，$k_{\mathrm{act}}$ 约为 8 至 20，取决于 kernel 融合程度），而 softmax、归一化等 $O(hn)$ 量级的小项相对主项可忽略，其代价一并归入效率系数。采用 FlashAttention 时注意力的访存量不含 $s^2$ 项。

## 3 Prefill 模型

一次 prefill 并行处理 $Bs$ 个 token。折合到每个 prompt token，其计算量为线性部分与注意力部分之和。注意力部分因因果掩码而只对在先位置打分，整条序列的注意力计算量为 $\sum_{i\le s}4Ldi\approx 2Lds^2$，摊到每 token 即 $2Lds$，故

$$C_{\mathrm{pre}}=2N+2Lds.\tag{6}$$

访存量方面，由于 $Bs$ 个 token 共享同一份权重、每个权重只需从 HBM 读取一次并由所有 token 均摊，权重访存摊薄为 $b_wN/(Bs)$；再加上激活读写与 KV 写入，得

$$M_{\mathrm{pre}}=\frac{b_wN}{Bs}+k_{\mathrm{act}}Ldb_a+2Ld_{kv}b_{kv}.\tag{7}$$

于是 prefill 的耗时与吞吐为

$$T_{\mathrm{pre}}(P)=Bs\cdot\max\left\lbrace \frac{C_{\mathrm{pre}}}{F(P)},\ \frac{M_{\mathrm{pre}}}{W}\right\rbrace ,\qquad X_{\mathrm{pre}}(P)=\frac{Bs}{T_{\mathrm{pre}}(P)}.\tag{8}$$

关键在于，prefill 在整个可行功率区间内始终计算受限，因而无需分段。这可由算术强度逐项验证：权重项对应的强度为 $2Bs/b_w$，只要 $Bs$ 达到几百个 token 量级（FP16 下即 $Bs\ge b_wR_{\max}/2$，实际服务几乎总能满足）便超过 $R_{\max}$；激活项对应的强度约为 $2N/(k_{\mathrm{act}}Ldb_a)$，量级为 $O(d)\sim 10^3$ FLOP/B；两者都远大于 $R_{\max}\ge R(P)$，且降低功率只会使 $R(P)$ 更小、工况更偏计算受限。因此式 (8) 中的较大者恒为计算项，prefill 吞吐化简为

$$X_{\mathrm{pre}}(P)=\frac{\eta_C^{\mathrm{pre}},F_{\mathrm{peak}}}{2N+2Lds}\ \phi(P).\tag{9}$$

该式表明 $X_{\mathrm{pre}}\propto(P-P_{\mathrm{stat}})^{1/\gamma}$，随功率单调上升且上凸，边际收益递减；首 token 时延即 $\mathrm{TTFT}=Bs/X_{\mathrm{pre}}$。prompt 长度 $s$ 只改变式 (9) 的系数，不改变吞吐对功率的函数形状。

## 4 Decode 模型

在 decode 阶段，每一步生成 $B$ 个 token，第 $t$ 步的上下文长度为 $n=s+t$。将一步折合到每个生成 token，并按物理性质拆为两个部分。线性部分涵盖 QKVO 投影、FFN 与 lm_head，其计算量为 $C_{\mathrm{lin}}=2N$；由于同一步的 $B$ 个 token 共享权重、只需读取一遍并按 $B$ 均摊，其访存量为 $M_{\mathrm{lin}}=b_wN/B+k_{\mathrm{act}}Ldb_a$——批大小对 decode 的全部影响都集中于这一项。注意力部分涵盖 KV 读取与打分，其计算量为 $C_{\mathrm{attn}}=4Ldn$，访存量为 $M_{\mathrm{attn}}=2Ld_{kv}b_{kv}n$；由于每条序列必须读取自己的整段 KV、无法跨批均摊，这一项随上下文长度线性增长，当步的 KV 写入仅为读取量的 $1/n$ 而略去。两部分在层内串行执行，故每 token 的耗时为二者各自 roofline 之和，加上每步的固定开销 $T_0$：

$$\tau(P)=\max\left\lbrace \frac{2N}{F(P)},\ \frac{M_{\mathrm{lin}}}{W}\right\rbrace +\max\left\lbrace \frac{4Ld\bar n}{F(P)},\ \frac{2Ld_{kv}b_{kv}\bar n}{W}\right\rbrace +\frac{T_0}{B},\tag{10}$$

其中对整段生成取平均上下文 $\bar n=s+n_o/2$，系统级 decode 吞吐为 $X_{\mathrm{dec}}(P)=1/\tau(P)$，逐 token 时延为 $\mathrm{TPOT}=B\tau$。

两个部分的算术强度决定了分段结构。线性部分的强度 $I_{\mathrm{lin}}=C_{\mathrm{lin}}/M_{\mathrm{lin}}\approx 2B/b_w$ 随批线性增大，而注意力部分的强度 $I_{\mathrm{attn}}=C_{\mathrm{attn}}/M_{\mathrm{attn}}=2d/(d_{kv}b_{kv})=2g/b_{kv}$ 是一个与 $B$、$n$ 都无关的小常数（FP16 KV 下 MHA 为 1、GQA-8 为 8）。由 §1 的判定规则，某部分在 $R(P)=I$ 处、即相对频率 $\phi=I/R_{\max}$ 处从计算受限切换为访存受限，对应的分界功率为

$$P_i=P_{\mathrm{stat}}+(P_{\max}-P_{\mathrm{stat}})\left(\frac{I_i}{R_{\max}}\right)^{\gamma},\qquad i\in\lbrace \mathrm{attn},,\mathrm{lin}\rbrace .\tag{11}$$

在通常的服务批大小下（$B>g,b_w/b_{kv}$，即权重与 KV 同精度时 $B>g$）有 $I_{\mathrm{attn}}<I_{\mathrm{lin}}$，故随功率升高，注意力部分先切换、线性部分后切换，$P_{\mathrm{attn}}<P_{\mathrm{lin}}$，功率轴被分为三段。

当 $P\le P_{\mathrm{attn}}$ 时两部分都受频率限制，处于计算受限段。此时吞吐

$$X_{\mathrm{dec}}(P)=\frac{F(P)}{2N+4Ld\bar n}=\frac{\eta_C^{\mathrm{dec}}F_{\mathrm{peak}}}{2N+4Ld\bar n},\phi(P),\tag{12}$$

与 prefill 同形、正比于 $\phi(P)$，且与批大小无关，因为每 token 的计算量是固定的，增大批只提高并行度而不减少总计算量。

当 $P_{\mathrm{attn}}<P<P_{\mathrm{lin}}$ 时进入混合段。注意力部分已被 KV 读取带宽限死，成为一段不随功率缩短的固定时间，而线性部分的 GEMM 仍随频率加速，故

$$X_{\mathrm{dec}}(P)=\left[\frac{2N}{F(P)}+\frac{2Ld_{kv}b_{kv}\bar n}{W}\right]^{-1}.\tag{13}$$

吞吐仍随功率上升，但增加的功率只作用于占比越来越小的 GEMM 时间上，收益按双曲线衰减。

当 $P\ge P_{\mathrm{lin}}$ 时两部分都受带宽限制，进入与功率无关的访存受限平台：

$$X_{\mathrm{dec}}(P)=\frac{WB}{b_wN+k_{\mathrm{act}}BLdb_a+2Ld_{kv}b_{kv}B\bar n}.\tag{14}$$

在小批时该式近似为 $WB/(b_wN)$，受权重带宽限制、平台高度随 $B$ 线性抬升；在大批时趋于 $\eta_M W_{\mathrm{peak}}/(2Ld_{kv}b_{kv}\bar n)$，受 KV 带宽限制，是一条无论如何增大批也无法突破的上限。将切换条件代入可验证三段在 $P_{\mathrm{attn}}$、$P_{\mathrm{lin}}$ 处取值连续（斜率不连续），整条 $X_{\mathrm{dec}}(P)$ 单调不减、上凸，以式 (14) 为水平上限。

由于两个分界功率均不含 $n$，它们与上下文长度无关：在固定功率下，一次生成自始至终停留在同一段内，上下文变长只改变段内数值（混合段中 KV 读取时间占比升高、边际收益变小，平台高度按 $1/\bar n$ 下降），而不改变所处的段。此外还有两点边界情形值得注明：当批极小以致 $B<g,b_w/b_{kv}$ 时切换顺序反转，且因两部分强度都很小，几乎整个功率区间都访存受限，这解释了 $B=1$ 解码时降低功率几乎不损失吞吐的现象；当批大到 $B\ge b_wR_{\max}/2$ 时 $P_{\mathrm{lin}}$ 超出 $P_{\max}$，平台移出可行区间，decode 全程计算受限。

## 5 goodput：带时延约束的有效吞吐

上述吞吐尚未计入服务质量约束。给定首 token 时延目标 $\mathrm{TTFT}\le T_1$ 与逐 token 时延目标 $\mathrm{TPOT}\le T_2$，定义 goodput 为两条 SLO 同时满足时的生成吞吐，否则记为零：

$$G(P)=X_{\mathrm{dec}}(P)\cdot\mathbf{1}!\left[\mathrm{TTFT}(P)\le T_1\ \wedge\ \mathrm{TPOT}(P)\le T_2\right].\tag{15}$$

由于 TTFT 与 TPOT 都随功率单调下降，每条 SLO 都等价于一个最低功率门槛。TTFT 约束由式 (9) 反解，给出

$$P_1=P_{\mathrm{stat}}+(P_{\max}-P_{\mathrm{stat}})\left(\frac{Bs,(2N+2Lds)}{\eta_C^{\mathrm{pre}}F_{\mathrm{peak}},T_1}\right)^{\gamma},\tag{16}$$

若括号内的量大于 1，则表示即便满功率也无法达标，该约束在此负载下无解，只能通过减小 $B$ 或 $s$ 来满足。TPOT 约束则须先判断可行性：由式 (14)，TPOT 的最小值在平台段取得，为 $B$ 倍的平台 $\tau$ 加固定开销，若此最小值仍大于 $T_2$，则加功率无济于事（平台与功率无关），只能靠减批或缩短上下文来满足——这是三段结构最直接的工程推论。若可行，则由混合段的式 (13) 令 $B\tau\le T_2$ 反解出最低功率

$$P_2=P_{\mathrm{stat}}+(P_{\max}-P_{\mathrm{stat}})\left(\frac{2NB}{\eta_C^{\mathrm{dec}}F_{\mathrm{peak}}\left(T_2-T_0-2Ld_{kv}b_{kv}B\bar n/W\right)}\right)^{\gamma}\tag{17}$$

（若解出的 $P_2$ 低于 $P_{\mathrm{attn}}$，则改用计算受限段的式 (12) 反解，形式相同）。综合两条约束，goodput 在 $P\ge P_{\min}:=\max\lbrace P_1,P_2\rbrace $ 时等于 $X_{\mathrm{dec}}(P)$，否则为零。因此功率的合理工作区间为 $[P_{\min},,P_{\mathrm{lin}}]$：低于 $P_{\min}$ 将违反 SLO，高于 $P_{\mathrm{lin}}$ 则吞吐不再增长而徒耗电能。