# 综合 workload 的功率封顶经济性 —— 非线性收益曲线（V100 & H200）

把 10 种使用类型按**数据集规模加权**共存成一个综合 workload，问一个仓库既有经济分析没回答的
问题:**一旦把电价当作真实成本，选择运行 cap 时利润长什么样?** 答案是——**非线性**，且最优点随
电价移动。

> ⚠️ 这是**纯吞吐/能耗**经济模型，**未建模延迟/SLO**。综合里 92%(Chat+Code)是交互类,压 cap 会
> 抬 TTFT/逐字延迟——真实会掉收入或违 SLO。所以下面算出的"利润最优 cap"是**对交互流量能压多深的
> 上界**,不是目标值(与机架长文 [WORKLOADS.zh.md §7](../rack_power_capping/v100/WORKLOADS.zh.md) 同款局限)。

> 脚本 [`plot_composite_economics.py`](plot_composite_economics.py) →
> [`fig_composite_economics.png`](fig_composite_economics.png)(利润 vs cap)·
> [`fig_composite_elec_sensitivity.png`](fig_composite_elec_sensitivity.png)(利润/收益 vs 电价)·
> [`composite_economics.csv`](composite_economics.csv)

## 与仓库既有经济分析的区别

[rack_power_capping/v100/economics.py](../rack_power_capping/v100/economics.py) 固定机架功率预算,
让两个机队(capped / 不 capped)**能耗相抵**,所以它的回本是**线性、与电价无关**的。这里问相反的、
更真实的问题:综合 workload 下,**能耗不相抵**时,利润随 cap 与电价如何变化。

## 综合 workload(数据集规模加权)

- **权重 φ_i**:取 `workload_ratios.csv` 的 `n`(样本量)归一化。结果由 **Chat 68% + Code 24%**
  主导,其余 8 类长尾(合计 8%)。**Caveat**:`n` 是**数据集规模**(Chat/Code 是 2.5 万/8.8 千的
  生产 trace,8 个 Dolly 类各几百样本),不是严格的生产流量占比——是"真实比例"的**代理**,可替换旋钮。
- **每请求 token 数 Lp_i / Ld_i**:取实测均值 `pre_mean` / `dec_mean`(真实请求长度)。
- **每相吞吐 T_pre_i(cap) / T_dec_i(cap)** 与 **实测功耗 P_pre_i(cap) / P_dec_i(cap)**:来自该类
  映射 workload 的 fitlib 拟合曲线与实测 `power_avg_w`(V100 portfolio / data_h200)。**注意口径**:
  token 数来自真实 trace,吞吐/功耗曲线来自映射 workload 在**其自身上下文规模**下的实测(如 Chat 用
  chat-phi3 的 ctx=256 decode 曲线,而 trace 里 Chat 请求上千 token)——因 `T_max ∝ 1/C`,这会**高估
  decode 吞吐、低估 decode 成本**,使利润偏乐观。是标准映射 caveat(WORKLOADS.zh.md §2),$ 数值取近似。

## 为什么是非线性(核心机制)

服务固定需求,单请求要租的 GPU 秒数
`g(p) = Σ_i φ_i ( Lp_i/T_pre_i(p) + Ld_i/T_dec_i(p) )`。单请求成本有两项**方向相反**:

| 成本项 | 随 cap 上升 | 极小点 |
|---|---|---|
| 能耗 = PUE·电价·Σφ_i(Lp_i·P_pre_i/T_pre_i + Ld_i·P_dec_i/T_dec_i) | ↑(瓦特升,GPU 秒降) | 能效甜点附近(U 型) |
| CapEx = (GPU价/寿命)·g(p) | ↓(每卡产出更多 token) | 最高吞吐(高 cap) |

能耗按**实测功耗 P(cap)** 计,不按设定 cap:电费付的是实际消耗。这很关键——V100 上 memory-bound 的
decode 实测功耗低于 cap(250 W 档只画 227 W),H200 低 cap 档反而**超过** cap(200 W 档 prefill 实测
~310–375 W)。用设定 cap 会单边虚增 capping 收益,故一律用实测 `power_avg_w`。CapEx 只含 GPU 时间
(GPU 秒×摊销),不含功率。

收入由需求固定,所以 `利润/Mtok(p) = 收入/Mtok − (能耗+CapEx)/Mtok(p)` 是**非线性**的,有一个
内部最优 cap;**电价越贵,最优点越往能效甜点(低 cap)滑,电价越便宜越往最高吞吐(TDP)靠**。

## 两种 cap 策略

- **UNIFORM**:两相同一个 cap(图1 的单旋钮曲线)。
- **DISAGGREGATED**:每类每相各自在自己曲线上取 cost 最小 cap(机架配方真正的做法)。理论上比 uniform
  更省(min 分离和 ≤ 单一 cap),V100 上明显;H200 上因 CapEx 太重,两者都≈TDP,差别可忽略(见结果)。

## 结果(见 `composite_economics.csv`)

**V100(便宜卡 $2.5k):经典非线性,capping 值得。** 电价 $0.05→$1.00/kWh,利润最优 cap 从
**250 W(=TDP)滑到 138 W**;capping 相对不 cap 的收益从 0 涨到 **+$0.008/Mtok(disagg)**,
且是**凸增长**(电价越高,加速)。

**H200(贵卡 $30k):CapEx 主宰,就该开满 TDP,capping 基本不划算。** 现实电价区间内 uniform 与
disaggregated 最优 cap **都≈700 W(=TDP)**——贵卡靠高吞吐摊薄折旧,压 cap 几乎不省:capping 收益
≤ **$0.0002/Mtok**(即便 $1/kWh、disagg),实务上可忽略;要到更贵的电价(>$1/kWh)红利才隐约出现。
小模型 + 低 token 价下,H200 在 ~$0.45/kWh 以上 **每 token 亏钱**(绝对利润为负)。结论:H200 上
这套 workload 就该 TDP 满跑。(H200 数据修订后 prefill 时钟扫、更干净,利润比旧数据高;Extract 因
classify-qwen7b 缺席在 H200 侧剔除,mix 在 9 类上重新归一——V100 仍 10 类,两侧 revenue 差 <0.2%。)

**跨设备结论**:capping 的经济价值取决于 **电价 : 卡价** 之比。便宜/已折旧的卡 + 贵电 → capping
显著回本(V100);贵的新卡 + 便宜电 → 开满更划算(H200),capping 红利小到可忽略。

## 经济旋钮(全部可改,脚本顶部)

GPU 价 $2.5k(V100)/$30k(H200) · 3 年直线摊销 · PUE 1.3 · token 价 $0.05/$0.20 per Mtok
(输入/输出,同仓库) · 电价曲线 $0.05–$1.00/kWh,敏感性图扫到 $1.5(高端含需量电费/碳价)。

## 局限

- **未建模延迟/SLO(最重要)**:模型令收入与 cap 无关,等于假设压 cap 不损服务质量。但 92%
  的量是交互类,压 cap 抬 TTFT/逐字延迟——真实要么掉收入要么违 SLO。**利润最优 cap 应视为交互
  流量的压 cap 上界,而非目标。** 要落地须给交互类的 cap 设 SLO 下界后再算。
- **UNIFORM cap 低估 H200 capping**:单旋钮无法单独 cap decode;disaggregated 才是可达上限(图2)。
  新数据下两者在 H200 上都≈TDP,差别可忽略。
- **能耗一律按实测 `power_avg_w` 计**,不按设定 cap。H200 修订版 prefill 改为时钟扫(功率轴=实测),
  旧的低 cap 未兑现问题已消失;decode 仍 cap 扫、memory-bound 下实测略低于 cap,故仍用实测功耗。
  **Extract 在 H200 缺席**(classify-qwen7b 无数据),H200 侧 mix 于 9 类重新归一。
- **token 数 vs 曲线上下文规模不一致**(见上节口径说明):$ 数值系近似,倾向乐观。
- **权重 φ 是数据集规模代理**,非严格生产占比;换 φ 重跑即可。
- token 价用小模型口径($0.05/$0.20);H200 跑大模型/高 token 价时经济性另算(本文口径下 H200
  在 $0.5/kWh 以上每 token 亏钱,是"贵卡跑小模型"的真实结论)。

## 复现

```bash
python3 workload_analysis/plot_composite_economics.py
```
