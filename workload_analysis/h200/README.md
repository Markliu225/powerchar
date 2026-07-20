# H200 版:按使用类型的功率曲线 + 机架 power capping

把 [../](../)(V100)的两步流程跑在 **H200 数据**(`../../data_h200/`,F_MAX 1980 MHz)上。
分类、P:D 比例、类别↔实测 workload 映射与 caveat 全部**从 V100 脚本 import**,不另拷贝;曲线加载器
与机架求解器同样直接复用(`../curves_lib.py`、`rack_power_capping/solve_workloads.py`),
仅在运行时把数据目录 / 功率区间 / 场景参数重定向到 H200。

## 数据修订(2026-07-15)——`classify-qwen7b` 补测,Extract 回归 → **10 类齐**

上一版缺 `classify-qwen7b`(Extract 的映射锚),H200 只有 9 类。本次补测了 classify-qwen7b,**Extract 类
回归,H200 恢复到全 10 类**。现 `data_h200/` 共 **8 个实测 workload**(仍缺 qwen3chat-4b / qwen3think-4b,
但二者非类锚,不影响类数)。Extract 机架增益 **+59%**——远高于其它类(+15%~+28%),是**结构性**的:
classify 的 decode 极弱(256×8,平台 ~396 tok/s),Extract 严重 **decode 受限**,capping 把 decode 卡从
TDP 的 18 张packing 到 29 张(1.6×)。OPT 配方(prefill@600 W / decode@421 W)与 TDP(700 W)**都落在
已兑现区**,故增益可信,非伪影。

> ⚠️ **classify-qwen7b 数据质量(全 10 类里最弱的一个,如实标注)**:
> - 它是**唯一仍用 v3「cap 扫」而非 v4「时钟锁扫」测 prefill 的 workload**,**低 cap 未兑现**:
>   cap=200 W 实测 337 W(+69%)、cap=313 W 才咬合。加载器按 cap 轴拟合,故 **prefill 曲线的低 cap 段
>   (<367 W)不可信**;但 Extract 的运行点在 ≥600 W 的已兑现区,recipe 不受影响。
> - **decode 近平坦**(256×8,257→396 tok/s),可加模型 **拟合 R²<0**(`fig_workload_pd` 映射注已标
>   "read the dots");平台量级 ~396 tok/s 大致可信,但曲线形状不可信。
> - **建议**:若要把 Extract 做干净,应对 classify-qwen7b **补测 v4 时钟锁扫 prefill + 更密的 decode 网格**
>   (抓住膝点),与其余 7 个 workload 口径统一。

## 数据修订(2026-07-14)——`fastchat-qwen15b` 补测

上一版 `fastchat-qwen15b` 因配置滑差实际加载了 Qwen2.5-3B(与 translate-qwen3b 重复),本次**补测为
真正的 Qwen2.5-1.5B**(decode ~1029 vs translate ~637 tok/s,已是独立 workload;prefill 时钟扫吞吐
高达 ~10 万 tok/s)。它是 **chat 类的成员之一,但不是本目录 class↔workload 映射的锚(chat 锚 = chat-phi3)**,
故**按类的机架配方与经济性数值不变**;补测只影响 portfolio 级 P↔T 验证——`../../data_h200/` 的
`portfolio_fits.csv`、`fig_{decode,prefill}_models.png`、`fig_portfolio_grid.png`、`fig_tmax_validation.png`
已按当前 7 个实测 workload(含真 fastchat)重拟(fastchat:prefill 统一模型 R²=0.98、decode 可加 R²=0.97,
BW_eff ≈207 GB/s)。其余 6 个 workload 的 CSV 与 07-13 版逐字节一致,故按类分析结果稳定。

## 数据修订(2026-07-13)——两点变化

1. **prefill 改为「固定 700 W cap 下扫 SM 时钟」**,不再扫 cap。所以 prefill 的功率轴 = **实测功耗
   `power_avg_w`**(~300–710 W),不是设定 cap。**这直接修掉了旧数据的低 cap 未兑现问题**(旧数据里
   cap=200 W 档 prefill 实测 ~310–375 W)。decode 仍是 cap 扫(200–700 W)。加载器 `_read` 自动识别:
   cap 有变化用 cap,否则用实测功耗——V100 不受影响。
2. **`classify-qwen7b`(Extract 的映射)在本修订被移除**,故当时 **Extract 类在 H200 缺席(10 类里剩 9 类)**。
   脚本自动跳过无数据的类。**(已于 2026-07-15 补测回归,见上;此条为历史记录。)**

## 机架场景(等比缩放自 V100)

| | V100 | H200 |
|---|---|---|
| 预算 | 5 kW | **14 kW** (= 5 kW × 700/250) |
| 槽位 / TDP / TDP 下装卡 | 32 / 250 W / 20 | 32 / 700 W / 20 |
| 功率区间 | [100, 250] W | [200, 700] W(decode);prefill ~300–710 W(实测) |

## 结果与 V100 的结构性差异

- **两相现在都有实测区间内的能效甜点**(旧数据 prefill 因 cap 未兑现看不到甜点,环钉在 700 W):
  分相 tok/J 见 `workload_power_curves.csv`——**prefill 甜点 515–575 W**、**decode 甜点 256–411 W**。
  clean 的时钟扫 prefill 是关键改善。
  > **tok/J 一律 = 吞吐 ÷ 实测功耗 `power_avg_w`**(不是设定 cap,更不是 CSV 里那个 energy-counter
  > 的 `tok_per_joule` 列——不可信)。decode 低 cap 未兑现(实测 > cap),故用实测功耗后低 cap 段的
  > tok/J 明显回落、甜点右移(旧口径 T/cap 会把那一段虚高)。机架级 tok/J 同样按 Σ 实测功耗计
  > (`opt_w_measured` 列,≈ 预算的 98%)。
- **机架:10/10 类全部撞 32 槽墙**,增益 **+14.6% ~ +28.1%**(Extract 因 decode 严重受限达 **+59%**,
  见上文 07-15 注)。Code(73.5:1)装满 **16+16=32 卡**——旧数据里 Code 因 prefill 吃到 ~650 W 而
  **预算先咬合**(只装 29 卡);干净 prefill 甜点落在 ~530 W,预算够填满槽位,所以回到"槽位墙先咬合"。
- **rack 配方的 cap**:prefill 288–700 W、decode 259–442 W(decode 饱和点 ~690 W,加瓦不加 token,
  故 cap 由预算劈分决定)。
- prefill 拟合 R² 0.96–1.00、decode 0.80–0.93;**Extract(classify)decode 近平坦、R²<0**(见上文注)。

## 数据质量说明

- **旧的低 cap 未兑现问题已修复**(prefill 时钟扫,功率轴=实测)。decode 仍 cap 扫,memory-bound
  下实测功耗略低于 cap;经济性里能耗一律按**实测 `power_avg_w`** 计(见 [../ECONOMICS.md](../ECONOMICS.md))。
- Chat 的 chat-phi3 prefill 网格较疏(6 点),图上以实测点为准。
- Extract 缺席(classify-qwen7b 无数据)。

## 文件

- `plot_power_curves.py` → `fig_workload_power_throughput.png` · `fig_workload_power_tokj.png` ·
  `workload_power_curves.csv`(prefill 轴=实测功耗,decode 轴=cap)
- `solve_rack_capping.py` → `fig_workload_rack_capping.png` · `workload_rack_capping.csv`
- 综合 workload 的经济性(V100 & H200 同图对比)在父目录:[../ECONOMICS.md](../ECONOMICS.md)

## 复现

```bash
python3 workload_analysis/h200/plot_power_curves.py
python3 workload_analysis/h200/solve_rack_capping.py
python3 workload_analysis/plot_profit_model.py   # 顺带刷新混合 workload 经济性(V100 & H200)
```
