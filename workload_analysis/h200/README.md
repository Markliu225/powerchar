# H200 版:按使用类型的功率曲线 + 机架 power capping

把 [../](../)(V100)的两步流程跑在 **H200 数据**(`../../data_h200/`,F_MAX 1980 MHz)上。
分类、P:D 比例、类别↔实测 workload 映射与 caveat 全部**从 V100 脚本 import**,不另拷贝;曲线加载器
与机架求解器同样直接复用(`../plot_power_curves.py`、`rack_power_capping/v100/solve_workloads.py`),
仅在运行时把数据目录 / 功率区间 / 场景参数重定向到 H200。

## 数据修订(2026-07-13)——两点变化

1. **prefill 改为「固定 700 W cap 下扫 SM 时钟」**,不再扫 cap。所以 prefill 的功率轴 = **实测功耗
   `power_avg_w`**(~300–710 W),不是设定 cap。**这直接修掉了旧数据的低 cap 未兑现问题**(旧数据里
   cap=200 W 档 prefill 实测 ~310–375 W)。decode 仍是 cap 扫(200–700 W)。加载器 `_read` 自动识别:
   cap 有变化用 cap,否则用实测功耗——V100 不受影响。
2. **`classify-qwen7b`(Extract 的映射)在本修订被移除**,故 **Extract 类在 H200 缺席(10 类里剩 9 类)**。
   脚本自动跳过无数据的类,图与经济性都相应少一类。

## 机架场景(等比缩放自 V100)

| | V100 | H200 |
|---|---|---|
| 预算 | 5 kW | **14 kW** (= 5 kW × 700/250) |
| 槽位 / TDP / TDP 下装卡 | 32 / 250 W / 20 | 32 / 700 W / 20 |
| 功率区间 | [100, 250] W | [200, 700] W(decode);prefill ~300–710 W(实测) |

## 结果与 V100 的结构性差异

- **两相现在都有实测区间内的能效甜点**(旧数据 prefill 因 cap 未兑现看不到甜点,环钉在 700 W):
  分相 tok/J 见 `workload_power_curves.csv`——**prefill 甜点 515–575 W**、**decode 甜点 337–414 W**。
  clean 的时钟扫 prefill 是关键改善。
- **机架:9/9 类全部撞 32 槽墙**,增益 **+14.6% ~ +28.1%**。Code(73.5:1)现在装满 **16+16=32 卡**
  ——旧数据里 Code 因 prefill 吃到 ~650 W 而**预算先咬合**(只装 29 卡);新的干净 prefill 甜点落在
  ~530 W,预算够填满槽位,所以又回到"槽位墙先咬合"。
- **rack 配方的 cap**:prefill 288–700 W、decode 344–442 W(decode 饱和点 ~690 W,加瓦不加 token,
  故 cap 由预算劈分决定)。
- decode 拟合 R² 0.83–0.93、prefill 0.96–0.99(旧数据 Extract 的 R²<0 已随该类移除而消失)。

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
python3 workload_analysis/plot_composite_economics.py   # 顺带刷新 H200 经济性
```
