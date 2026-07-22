# H200 版：按生产负载类的功率曲线 + 机架 power capping

把 [../](../)（V100）的两步流程跑在 **H200 数据**（`../../data_h200/`，F_MAX 1980 MHz）上。
**分类 = 论文 II-C 七类生产负载**（[../workload_classes.csv](../workload_classes.csv)，ρ̄ 0.83–110.7），
类↔实测锚定映射与 caveat 从 `../curves_lib.py` import，机架求解器复用
`rack_power_capping/solve_workloads.py`，仅在运行时把数据目录 / 功率区间 / 场景参数重定向到 H200。

## 机架场景（等比缩放自 V100）

| | V100 | H200 |
|---|---|---|
| 预算 | 5 kW | **14 kW**（= 5 kW × 700/250） |
| 槽位 / TDP / TDP 下装卡 | 32 / 250 W / 20 | 32 / 700 W / 20 |
| 功率区间 | [100, 250] W | [200, 700] W（decode）；prefill ~300–710 W（实测） |

## 结果（7 类，`workload_rack_capping.csv`）

- **7/7 类全部撞 32 槽墙**，OPT 增益 **+18%（助手API/长上下文）~ +63%（对话）**；
  对话类增益最高——code-phi3 锚的 decode 在 TDP 下严重受限，capping 把 decode 卡从 16 packing 到 25。
- **配方 cap**：prefill 304–642 W、decode 248–442 W（decode 饱和 ~690 W，加瓦不加 token）。
- **两相都有实测区间内的能效甜点**（prefill 甜点 ~515–575 W、decode ~256–426 W，
  见 `workload_power_curves.csv`）。tok/J 一律 = 吞吐 ÷ **实测功耗** `power_avg_w`
  （不是设定 cap，不是 CSV 的 energy-counter 列）。

## 数据说明（测量历史，与分类无关）

- **prefill = 固定 700 W cap 下扫 SM 时钟**（2026-07-13 修订），功率轴 = 实测功耗——修掉了旧数据
  低 cap 未兑现的问题；decode 仍 cap 扫（200–700 W）。加载器自动识别两种轴。
- `data_h200/` 现有 **8 个实测 workload**（缺 qwen3think-4b / qwen3chat-4b，均非七类锚定，不影响）。
  七类实际用到 5 个锚：longform-phi3、translate-qwen3b、rag-phi3、code-phi3、summarize-qwen7b。
- `fastchat-qwen15b`（07-14 补测为真 Qwen2.5-1.5B）与 `classify-qwen7b`（07-15 补测；prefill 仍是
  v3 cap 扫、低 cap<367 W 未兑现，decode 近平坦 R²<0）**都不是七类的锚定**，其数据质量问题不影响
  本目录任何结果；二者仅参与 `data_h200/` 的 portfolio 级拟合验证。
- chat-phi3 prefill 网格较疏（6 点）——同样非锚定，仅供参考。

## 文件

- `plot_power_curves.py` → `fig_workload_power_throughput.png` · `fig_workload_power_tokj.png` ·
  `workload_power_curves.csv`（prefill 轴=实测功耗，decode 轴=cap）
- `solve_rack_capping.py` → `fig_workload_rack_capping.png` · `workload_rack_capping.csv`
- 混合负载经济性（V100 & H200 同图对比）在父目录：[../ECONOMICS.md](../ECONOMICS.md)

## 复现

```bash
python3 workload_analysis/h200/plot_power_curves.py
python3 workload_analysis/h200/solve_rack_capping.py
python3 workload_analysis/plot_profit_model.py   # 顺带刷新混合 workload 经济性(V100 & H200)
```
