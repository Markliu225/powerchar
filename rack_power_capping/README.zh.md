# rack_power_capping — 机架级功率封顶规划

单卡的 P↔T 曲线（见根目录 [MODEL_AND_RESULTS.zh.md](../MODEL_AND_RESULTS.zh.md)）向上聚合成
机架级决策：给定机架功率预算 W 与**物理插槽上限 N_GPU_MAX**，按 **workload 类别**决定
prefill / decode 各配几张卡、每张卡 cap 到多少瓦，以及每类的吞吐/能效/回本。

**按硬件分目录**——每种卡的曲线、求解结果、图表自成一套：

| 目录 | 硬件 | 状态 |
|---|---|---|
| [v100/](v100/) | Tesla V100-DGXS-32GB（portfolio v3 实测曲线） | ✅ 完整 |
| h200/ | H200（曲线待 `data_h200` 决胜数据） | ⏳ 计划 |

## v100/ 内容

| 文件 | 作用 |
|---|---|
| [WORKLOADS.zh.md](v100/WORKLOADS.zh.md) | **唯一文档**：规划框架 → workload 分类（6 类 × 10 实测负载）→ 每类机架配方 → 分析与经济性 |
| [solve_workloads.py](v100/solve_workloads.py) | **唯一求解器**（含 N_GPU_MAX 物理约束）：分类定义 + 按类求配方 → `workloads_results.csv` |
| [plot_workloads.py](v100/plot_workloads.py) | → `fig_workloads.png`（按类别标注的配方三联图） |
| [economics.py](v100/economics.py) | 按类回本分析 → `economics.csv` / `fig_payback.png` |

```bash
cd v100
python3 solve_workloads.py   # 每类 workload 的机架配方
python3 plot_workloads.py    # 配方图
python3 economics.py         # 按类回本
```
