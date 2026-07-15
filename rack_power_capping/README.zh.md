# rack_power_capping — 机架级功率封顶求解器（共享内核）

本目录只留 **共享求解器内核**。机架级规划的**分析、图、文档、经济性**已统一到
[../workload_analysis/](../workload_analysis/)（按 10 类文献用途分类，V100 + H200；见
[../workload_analysis/PLANNING.zh.md](../workload_analysis/PLANNING.zh.md) 与
[../workload_analysis/README.md](../workload_analysis/README.md)）。

| 文件 | 作用 |
|---|---|
| [solve_workloads.py](solve_workloads.py) | **求解器内核**：`load_workload`（fitlib 拟合单卡曲线）、`solve_opt` / `solve_tdp`（整数卡、每相 ≥1、`N_GPU_MAX` 插槽墙、cap∈实测区间、decode 不超饱和 cap、花满预算）、`sweet_spot`（tok/J 按**实测功耗**计）、`cont_bound`。`workload_analysis/{v100,h200}/solve_rack_capping.py` 直接 import 它并重定向数据目录/场景参数——不重复实现，两侧结果不会漂移。 |

内核也可独立运行做自检（`python3 solve_workloads.py` → 打印 6 个 app-class 的配方表、
写 `workloads_results.csv`），但正式展示以 `workload_analysis/` 的按类图表为准。
