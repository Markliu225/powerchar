# LLM 推理工作负载分析 —— 生产负载分类与按类功率规划

用**生产 trace 实测的负载分类**刻画 LLM 推理流量:每类的 prefill(输入)/ decode(输出)token
总量之比 `ρ̄ = ΣL_p / ΣL_d`(论文 II-C 式 (1)),为上层机架功率规划
([PLANNING.zh.md](PLANNING.zh.md))提供真实的负载形状,替代拍脑袋的 1:1 / 1:10。

## 分类法(论文 II-C,当前口径)

**7 个生产负载类**,P:D 从 0.83 到 110.7,全部来自公开生产 trace
(数据文件 [workload_classes.csv](workload_classes.csv)):

| 类别 | ρ̄ | 来源 | | 类别 | ρ̄ | 来源 |
|---|--:|---|---|---|--:|---|
| 推理 | 0.83 | ServeGen | | 长上下文对话 | 35.1 | Mooncake |
| 助手 API | 7.7 | ServeGen | | Agentic 工具调用 | 47.2 | Mooncake |
| 多模态图文 | 9.4 | ServeGen | | 代码补全 | 110.7 | DynamoLLM/Azure'24 |
| 对话 | 15.5 | DynamoLLM/Azure'24 | | | | |

每类按**最近 decode 上下文规模**映射到一个实测 workload 的功率曲线(`curves_lib.MAP`;
映射 caveat 在各 CSV 的 `mapping_caveat` 列),类形状 ρ̄ 进入机架求解器的 token 平衡方程。

> 旧版 InstructGPT/Dolly 十类分类(2022 请求式口径)及其方法学文档
> [CLASSIFICATION.zh.md](CLASSIFICATION.zh.md) / [REFERENCES.zh.md](REFERENCES.zh.md) 保留作底稿,
> 统计管线 `analyze.py` / `plot.py` / `workload_ratios.csv` 仍可复现,但**图表与经济模型均已切换到
> II-C 七类**。

## 文件

- `workload_classes.csv` —— **II-C 七类数据**(L̄p、L̄d、ρ̄、来源、锚定 workload),四个绘图/求解
  脚本的唯一分类数据源
- `curves_lib.py` —— **共享库**(不单独运行):分类法(NAME/MAP/CAVEAT/BANDS)、曲线加载
  (`fitlib.fit_*_theory` 拟合)、每类功率曲线图构造;v100/h200 包装器与 solve_rack_capping 复用,
  一套实现两硬件不漂移
- `v100/` `h200/` —— **两硬件对称产物**(V100:5 kW / ≤32 槽 / cap 100–250 W;H200:14 kW / ≤32 槽 /
  cap 200–700 W,prefill 时钟扫):
  - `plot_power_curves.py` → `fig_workload_power_throughput.png` · `fig_workload_power_tokj.png`
    (每类 2×4 面板,prefill/decode 两相,log-y)· `workload_power_curves.csv`
  - `solve_rack_capping.py`(优化内核 import 自 `../../rack_power_capping/solve_workloads.py`)
    → `fig_workload_rack_capping.png` · `workload_rack_capping.csv`
- `plot_profit_model.py` —— **论文经济模型**(式 (1)–(15))按 II-C 七类混合核算 CAP vs TDP 的
  **累计净现金流**(起点 −K,末端 = Φ(n)):混合 w_j = **研究查证的请求份额 r_j**(ServeGen 实测
  请求量、Copilot/Cursor 补全规模、OpenRouter 份额;逐类置信度见 ECONOMICS.md §3)× 各类 trace
  每请求 token 数;各类 2026 分档定价(式 8)× **1/3 小模型折价**(`PRICE_SCALE`,见 ECONOMICS.md
  §6);N_j ∝ w_j/X_j 按类分机架,1 MW 集群。两组图:
  `fig_profit_model.png`(2×4,设备 × λ=0/10/20/30%,标注 T× 与 G)·
  `fig_profit_mix.png`(1×2,每类份额 ±20% 逐类扰动敏感带)· `profit_model.csv`;
  详见 [ECONOMICS.md](ECONOMICS.md)
- 旧分类管线:`analyze.py`(拉数据+分词 → `workload_ratios.csv`)· `plot.py`(→
  `fig_workload_pd.png`)· `data/`(Azure conv/code、BurstGPT trace 缓存)

## 复现

```bash
python3 workload_analysis/v100/plot_power_curves.py
python3 workload_analysis/v100/solve_rack_capping.py
python3 workload_analysis/h200/plot_power_curves.py
python3 workload_analysis/h200/solve_rack_capping.py
python3 workload_analysis/plot_profit_model.py
```
