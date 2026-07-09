# 离线 H200 一键测量 —— 搬运与运行手册

把 portfolio 功率-cap 实验（10 个 workload × prefill+decode，**方法学 v3**）搬到一台**离线**的
H200 主机上跑。测量规范遵循 [../../H200_EXPERIMENT_MANUAL.zh.md](../../H200_EXPERIMENT_MANUAL.zh.md)：
能量累加器读窗口功率（§7.3）、降频原因门控（§7.4）、约束落盘 meta.json（§2.3）。
代码全部 GPU 无关：cap 网格从本机 `-pl` 约束自动生成，f_max 写进 meta.json 供拟合使用。

## 一、在线机器上备料（两样东西）

**1. 代码**：整个仓库（或最小集：`code/` + `pt_cap_gpu1/portfolio/`）。

**2. 模型权重（~44 GB）**：
```bash
python3 download_models.py          # 下到本机 HF cache；结束时打印要拷贝的目录清单
```
拷到离线机的同一位置（默认 `~/.cache/huggingface/hub/`）：
```bash
rsync -a ~/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct \
         ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct \
         ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct \
         ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct \
         ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507 \
         h200-host:~/.cache/huggingface/hub/
```
> HF cache 放在别处的话，离线机上 `export HF_HOME=<那个位置的上级>`。

**离线机软件需求**（preflight 会逐项检查）：CUDA 版 torch（H200 需 sm_90 构建，torch≥2.9）、
`transformers>=4.51`（Qwen3）、`pynvml`、`numpy`；`matplotlib` 可选（缺了就把数据带回来再画）。

## 二、H200 上运行（一键）

```bash
cd pt_cap_gpu1/portfolio
sudo nvidia-smi -pm 1                      # 持久模式（手册 §2.2；能量计数器保持单调）

# 1. 先验机（~4 分钟：chat-phi3 单 workload、4 个 cap，全链路走一遍）
./run_all.sh --gpu 0 --smoke

# 2. 全量（10 workload × 2 相 × ~10 cap；数小时，可随时中断、重跑即断点续跑）
./run_all.sh --gpu 0
```
权限三选一：root 运行 / `nvidia-smi` 免密 sudo / `SUDO_PASS='...' ./run_all.sh`。

`run_all.sh` 做三件事：**preflight 自检门**（依赖、NVML、`-pl` 权限、5 个模型齐不齐——
有缺项直接退出并打印修法）→ **扫描**（cap 网格默认取本机 `[min,max]` 均分 10 点；
H200 SXM 即约 [100..700]，NVL 约 [.. 600]）→ **拟合出图**。

产物全部落在 `data_h200/`（目录名自动按 GPU 名生成，可用 `--outdir` 指定）：
```
data_h200/
  meta.json                    # GPU 名/driver/约束/f_max/温度阈值/cap 网格/方法学标签
  <id>_prefill.csv ×10         # 每 cap 一行；power_avg_w 为能量法窗口功率
  <id>_decode.csv  ×10         # 含 ctx_eff/steps/spread/n_runs/throttle_mask/thermal_frac
  fig_decode_models.png        # 可加三阶段 vs 旧 min() 逐 workload 对比
  fig_portfolio_grid.png       # prefill+decode 总览
  fig_tmax_validation.png      # decode 天花板：理论 vs 实测
  decode_model_compare.csv / portfolio_fits.csv
  run_all.log
```
**带回来的就是整个 `data_h200/` 目录。**

## 三、H200 上的预期与判读

- **cap 网格**：自动 [min..max]（SXM ~700W）。部分 workload 的自然功耗到不了高 cap
  （decode 大多在 300–450W 就进平台）——高端点会在功率轴上聚拢，属正常，`P(T)` 曲线以
  实际功率（能量法）为横轴，不受影响。
- **decode 平台会比 V100 高一个量级**（HBM3e ~4.8 TB/s vs V100 0.9），但**曲线形状**
  （V²f 凸 prefill、三阶段 decode、平台 `T_max=B·BW_eff/D_mem`）应当复现——这正是这次
  跨硬件验证的目的。
- **`power.draw` 在 Hopper 是 ~1s 滑动平均**：CSV 的 `power_avg_w` 已经用能量累加器法，
  不受此影响；`power_sample_avg_w` 列保留采样均值供对照（两者差异大即证明此点）。
- **降频门控**：`thermal_frac>5%` 的点采集时已自动冷却重测一次；CSV 保留 `throttle_mask`
  （期望只有 SwPowerCap 0x4 / ApplicationsClocksSetting 0x2 位）供事后审计。
- 结束后脚本自动把 `-pl` 恢复到运行前的值；离场前可 `nvidia-smi -q -d POWER` 复核。

## 四、回来之后

```bash
# 若离线机没装 matplotlib，把 data_h200/ 拷回本仓库同目录后：
PORTFOLIO_DATA=data_h200 python3 plot_decode_models.py
PORTFOLIO_DATA=data_h200 python3 plot_portfolio.py
# 图和拟合表会写进 data_h200/，不会覆盖 V100 的结果
```
V100 基线在 `data/`（提交在 git），直接可做跨硬件对比（同一批 workload、同一套方法学）。

## 常见问题

| 症状 | 处理 |
|---|---|
| preflight 报某模型缺失 | 按 §一 重拷该 `models--...` 目录（注意连同 `snapshots/`、`blobs/` 整目录拷） |
| `-pl` 权限失败 | root / 免密 sudo / SUDO_PASS 三选一；云实例先 `sudo nvidia-smi -pl <当前值>` 验证 |
| 扫描中途被打断 | 直接重跑同一条命令：已完成的 workload 自动跳过（`FORCE=1` 强制重来） |
| Qwen3 加载报 KeyError 'qwen3' | transformers 太旧，需 ≥4.51 |
| torch 报 sm_90 不支持 | torch 不是 CUDA 12.x/sm_90 构建，换新版 |
