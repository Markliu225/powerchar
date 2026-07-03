# H200 大模型推理 功率 ↔ 吞吐 表征实验手册

本手册规定在 **NVIDIA H200** 上表征大模型推理两个阶段——**prefill（提示摄入，计算受限）**与 **decode（逐 token 生成，访存受限）**——的 **GPU 功率与 token 吞吐关系 `P ↔ T`**，并给出功率调节方法、功率区间、扫描网格、测量时序、遥测规范、数据产物与安全复位流程。

配套测量代码：[code/measure.py](code/measure.py)、[code/power_sampler.py](code/power_sampler.py)、[code/pt_cap_sweep.py](code/pt_cap_sweep.py)、[code/decode_clk_sweep.py](code/decode_clk_sweep.py)、[code/decode_powercap_sweep.py](code/decode_powercap_sweep.py)、[code/measure_dvfs.py](code/measure_dvfs.py)。

> **数值约定**：本手册给出的功率区间、时钟、温度阈值为标称/参考值。H200 的若干动态行为数值（HBM 运行频率、功率地板、带宽饱和拐点、时钟范围）以本机 §2.3 的实测查询为准；实验开始前先查询一次并写入 `meta.json`。

---

## 0. 实验目标

1. 测得 prefill 与 decode 各自的功率–吞吐曲线 `P(T)`，并与解析模型对照拟合。
2. 刻画功率上限与工作频率对吞吐/能效的影响，给出各阶段的能效（tok/J）曲线与拐点。
3. 考察显存频率对 decode 吞吐上限的作用，确定 decode 天花板 `T_max` 的取值与所在功率。
4. 在多个模型规模与两种推理引擎（eager 与优化引擎）下验证曲线形状的一致性。

---

## 1. 硬件规格

H200 有 SXM 与 NVL（PCIe）两种变体，功率与时钟不同；开测前确认所测卡的型号。以下为参考规格，**以本机 §2.3 查询为准**。

| 项 | H200 SXM | H200 NVL (PCIe) |
|---|---|---|
| 架构 / die | Hopper GH100，`sm_90`，132 SM，528 Tensor Core | 同 |
| 显存 | 141 GB HBM3e（6144-bit 总线） | 141 GB HBM3e |
| 峰值显存带宽 β | ~4.8 TB/s | ~4.8 TB/s |
| HBM 运行频率 | ~3201 MHz（运行时固定） | ~3201 MHz |
| FP16/BF16 Tensor（稠密） | ~989 TFLOPS（2:4 稀疏 ~1,979） | 峰值同，持续受 600W 限制偏低 |
| FP8 Tensor（稠密） | ~1,979 TFLOPS（稀疏 ~3,958） | 同上 |
| 最大功率（=默认 `-pl`） | ~700 W（可配置） | ~600 W（可配置） |
| Boost SM 频率 | ~1,980 MHz | ~1,785 MHz |
| Base SM 频率 | ~1,665 MHz | ~1,365 MHz |
| `-pl` 功率下限 | ~200 W（以实测约束为准） | 同量级 |
| 脊点 I\* = Φ/β | ~206 FLOP/byte | ~206 |

**开测前须落定的 5 个量**（决定网格与实验设计）：
1. `-pl` 的 Min / Max / Default（决定功率网格与 decode 的低功率兜底方式）。
2. supported memory clocks 的工作档数（**1 档：显存频率不可调**；**≥2 档：可做显存频率扫描**，见实验 4）。
3. `nvidia-smi --lock-memory-clocks-info` 报告的显存锁定风格（Hopper 预期为 `deferred`）。
4. supported SM(graphics) clocks 的 min → max。
5. 温度阈值：GPU Max Operating / Slowdown / Shutdown，Memory Max Operating。

---

## 2. 环境与前置准备

### 2.1 软件
```
torch>=2.9         # 带 sm_90 / CUDA 12.x 构建
transformers>=4.44
accelerate>=0.30
pynvml>=11.5       # NVML 遥测（需支持 GetTotalEnergyConsumption / GetFieldValues）
matplotlib>=3.7
numpy>=1.24
# 可选：nvidia-dcgm / dcgmi（能量与降频原因流）；vllm（优化引擎轨道，见 §5.4）
```
确认 GPU 与架构：
```bash
python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# 期望 H200，capability (9, 0)
```

### 2.2 权限与持久模式
`-pl` / `-lgc` / `-lmcd` / `-pm` 需 root（脚本通过 `SUDO_PASS` + `sudo -S` 提供）；遥测与 `-q` 查询无需 root。
```bash
sudo nvidia-smi -pm 1     # 持久模式：驱动常驻，时钟/功率设置稳定，能量累加器保持单调
```
- 关闭 MIG（`nvidia-smi -q | grep -i "MIG Mode"` 应为 Disabled）。功率与时钟锁作用于整卡。
- 实验期间该卡不运行其它负载。
- 云实例先验证权限：`sudo nvidia-smi -lgc 1500,1500 && sudo nvidia-smi -rgc`；若报 `Not Supported / Insufficient Permissions`，需更换实例或申请权限。

### 2.3 查询本机约束（开测前必做，写入 meta.json）
```python
import pynvml
pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
print("name :", pynvml.nvmlDeviceGetName(h))
mn, mx = [x/1000 for x in pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)]
dft    = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h)/1000
print(f"-pl range [{mn:.0f},{mx:.0f}] W, default {dft:.0f} W")
mems = sorted(pynvml.nvmlDeviceGetSupportedMemoryClocks(h))
print("mem clocks (MHz):", mems)                       # 档数决定实验 4
sm   = sorted(set(pynvml.nvmlDeviceGetSupportedGraphicsClocks(h, mems[-1])))
print(f"SM clocks {sm[0]}..{sm[-1]} ({len(sm)} 档)")
for name, attr in [("GPU_MAX",  pynvml.NVML_TEMPERATURE_THRESHOLD_GPU_MAX),
                   ("MEM_MAX",  pynvml.NVML_TEMPERATURE_THRESHOLD_MEM_MAX),
                   ("SLOWDOWN", pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN),
                   ("SHUTDOWN", pynvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)]:
    try: print(name, pynvml.nvmlDeviceGetTemperatureThreshold(h, attr), "C")
    except pynvml.NVMLError as e: print(name, "n/a", e)
```
```bash
nvidia-smi -q -d SUPPORTED_CLOCKS | sed -n '1,40p'   # Memory 时钟档数
nvidia-smi --lock-memory-clocks-info                 # 显存锁定风格（预期 deferred）
nvidia-smi -q -d POWER                               # Min/Max/Default/Current Power Limit
```

### 2.4 基线记录
将以下写入 `meta.json`：GPU 名 / driver / CUDA / `-pl [min,max,default]` / supported SM & mem clocks / 显存锁定风格 / 温度阈值 / 怠速温度与功率 / 冷却方式（风冷 / 液冷·DLC / 云）/ 室温。

---

## 3. 测量原理

（解析模型详见 [POWER_THROUGHPUT_MODEL.zh.md](POWER_THROUGHPUT_MODEL.zh.md) 与 [pt_cap_gpu1/decode_model_theory.md](pt_cap_gpu1/decode_model_theory.md)；此处给出对测量方法的约束。）

- **建模对象是 `P ↔ T`**：功率为自变量，工作频率是其内部实现机制。
- **prefill 计算受限**（算术强度 I ≫ 脊点 I\*≈206）：吞吐随核心频率上升，功率遵循动态功耗律，得单段凸曲线
  `P(T) = P₀ + κ·T·(1 + ρT)²`。
- **decode 访存受限**（I ≪ I\*）：单 token 耗时 = 访存时间 `T_mem` + 计算时间 `T_comp`，
  `Throughput = Batch / (T_mem + T_comp)`。随功率上升呈三阶段——低功率核心被压、计算时间主导的上升段；中功率边际递减段；高功率访存平台 `T_max = Batch / T_mem`。
- **控制变量取功率/频率，而非 batch**：单纯扫 batch 会使功率很快顶到功率上限，得到退化的 L 形曲线；因此描 `P–T` 曲线须扫功率上限或锁定工作频率（§4），使功率与吞吐同时变化。
- **prefill 固定序列长度 S、只扫 batch**：固定每 token 成本后吞吐是 batch 的单调函数，`P(T)` 单值；若改扫 S，注意力 `O(S²)` 会使吞吐先升后降、同一吞吐对应两个功率点。故固定 S。

---

## 4. 功率调节方法

H200 提供两个运行时可用的旋钮——**功率上限 `-pl`** 与 **SM 频率锁 `-lgc`**；显存频率只能跨运行以延迟锁方式设定（§4.3）。

### 4.1 功率上限 `-pl`
设定后 GPU 在该功率下自选能维持的频率，功率与吞吐同时移动，运行中可改。
```bash
sudo nvidia-smi -i 0 -pl 400
nvidia-smi -i 0 --query-gpu=power.limit,power.draw,power.draw.instant --format=csv   # 验证
sudo nvidia-smi -i 0 -pl <default>   # 复位
```
- 仅当 workload 稳态功率 **超过** 所设上限时该上限才起作用；故 prefill 用重载（大 batch、长 S）确保任意上限都被触及。
- `-pl` 接受整数瓦（1 W 粒度），可用较细步长。
- `-pl` 存在硬下限（约 200 W）。访存类负载的实际功耗高于该下限——显存子系统本身即占约 220 W，因此 `-pl=200` 下 decode 实际仍抽约 250 W，无法更低。

### 4.2 SM 频率锁 `-lgc`
直接钉住核心频率以精确选择工作点、并覆盖 `-pl` 下限以下的低功率区，运行中可改。
```bash
sudo nvidia-smi -i 0 -lgc 1200,1200   # 锁 SM = 1200 MHz
sudo nvidia-smi -i 0 -rgc             # 复位
```
- decode 采用闭环：不断调整 `-lgc` 直到窗口内**持续功率**命中目标（[decode_powercap_sweep.py](code/decode_powercap_sweep.py)）。
- 低功率上限或低锁频下，SM 可被压至最低约 345 MHz，而 HBM 保持约 3201 MHz，对应 decode 上升段的低功率端。

### 4.3 显存频率（Hopper 延迟锁）
H200（Hopper）的 HBM3e 运行频率在**单次运行内固定**，运行时无法锁定：`nvidia-smi -lmc` 与 `nvmlDeviceSetMemoryLockedClocks` 在 Hopper 返回不支持。因此在一次 decode 运行内 HBM 恒为约 3201 MHz、β 恒定，decode 天花板 `T_max` 与功率上限无关；可在 `-pl` 扫描中用 CSV 的 `mem_clk_avg` 恒定加以确认。

若需改变显存频率，只能用**延迟锁**并重载驱动，且仅当 supported memory clocks 存在多个工作档时有意义：
```bash
sudo nvidia-smi -i 0 -lmcd <MHz>          # 延迟锁（下次 GPU 初始化生效）
sudo nvidia-smi --gpu-reset -i 0          # 或 rmmod/modprobe nvidia* 使其生效
#  ... 在此频率下运行一整轮 decode 实验 ...
sudo nvidia-smi -i 0 -rmcd                # 撤销延迟锁
sudo nvidia-smi --gpu-reset -i 0          # 再次重载恢复
```
`nvidia-smi --lock-memory-clocks-info` 报 `deferred` 即走此路径。据此，实验 4（§6）以"每个显存频率一整轮"的方式进行。

### 4.4 复位（每个脚本 `finally` 与每日收尾必做）
```bash
sudo nvidia-smi -i 0 -rgc                 # 解 SM 频率锁
sudo nvidia-smi -i 0 -pl <default>        # 恢复默认功率上限
# 若本轮用过延迟锁：
sudo nvidia-smi -i 0 -rmcd && sudo nvidia-smi --gpu-reset -i 0
```
离场前以 `nvidia-smi -q -d POWER,CLOCK` 确认无残留锁定。

### 旋钮选择

| 目的 | 旋钮 | 适用阶段 |
|---|---|---|
| 描整条 P–T（中高功率） | `-pl` 扫功率上限 | prefill（重载） |
| 低功率点 / decode 持续功率闭环 | `-lgc` 锁 SM 频率 | decode / prefill 低端 |
| 机理验证 `T∝f` 与功率律 | `-lgc` 全程锁频（`nvmlDeviceSetGpuLockedClocks`） | prefill + decode 对照 |
| 显存频率对天花板的影响 | `-lmcd` 延迟锁 + 重载（需 ≥2 档） | decode |

---

## 5. 功率区间与扫描网格

以下为建议起点，按 §2.3 的实测约束裁剪到 `[min,max]` 内（脚本会自动过滤越界点）。

### 5.1 功率网格
- **prefill 功率上限**（`-pl`，计算受限、上限被诚实执行）：
  ```
  CAP_GRID = [200,250,300,350,400,450,500,550,600,650,700]   # W（SXM；NVL 到 600）
  ```
- **decode 持续功率目标**（`-lgc` 闭环）：地板约 250 W，且因高 batch 下 SM 利用率低、decode 在远低于 700 W 处即进入平台，可用跨度较窄：
  ```
  TARGETS = [250,275,300,325,350,375,400,425,450,...]   # W，扫至吞吐不再随功率上升为止
  ```

### 5.2 频率网格
- **SM 频率**（`-lgc` / DVFS，实验 2/3）：从 supported 最低（可低至约 345 MHz）到 boost（约 1980 MHz），低端加密，取 12–16 点。
- **显存频率**（实验 4，仅 ≥2 档）：枚举 `nvmlDeviceGetSupportedMemoryClocks` 全部工作档，每档一整轮。

### 5.3 Batch / 上下文网格
先跑实验 5 定位显存墙与饱和点，再据此回填其它实验的固定 batch。
- **prefill（找计算顶）**：
  ```
  PREFILL_SEQ_LEN = 512            # 或 1024，长 S 更快喂满 Tensor Core
  PREFILL_BATCHES = [1,2,4,8,16,32,64,128,192,256]   # 扫至吞吐进入平台
  ```
- **decode（找带宽顶与显存墙）**：
  ```
  DECODE_CTX     = 256            # 主扫；另加一轮 ctx=2048/4096 加重 KV 项
  DECODE_BATCHES = [1,2,4,8,16,32,64,128,192,256,384,512,768,1024]   # 扫至 OOM（脚本自动标记显存墙）
  ```
  平台拐点约等于 `权重字节 / (ctx · 每 token KV 字节)`。

### 5.4 模型与引擎
选择覆盖不同规模与注意力结构（MHA / GQA）的模型；换模型仅需设置 `POWERCHAR_MODEL`（[config.py:25](code/config.py#L25)），分析时按模型的 `L/d/n_kv/h` 填入理论常数。

| 模型 | 参数 / fp16 权重 | 注意力 | 用途 |
|---|---|---|---|
| Llama-3.1-8B-Instruct | 8B / ~16 GB | GQA | 主力；高并发 decode，可承载大 batch |
| Qwen2.5-7B / 14B / 32B | 7–32B / 至 ~64 GB | GQA | 跨规模；32B 权重大，decode 保持权重流主导 |
| Llama-3.1-70B / Qwen2.5-72B | 70–72B（fp16 约 140–144 GB 接近显存上限，用 FP8 约 70–72 GB） | GQA | 逼近显存墙、充分利用带宽；FP8 权重减半，decode 上限更高 |
| Phi-3-mini-4k | 3.8B / ~7.6 GB | MHA | KV 较重的对照点，观察 MHA 与 GQA 对 `T_max`、显存墙位置的影响 |

- **GQA 与 MHA**：GQA 的 KV 头少、每 token KV 字节小，decode 更偏权重流主导，平台拐点与显存墙都更靠后。
- **两条引擎轨道**：
  - **eager HF `generate`**（现有脚本）——基线口径；受逐 token kernel 启动开销影响，达成带宽利用率（MBU）较低。
  - **vLLM + CUDA Graph（或 TensorRT-LLM）**——消除逐步启动开销、逼近带宽上限，MBU 显著更高，同功率下 `T_max` 更高。曲线的三阶段/平台结构在两轨道下一致。

---

## 6. 实验清单

通用启动：
```bash
SUDO_PASS=... CUDA_VISIBLE_DEVICES=0 POWERCHAR_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  PYTHONPATH=code python3 <脚本>
```
运行前 `nvidia-smi -pm 1`；运行后确认 `finally` 已复位（§4.4）。所有点按 §7 的能量法读功率并执行降频原因门控。

### 实验 1 — Prefill 功率上限扫描 → `P(T)`
- **目的**：拟合 `P = P₀ + κT(1+ρT)²`，验证凸性/超线性。
- **控制变量**：`-pl`。**脚本**：[pt_cap_sweep.py](code/pt_cap_sweep.py)（prefill 部分）。
- **网格**：`CAP_GRID`（§5.1）；重载 `PREFILL_BATCH=16, S=1024`。
- **产物**：`h200_pt_cap/pt_cap.csv`。
- **预期**：功率上限下降时吞吐明显下降；凸曲线，200→400 W 增益大，500→700 W 每 100 W 约 10%。

### 实验 2 — Decode 功率扫描 → 三阶段 `Throughput(P)`
- **目的**：刻画三阶段曲线，确定拐点功率与平台高度 `T_max`。
- **控制变量**：`-lgc` 闭环命中持续功率目标。**脚本**：[decode_powercap_sweep.py](code/decode_powercap_sweep.py)。
- **网格**：`TARGETS`（§5.1，约 250–450 W）；`DECODE_BATCH` 取实验 5 所得显存墙的约 80%；`CTX`（§5.3）。
- **产物**：`h200_pt_cap/decode_fixedbatch.csv`。
- **预期**：随功率上升有一段上升区（SM 提频），在高功率拐点（参考约 550 W，以实测为准）后进入访存平台；`mem_clk_avg` 全程约 3201，说明带宽与功率无关。

### 实验 3 — SM 频率 DVFS 扫描
- **目的**：验证 prefill `T∝f`（计算受限）、decode 对核心频率不敏感（访存受限），并测 prefill 的功率–吞吐指数。
- **控制变量**：全程锁 SM 频率（`nvmlDeviceSetGpuLockedClocks`）。**脚本**：[measure_dvfs.py](code/measure_dvfs.py)。
- **网格**：`FREQS`（§5.2，约 345…1980 MHz）；固定一组轻 prefill 与一组 decode workload。
- **产物**：`h200_dvfs/dvfs.csv` → `python3 code/analyze.py --step dvfs`。

### 实验 4 — 显存频率对 decode 天花板的影响
- **目的**：确定 decode 天花板 `T_max` 是否随显存频率变化。
- **前置判定**（§2.3）：
  - supported memory clocks **仅 1 个工作档** → 显存频率不可调；只需在 `-pl` 扫描中确认 `mem_clk_avg` 恒定、decode 平台与功率无关，记录固定的 `T_max`。
  - **≥2 个工作档** → 执行下述延迟锁扫描。
- **步骤**（在 [decode_clk_sweep.py](code/decode_clk_sweep.py) 基础上改为对显存频率扫描）：
  ```
  对每个 supported 显存频率 f_mem：
    sudo nvidia-smi -i 0 -lmcd f_mem          # 延迟锁
    sudo nvidia-smi --gpu-reset -i 0          # 使其生效
    sudo nvidia-smi -i 0 -lgc 1980,1980       # SM 锁 boost，确保计算不成为瓶颈
    运行 decode（固定接近满载 batch）→ 记录 (mem_clk_avg, T, P, tok/J)
    sudo nvidia-smi -i 0 -rmcd && sudo nvidia-smi --gpu-reset -i 0
  ```
- **产物**：`h200_memclk/decode_memclk.csv`。
- **判读**：吞吐随 `mem_clk` 上升（近 `T∝f_mem`）→ 显存频率是天花板高度的调节量，得一族不同高度的平台；吞吐不随 `mem_clk` 变或仅 1 档 → `T_max` 固定。

### 实验 5 — Batch 扫描（定位显存墙 / 带宽顶 / 计算顶）
- **目的**：为其它实验校准固定 batch。
- **控制变量**：batch（默认功率上限）。**脚本**：[measure.py](code/measure.py) `--phase both`，网格用 §5.3。
- **产物**：`h200_batch_sweep/{prefill,decode}.csv`（脚本自动捕获 OOM 并标记显存墙）。**先跑此实验**。

### 实验 6 — 热降频标定
- **目的**：标定持续满载下的降频行为，为其它实验设定冷却触发与窗口时长。
- **控制变量**：不锁频，持续满载 GEMM。**脚本**：[schedule_lab/thermal_throttle/](schedule_lab/thermal_throttle/)。
- **记录**：GPU die 温度与 **HBM 显存结温**（`nvmlDeviceGetTemperature` 显存传感器 / DCGM `DCGM_FI_DEV_MEMORY_TEMP`）；HBM 可能先于 die 触发降频。用 `nvmlDeviceGetCurrentClocksEventReasons` 记录降频原因位。温度阈值以 §2.3 现场读取为准（Hopper 报告为"离阈值余量"，逼近 0 触发降频）。
- **产物**：`h200_thermal/{throttle.csv,meta.json}`（峰值 die/HBM 温、阈值、最低频率、降频原因分布）。

### 实验 7 — 引擎对照
- **目的**：在 eager 与优化引擎（vLLM + CUDA Graph）下比较 decode 的 `T_max` 与达成带宽 `达成β = T·(权重字节 + B·C·kv)/B`，确认曲线结构一致、优化引擎更逼近带宽顶。
- **产物**：两引擎对照表（达成带宽% / 同功率 `T_max` 比）。

---

## 7. 测量规范

### 7.1 时序
每个扫描点执行 `WARMUP → SETTLE → MEASURE`（功率在 MEASURE 窗口内积分，[config.py:32](code/config.py#L32)）。适当加长 `SETTLE` / `MEASURE`，直到温度与 SM 频率都平稳（`dT/dt≈0`、时钟稳定在目标）再开窗——700 W 下热达稳态较慢，短窗易落在瞬态。液冷/DLC 卡入口温度恒定、点间可比；风冷卡会热漂移，需延长点间冷却。

### 7.2 采样与窗口对齐
NVML 后台线程以 50–100 Hz 采样（`SAMPLE_INTERVAL_S=0.01–0.02`）；吞吐只在功率积分的同一 `[t0,t1]` 窗口内计算（`stats_between`，[power_sampler.py:72](code/power_sampler.py#L72)），warmup 与尾部不进入统计。

### 7.3 功率读数（能量累加器法）
窗口平均功率由能量累加器求得，避免短窗口读数偏差：在窗口首尾各读一次 `nvmlDeviceGetTotalEnergyConsumption(h)`（单位 mJ，单调；持久模式下不清零），
```
P_win = (E_end − E_start) / (t1 − t0) / 1000   # W
```
`nvmlDeviceGetPowerUsage()`（即 `nvidia-smi power.draw`）在 Hopper 返回约 1 秒滑动平均，会平滑并滞后功率上限/频率的阶跃，不作为短窗口主口径；高频瞬时功率用 `nvmlDeviceGetFieldValues` 取 `NVML_FI_DEV_POWER_INSTANT`（约 25 ms），用于 `power_max`/`power_std` 与瞬态检查。DCGM 可用时以 `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION`(156) 与 `POWER_USAGE_INSTANT`(157) 一条流采集。

### 7.4 降频原因门控
每个采样读 `nvmlDeviceGetCurrentClocksEventReasons`（`nvidia-smi -q -d PERFORMANCE` 显示为 "Clocks Event Reasons"；DCGM 字段 112），对整个窗口取 OR 掩码与各位占比：
- **接受该点**：主导原因为 `SwPowerCap (0x04)`（功率上限扫描）或 `ApplicationsClocksSetting (0x02)`（锁频）——即受功率/频率限制。
- **丢弃并重测**：窗口内出现 `SwThermalSlowdown (0x20)` / `HwThermalSlowdown (0x40)` / `HwSlowdown (0x08)` / `HwPowerBrakeSlowdown (0x80)`——反映的是热/保护状态，而非目标工作点。
- 每点存 OR 掩码与各原因样本占比，供事后审计。

### 7.5 遥测字段与冷却
- 记录：`power_avg_w`（能量法）/ `power_instant_max_w` / `power_instant_std_w` / `energy_mj` / `sm_clk_avg` / `mem_clk_avg` / `gpu_temp` / `mem_temp`（HBM）/ `util_gpu` / `util_mem` / `throttle_mask` / `enforced_pl_w`（确认上限已生效）。
- 点间冷却，阈值按实验 6 标定，或直接按降频原因位触发（出现热位即冷却）。温度阈值由 `nvmlDeviceGetTemperatureThreshold` 在运行时读取，不使用固定常数。

### 7.6 持续功率
能量累加器法天然把逐步之间的 GPU 空隙计入，给出真实窗口平均功率；优化引擎轨道（CUDA Graph）下空隙减小，瞬时与平均更接近。

---

## 8. 数据记录与产物

### 8.1 CSV 列
`phase, batch, seq_len/ctx_len, throughput_tok_s, power_avg_w, power_instant_max_w, power_instant_std_w, energy_mj, util_gpu_avg, util_mem_avg, sm_clk_avg, mem_clk_avg, gpu_temp_avg, mem_temp_avg, throttle_mask, enforced_pl_w, tok_per_joule, engine, quant, n_samples, window_s`。

### 8.2 目录布局（每实验自包含）
```
h200_pt_cap/       实验1/2   pt_cap.csv, decode_fixedbatch.csv, *.png
h200_dvfs/         实验3     dvfs.csv, fig
h200_memclk/       实验4     decode_memclk.csv, fig
h200_batch_sweep/  实验5     prefill.csv, decode.csv
h200_thermal/      实验6     throttle.csv, meta.json, fig
h200_meta.json     全局元数据（§2.4 + 达成带宽%、引擎、量化）
```

---

## 9. 安全与复位检查清单
- [ ] 开测 `nvidia-smi -pm 1`。
- [ ] 每脚本 `finally` 复位：`-rgc`、`-pl <default>`；用过延迟锁则 `-rmcd` + `--gpu-reset`。
- [ ] 不设置超过 `-pl max` 的功率、不逼近温度阈值。
- [ ] 每点记录降频原因，剔除热污染点（应由功率/频率主导）。
- [ ] 功率读数用能量累加器法。
- [ ] 云实例先验证 `-pl/-lgc` 权限（`-lmcd` 需能重载驱动 / reset）。
- [ ] 离场以 `nvidia-smi -q -d POWER,CLOCK` 确认无残留锁定。

---

## 10. 执行流程（建议顺序与时长）
1. **环境自检**（10–15 min）：§2.1 软件、§2.2 权限、§2.3 查约束写入 `h200_meta.json`、`-pm 1`。
2. **配置遥测**：按 §7.3/7.4/7.5 使采样器采用能量法、降频原因门控与 HBM 温度。
3. **实验 6 热标定**（15 min）：确定冷却触发与窗口时长。
4. **实验 5 batch 扫描**（每模型 20–30 min）：定位显存墙与饱和点，回填固定 batch。
5. **实验 1 prefill 功率上限扫描**（约 15 min/模型）。
6. **实验 2 decode 三阶段功率扫描**（约 20 min/模型）。
7. **实验 3 SM 频率 DVFS**（约 15 min）。
8. **实验 4 显存频率扫描**（前提 §2.3 显存 ≥2 档；含多次驱动重载，预留 30–45 min）。
9. **实验 7 引擎对照**（vLLM 环境就绪时）。
10. **复位与打包**：§9 清单，打包各子目录与 `h200_meta.json`。

> 建议先用 Llama-3.1-8B（GQA）与 Phi-3-mini（MHA）跑通全流程，再扩展至 14B / 32B / 72B(FP8)。

---

## 附录 A — 命令速查
```bash
# 查询约束（开测前）
nvidia-smi -q -d POWER,CLOCK,SUPPORTED_CLOCKS,TEMPERATURE,PERFORMANCE
nvidia-smi --lock-memory-clocks-info

# 持久模式
sudo nvidia-smi -pm 1

# 功率上限
sudo nvidia-smi -i 0 -pl 400 ;  sudo nvidia-smi -i 0 -pl <default>

# SM 频率锁
sudo nvidia-smi -i 0 -lgc 1500,1500 ;  sudo nvidia-smi -i 0 -rgc

# 显存频率（Hopper：延迟锁 + 重载）
sudo nvidia-smi -i 0 -lmcd <MHz> && sudo nvidia-smi --gpu-reset -i 0
sudo nvidia-smi -i 0 -rmcd       && sudo nvidia-smi --gpu-reset -i 0

# 全复位（离场前）
sudo nvidia-smi -i 0 -rgc ; sudo nvidia-smi -i 0 -pl <default>

# 精确窗口功率（能量法，§7.3）
python3 -c "import pynvml as n;n.nvmlInit();h=n.nvmlDeviceGetHandleByIndex(0);print(n.nvmlDeviceGetTotalEnergyConsumption(h),'mJ')"

# 运行一个实验
SUDO_PASS='***' CUDA_VISIBLE_DEVICES=0 \
  POWERCHAR_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  PYTHONPATH=code python3 code/pt_cap_sweep.py
```

## 附录 B — 参数配置汇总

| 参数 | 取值 |
|---|---|
| prefill 功率上限 `CAP_GRID` | 200…700 W（50 W 步长，SXM；NVL 至 600） |
| decode 持续功率 `TARGETS` | 250…450 W（至平台） |
| SM 频率 `FREQS` | supported 最低（约 345）…1980 MHz，低端加密，12–16 点 |
| 显存频率 | 枚举 supported 工作档（仅 ≥2 档时扫描） |
| `PREFILL_SEQ_LEN` | 512（或 1024） |
| `PREFILL_BATCHES` | 1…256（至计算平台） |
| `DECODE_CTX` | 256（另加一轮 2048/4096） |
| `DECODE_BATCHES` | 1…1024（至显存墙） |
| 静态底 `P₀`（拟合参考） | 约 150–250 W |
| 功率读数 | 能量累加器 `(E1−E0)/Δt` |
| 模型 | Llama-3.1-8B / Qwen2.5-7B~32B（GQA）/ Phi-3-mini（MHA）/ 70B·72B(FP8) |
| 引擎 | eager HF 与 vLLM+CUDA Graph 两轨 |

## 附录 C — 曲线预期与合理性检查

| 曲线 | 预期形状 | 异常排查 |
|---|---|---|
| prefill P–T（实验 1） | 凸、超线性；上限下降吞吐明显下降 | 吞吐几乎不变 → 上限未触及（增大 batch/S） |
| decode Throughput–P（实验 2） | 上升段 → 高功率拐点 → 平台；`mem_clk` 恒 3201 | 全程平 → batch 过小或已在平台；`mem_clk` 变化 → 检查是否触发其它状态 |
| prefill T–f（实验 3） | `T∝f^≈0.9` | 指数偏低 → 非计算受限（S 太短或 batch 过大） |
| decode T–f_sm（实验 3） | 高频段趋平（对核心频率不敏感） | 全程随 f_sm 上升 → 仍在低功率上升段（增大 batch/ctx） |
| decode T–f_mem（实验 4） | ≥2 档且随 f_mem 上升→可调；仅 1 档或不变→固定 | 见实验 4 判读 |
| 能效 tok/J | prefill 单峰；decode 因固定显存功率底，低利用率时偏低 | 与解析模型能效段对照 |

---

*本手册所列 H200 硬件数值为参考；实测前以本机 §2.3 查询为准并记入 `meta.json`。*
