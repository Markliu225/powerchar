# `fig_model_vs_measured` —— 实验数据 + 理论模型说明

本文件夹自包含 `fig_model_vs_measured.png` 这张图背后的**全部实验数据、理论模型与可复现脚本**。

- 对象:`microsoft/Phi-3-mini-4k-instruct`(fp16)
- 硬件:`NVIDIA Tesla V100-DGXS-32GB`(Volta sm_70,32GB,功耗墙 250W,显存时钟定 877MHz,核心最高 1530MHz)
- 主题:LLM 推理两阶段(**prefill** 提示处理 / **decode** 自回归生成)的 **GPU 功率 P ↔ token 吞吐 T** 关系,实测对照架构级解析模型。

```
fig_model_vs_measured_bundle/
├── README.md                     # 本文件
├── fig_model_vs_measured.png     # 图(4 面板)
├── THEORY.zh.md / THEORY.en.md   # 架构级解析模型(详细推导)
├── code/plot.py                  # 独立复现脚本(只读 data/,重画该图)
└── data/
    ├── prefill_freq_sweep.csv    # prefill 实验:频率扫描(锁频)
    ├── decode_batch_sweep.csv    # decode 实验:batch 扫描(定满频)
    └── model_info.json           # 模型架构 + 该卡硬件常数
```

---

## 1. 图在画什么(4 个面板)

| | 左 | 右 |
|---|---|---|
| **上** | PREFILL:功率 vs 吞吐(频率扫描) | DECODE:功率 vs 吞吐(batch 扫描) |
| **下** | PREFILL:能效(tok/J) vs 功率 | DECODE:能效(tok/J) vs 功率 |

每个面板:**实测点 + 解析模型曲线**(prefill 用 V²f 物理形式,decode 用仿射形式)。空心点为排除点(见 §5)。

---

## 2. 两个实验怎么做的(关键:两阶段用不同的"旋钮")

两阶段的 P–T 关系必须用**各自能让 P、T 同时变化的控制变量**来测,否则曲线退化:

- **PREFILL = 计算受限** → 用**频率旋钮**(锁 SM 时钟)。
  把 SM 时钟锁在 510→1530 MHz 一系列值(`sudo nvidia-smi -lgc`),固定一个轻量 prefill 负载(batch=4, seq_len=256),在每个时钟下测吞吐与功率。时钟变 → T、P 同时变 → 得到 P–T 曲线。
  - **冷启动 + 短窗口 + 每点重复 3 次取均值**:这张卡在持续重载下会热降频(~82°C 触发),为拿到"未降频"的干净点,每点先冷却再短测;`1260 MHz` 那点是后来单独冷态补测的(`act_clk` 实际守住=req 即为干净点)。
  - **高频热降频**:`1410/1530 MHz` 请求下实际时钟掉到 802/942 MHz(`act_clk < req_clk`),功率/吞吐方差大 → **判为撞墙降频,排除出拟合**(图中空心)。

- **DECODE = 访存受限** → 用 **batch 旋钮**(定满频 1530MHz,扫 batch)。
  decode 吞吐几乎不随核心频率变(`T∝f^0.26`),其 P–T 关系来自**并发**:固定 ctx=256、时钟 1530MHz,batch 从 1 扫到 64;batch 变 → T、P 同时变 → 得到 P–T 直线。
  - `batch=64` 这点温度升到 82°C 边缘、时钟掉到 1415MHz、功率方差大 → **判为热边缘,排除出拟合**(图中空心)。

**测量协议(每点)**:warmup → settle → 在与功率平均**完全同一个**时间窗口内计吞吐;NVML 以 50Hz 采功率/时钟/温度,只对窗口内样本取平均。详见 [code/plot.py](code/plot.py) 复现路径与上层仓库的 `power_sampler.py` / `measure.py`。

---

## 3. 数据文件与列含义

### `data/prefill_freq_sweep.csv`(频率扫描;含 prefill 与同步测的 decode 行)
| 列 | 含义 |
|---|---|
| `workload` | prefill / decode |
| `req_clk_mhz` | 请求锁定的 SM 时钟 |
| `act_clk_mhz` | **实际达到的** SM 时钟(< req 即被热降频) |
| `throughput_tok_s` / `throughput_std` | 吞吐均值 / 3 次标准差 |
| `power_avg_w` / `power_std_w` | 功率均值 / 标准差 |
| `temp_avg`, `n_rep` | 平均温度、重复次数 |

> 画 prefill 用其中 `workload=prefill` 且 `act≈req`(未降频)的点:**510–1260 MHz,共 6 点**(`T 3362→7334 tok/s`,`P 103.6→235.6 W`,误差棒极小)。

### `data/decode_batch_sweep.csv`(batch 扫描,定满频 1530MHz)
列为完整遥测:`batch, ctx_len, throughput_tok_s, power_avg_w, sm_clk_avg, power_std_w, temp_avg, …`。用 `batch≤32` 的 6 点拟合(b=64 排除)。

### `data/model_info.json`
模型架构(L=32, d=3072, n_q=n_kv=32(MHA), head_dim=96, d_ff=8192, 总参 3.821B)+ 该卡硬件常数(峰值算力 ~87 TFLOP/s、带宽 ~782 GB/s、功耗墙 250W、`kv_bytes_per_token=393216`)。

---

## 4. 理论模型与拟合结果

完整推导见 [THEORY.zh.md](THEORY.zh.md)。核心:动态功率 `P_dyn ∝ V²·f`,吞吐取决于瓶颈资源。

### PREFILL(计算受限,频率旋钮)
吞吐随时钟 `T ∝ f`;功率走 V²f 律。在 `V=V₀+γf` 下展开为**完全平方形式**(线性+二次+立方三项捆绑,`b²=4ac`):

$$\boxed{\,P_{\text{prefill}}(T) = P_0 + \kappa\,T\,(1+\rho T)^2\,}$$

**本数据拟合**(在未降频的 6 点上,以"贴合能效"为目标):
- `P₀ ≈ 92 W`(带载活跃静态底),`κ ≈ 3.3e-6`、`ρ ≈ 1.0e-2`(脚本会打印);
- **功率–吞吐 R² ≈ 0.99**,**能效–功率 R² ≈ 0.83**。
- 物理含义:中低频段电压触底(V≈V₀)→ 线性+二次主导;立方(纯 `V∝f`)只是远端尾。所以"prefill 功率随吞吐近似立方"在本卡测到的频段**并不严格成立**,是"线性+二次为主、立方为尾"。

### DECODE(访存受限,batch 旋钮)
吞吐随带宽/并发上升,功率随并发线性上升:

$$\boxed{\,P_{\text{decode}}(T) = a + s\cdot T\,}$$

**本数据拟合**(b≤32):`P = 110.9 + 0.190·T` W,**R² ≈ 0.996**(几乎完美线性);能效随功率上升、趋向渐近 `1/s ≈ 5.3 tok/J`。

### 关键对比
同等/相近功率下,**prefill 吞吐 ~一个数量级高于 decode**,能效约 10×——根因:prefill 复用权重(算术强度 ≫ roofline 脊点 111 FLOP/byte),decode 每步重读全部 7.64GB 权重只产 batch 个 token(强度 ≈ batch ≪ 脊点)。

---

## 5. 排除点(诚实标注)
- prefill `req 1410/1530`:热降频(`act` 掉到 802/942),非该时钟的真实点 → 排除;
- decode `b=64`:热边缘(82°C、时钟掉、功率方差大)→ 排除。
两者仅参与"撞墙/热边缘"的说明,不参与定律拟合。

---

## 6. 复现
```bash
cd fig_model_vs_measured_bundle
python3 code/plot.py        # 需 numpy + matplotlib;重画 fig_model_vs_measured.png
```
脚本会打印两条拟合(prefill V²f 的 P₀/κ/ρ 与 R²,decode 仿射的 a/s 与 R²),并重新生成图。

## 7. 重要 caveat
- 这是**未降频(冷启短突发)**口径的 P–T;该 V100 在**持续**重载下会热降频到 ~82°C / 更低时钟,持续功率上不到 250W 墙(热限,非功率限)。
- 硬件峰值(Φ、β)随测量时卡温漂移;`model_info.json` 用的是冷态专测的卡峰值(~87 TFLOP/s、~782 GB/s)。
