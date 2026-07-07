# Decode 数据正确性 —— 测量方法学 v3 与证据链

v1 数据（首轮）事后审查发现四类测量伪影。本文档记录每一项的**症状 → 隔离实验 → 修复 → 验证**，
v3（当前 `data/`）是全部修复后的最终数据；v1 保留在 `data_v1/` 供对照（也在 git 历史中）。

## 1. KV cache 构建方式伪影（最大的一项，~10–70% 偏差）

**症状**：v2 重跑时 chat 平台 820 tok/s，v1 只有 739（同一 workload、同一卡型）。

**隔离**（`ab_chat_probe.py`，GPU2，同一时钟 ~1530 MHz，chat 点 C=256 B=64）：
KV cache 用多大的 chunk 逐段 seed 出来，决定了后续 decode 的稳态吞吐：

| seed chunk | 吞吐 (tok/s) | GPU util | 功率 |
|--:|--:|--:|--:|
| 32 | 225 / 233 | **37%** | 113 W |
| 64 | 558 / 608 | 81% | 187 W |
| 128 | 838 / 829 | 99% | 242 W |
| 256 (=C) | 830 / 823 | 96% | 233 W |

机理：transformers 的 DynamicCache **每个 decode 步都对整层 KV 做 torch.cat**（全量重分配+拷贝）。
小 chunk seed 在 PyTorch caching allocator 里留下碎片化的块历史，此后每步的大块 cat 都要经历
分配器重组/新段分配，**CPU 侧停顿让 GPU 饿死**（util 37%）——测的不再是 GPU 物理，是分配器。

**修复**：`C.DECODE_SEED_CHUNK=256`（整段 prefill 建 cache，正是真实 serving 的方式）。
**判据**：chunk 128 与 256 结果一致（GPU 受限），且 util ≈100%。v1 全量数据被此伪影拖低（轻负载最重）。

## 2. 分配器块复用抽签（rag 形状的 ±9% 双模态 + 系统性低估）

**症状**：v2 中 rag（B=32,C=1024）几乎每个 cap spread 7–31%，同卡同会话的 code/longform 却 ≤1.4%。

**隔离**（GPU1，rag 点 @250W，8 次重复）：

| 分配器 | 8 次吞吐 (tok/s) | 离散 |
|---|---|--:|
| 默认 | 237 227 225 241 267 265 243 229 | ±9%，双模 |
| `expandable_segments:True` | 281 280 272 279 277 280 279 279 | **±1.6%，且快 ~15%** |

**修复**：`run_portfolio.py` 顶部 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（先于 torch import）。
v3 全量在此设置下采集；rag 平台从 v2 的 228 恢复到 ~275。

## 3. 慢点的步数量化噪声

**症状**：summarize（6 tok/s, B=4）在固定 2.5 s 窗口内只有 ~4 个 decode 步 → 量化噪声 ±25%。

**修复**：步数目标窗口（`target_steps=32`，且 ≥2.5 s、上限 45 s）；每个 cap 重复 2 次，
spread>5% 再补第 3 次取中位（CSV 记录 `spread_pct`/`n_runs`）。v3 中 summarize 单点窗口自动拉长到 ~27 s。

## 4. 窗口内 KV 漂移（小上下文）

**症状**：测量窗口内 cache 从 C 长到 C+steps，chat（C=256）访存量漂移 ~6–12%。

**修复**：不硬止（那是 decode 的本性），改为**记录并入模**——CSV 增加 `ctx_eff = C + steps/2`
（窗口内流量加权的有效上下文），下游 T_max 预测用 `ctx_eff` 而非标称 C；步数目标窗口同时把漂移
上界钉死在 32 步。

## 交叉验证

- **跨卡一致性**（chunk=256 参考点，chat）：GPU0 841 / GPU1 830 / GPU2 827 / GPU3 825 —— 相差 <2%，
  多卡并行采集不引入系统差；v1 的 739 纯属伪影 1，不是硅差。
- **classify 的非单调（保留，非错误）**：cap ≥180 W 后该负载实际功率只有 ~150 W（cap 不再咬合），
  DVFS 调速器对 B=8、util~70% 的欠饱和突发负载自行浮动降频，吞吐在平台附近散动（GPU2/GPU3 均复现）。
  功率空间里这些点聚在同一 x 附近，属于"cap 不再是有效旋钮"的区域，如实呈现。

## 结论

v3 = chunk-256 seeding + expandable_segments + 步数窗口 + 重复中位 + ctx_eff 记录。
每一项修复都有独立的隔离实验支撑；v1→v3 的差异全部可归因、可复现。
