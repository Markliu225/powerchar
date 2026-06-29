# Schedule Lab — 手动编排 GPU 工作负载的可视化界面

给定**总工作量**,**手动编排**每次执行(分几块、每块多少、块后停多久或冷到几度),点运行 → 在 GPU 上跑、20Hz 记录温度/频率/功率 → 结果**直接写到本地文件**:

- `schedule_lab/result.png` — 图(累计完成量 / 温度 / 频率随时间 + JCT)
- `schedule_lab/result.csv` — 原始 20Hz 遥测

网页只是从磁盘加载 `result.png` 显示;这两个文件你也能直接在编辑器里打开(每次运行覆盖)。

## 启动
```bash
CUDA_VISIBLE_DEVICES=1 python3 schedule_lab/app.py
```
然后浏览器打开 **http://localhost:8000**(VS Code 会自动转发端口;远程 SSH 用 `ssh -L 8000:localhost:8000`)。无需 sudo、不锁频——观察系统自己的热行为。

## 怎么用
- **总工作量 total**:要做多少个 GEMM(每个 = fp16 8192² 矩阵乘 ≈ 满频 ~12ms)。
- **编排 schedule**:JSON 步骤列表,**循环执行直到做满 total**。每步:
  - `gemms`:这步连续跑多少个(满频);然后二选一停顿——
  - `pause_s`:固定空闲 N 秒,或
  - `cool_to`:空闲直到温度 ≤ 该值(°C)。

### 预设(点按钮自动填)
| 预设 | schedule | 含义 |
|---|---|---|
| 一次性灌满 | `[{gemms:2000,pause_s:0}]` | 全程连续,撞 83°C 降频 |
| 涓流 | `[{gemms:50,pause_s:1.0}]` | 小块+小停,控温不降频 |
| 冷却闸 | `[{gemms:400,cool_to:72}]` | 跑一块,冷到 72°C 再跑 |
| 自定义阶梯 | `[{gemms:800,pause_s:0},{gemms:200,pause_s:4},{gemms:100,pause_s:8}]` | 先猛后缓 |

自己改 JSON 就能编排任意模式。例如"先满灌 1000,再每 50 个停 2s":
```json
[{"gemms":1000,"pause_s":0},{"gemms":50,"pause_s":2}]
```

## 输出图(三面板,共享时间轴)
- **累计 GEMMs vs 时间**:斜率=速度;蓝底=停顿;终点=JCT。
- **温度 vs 时间**:含 83°C 降频线。
- **频率(绿)+ 功率(灰)vs 时间**。
顶部数字:JCT、停顿占比、峰值温度、运行时最低频率。

## 说明 / 边界
- 安全上限:total ≤ 20000,单次 pause ≤ 120s,cool_to 等待 ≤ 180s。
- 运行期间网页同步等待(几十秒~几分钟);同一时刻只允许一个运行。
- 改 `app.py` 顶部:`GEMM_N`(矩阵大小/单 GEMM 热量)、`PORT`、`INNER`(日志粒度)。
