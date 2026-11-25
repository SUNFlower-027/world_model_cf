# 展示素材图库（图片与视频说明）

本文把项目目前已经产出的**展示类图片与视频**集中到一处，逐项说明：文件位置、规格、画面内容、数据来源与解读边界。

> 素材归属：以下文件全部是**本机运行产物**，位于 `outputs/` 下。`outputs/` 被 `.gitignore` 排除，**不随仓库分发**；克隆仓库后需按第 7 节命令自行生成。文中路径均相对项目根目录 `world_model_cf/`。

---

## 1. 素材总览

| # | 文件 | 类型 | 规格 | 内容 | 用途 |
| --- | --- | --- | --- | --- | --- |
| 1 | `outputs/carla_showcase/videos/drive_showcase.mp4` | 视频 | 1024×580，10 fps，91.6 s，13.5 MB | 双视角闭环演示（左相机 / 右 BEV）+ 动作条，两段 episode | 答辩、课堂主视频 |
| 2 | `outputs/carla_showcase/videos/episode_001.mp4` | 视频 | 512×512，10 fps，20.2 s | 单视角前视相机，一次完整闭环 | 备选片段 |
| 3 | `outputs/carla_showcase/videos/episode_002.mp4` | 视频 | 512×512，10 fps，12.7 s | 同上 | 备选片段 |
| 4 | `outputs/carla_showcase/videos/episode_003.mp4` | 视频 | 512×512，10 fps，15.0 s | 同上 | 备选片段 |
| 5 | `outputs/carla_showcase/videos/episode_004.mp4` | 视频 | 512×512，10 fps，45.8 s | 同上，最长的一段 | 备选片段 |
| 6 | `outputs/carla_demo_video/videos/episode_001.mp4` | 视频 | 512×512，10 fps，16.9 s | 最早跑通链路时的相机录像 | 只作过程记录 |
| 7 | `outputs/carla_demo_video/videos/episode_002.mp4` | 视频 | 512×512，10 fps，38.4 s | 同上 | 只作过程记录 |
| 8 | `outputs/carla_cf/divergence.png` | 图片 | 1280×768 | 60 步反事实 JS 分歧曲线 + 噪声底 + 阈值 | 报告核心图 |
| 9 | `outputs/carla_cf/videos/cf_grid.mp4` | 视频 | 768×848，5 fps，12.0 s | 8 格对比：真实 / 事实想象 / 5 个干预 / 噪声底 | 答辩动态展示 |
| 10 | `outputs/carla_cf/videos/cf_contact_sheet.png` | 图片 | 768×834 | `cf_grid.mp4` 的最后一帧（静态九宫格） | 报告插图 |
| 11 | `outputs/carla_cf_branches/videos/cf_contact_sheet.png` | 图片 | 768×834 | 另一组 60 步反事实的九宫格 | 与 #10 二选一 |
| 12 | `outputs/carla_cf_branches/divergence.png` | 图片 | 1280×768 | 另一组 60 步分歧曲线（噪声底更低） | 复现性对照 |
| 13 | `outputs/carla_cf_smoke/videos/cf_grid.mp4` | 视频 | 768×848，5 fps，1.6 s | 8 步冒烟版宫格视频 | 流程自检 |
| 14 | `outputs/carla_cf_smoke/divergence.png` | 图片 | 1280×768 | 8 步冒烟分歧曲线 | 流程自检 |
| 15 | `outputs/cf/smoke/divergence.png` | 图片 | 1760×1120 | 合成小球环境的世界模型反事实四联图 | 机制自检（非 CARLA） |

---

## 2. 闭环演示视频

### 2.1 `drive_showcase.mp4`（主演示视频）

- 路径：`outputs/carla_showcase/videos/drive_showcase.mp4`
- 规格：1024×580，10 fps，916 帧，91.6 秒
- 生成脚本：`wmcf/carla/run_demo.py`（`--save-video`）

画面结构（`run_demo.py` L193-230）：

| 区域 | 内容 | 说明 |
| --- | --- | --- |
| 左半屏 | `camera (what driver sees)` | CARLA 前视相机真实渲染，**只给人看**，不进策略 |
| 右半屏 | `birdeye_wpt (what world model sees)` | 策略真正使用的 128×128 鸟瞰图 |
| 底部动作条 | 加速度 + 转向文字，外加 15 维 one-hot 小格 | 来自 `np.argmax(action)` 的解码显示 |
| 最底部 | `camera + BEV -> encoder -> RSSM latent -> policy -> discrete action` | 数据流说明文字 |
| 卡片 | `Episode 2` | 两段 episode 之间的分隔帧（5 帧黑底白字） |

两段 episode 的实际数据（与 `outputs/carla_showcase/metrics.jsonl` 第 11–12 条一致）：

| 段 | 步数 | 总回报 | 每步回报 |
| --- | --- | --- | --- |
| Episode 1 | 461 | 1061.4 | 2.302 |
| Episode 2 | 452 | 1037.6 | 2.296 |

解读边界：

- 两段是**同一策略的独立采样**，起点车道、随机路线、背景交通都不同；总回报差异主要来自 episode 长度，不能当作策略优劣或方法对比。
- 录制时把 `camera.fov` 从 120 改为 90 让画面更接近人眼；该参数**只影响录像**，策略输入仍是 `birdeye_wpt`，行为不受影响。
- 这一版来自 50 辆背景车的配置；曾把 `num_vehicles` 降到 10，episode 提前终止、回报跌到 281–409，因此保留 50 辆。

### 2.2 `episode_001.mp4` ~ `episode_004.mp4`（单视角分段录像）

- 目录：`outputs/carla_showcase/videos/`
- 规格：512×512，10 fps，仅前视相机，片头全黑帧已剔除
- 生成方式：同一次 `run_demo.py` 运行，每个 episode 各写一段（`run_demo.py` L189）

| 文件 | 帧数 | 时长 | 对应步数 | 对应总回报 |
| --- | ---: | ---: | ---: | ---: |
| `episode_001.mp4` | 202 | 20.2 s | 203 | 408.8 |
| `episode_002.mp4` | 127 | 12.7 s | 128 | 281.4 |
| `episode_003.mp4` | 150 | 15.0 s | 151 | 303.2 |
| `episode_004.mp4` | 458 | 45.8 s | 459 | 1033.8 |

> **重要**：`metrics.jsonl` 是**追加写入**（`JSONLOutput` 以 `"a"` 模式打开），因此该文件累积了多次运行，共 16 条记录。上表对应**最后 4 条**；`drive_showcase.mp4` 用的是第 11–12 条（461 / 452 步），两者**不是同一批**，不要混用。
> 这 4 段中只有 `episode_004` 的回报（1033.8）接近其他到达终点 episode 的量级，前 3 段在 128–203 步提前终止，属于正常采样波动，不宜单独作为成功率证据。

### 2.3 `carla_demo_video`（早期链路验证）

- 路径：`outputs/carla_demo_video/videos/episode_001.mp4`（16.9 s）、`episode_002.mp4`（38.4 s）
- 规格：512×512，10 fps，仅相机视角
- 该目录的 `metrics.jsonl` 为空（当次未写 KPI），只用于确认「CARLA → CarDreamer → 录像」链路能跑通，**不建议放进报告主图**。

---

## 3. 反事实干预素材（`outputs/carla_cf`，60 步主结果）

### 3.1 `divergence.png` — 潜空间分歧曲线

![CARLA 四车道反事实分歧曲线（60 步）](../outputs/carla_cf/divergence.png)

- 规格：1280×768（8×4.8 英寸，160 dpi）
- 坐标轴（`counterfactual_demo.py` L383-404）：横轴「干预后想象步数」1–60，纵轴「潜空间 JS 分歧（nats）」
- 曲线：5 条 `do(...)` 分支 + 灰色虚线「噪声底（同动作重采样）」+ 点线「显著性阈值」；实线阴影为跨 4 个起点的 ±1 标准差

实测数值（`divergence.json`）：

| 分支 | 终点分歧 | 峰值 | 首次超阈值步 | 对数增长率（1/步） |
| --- | ---: | ---: | ---: | ---: |
| do(刹车) | 0.439 | 0.442 | 42 | 0.0158 |
| do(保持) | 0.415 | 0.421 | 未越过 | 0.0223 |
| do(左转) | 0.403 | 0.468 | 16 | 0.0085 |
| do(右转) | 0.396 | 0.488 | 11 | 0.0077 |
| do(加速) | 0.378 | 0.379 | 未越过 | 0.0185 |
| 噪声底 | 0.164 | 0.194 | — | — |

- 干预起点：4 个 anchor `[47, 168, 289, 411]`；显著性阈值 0.4331，定义为「逐步 (噪声均值 + 2×标准差) 的最大值」。
- 结论方向：转向最早越过阈值（11–16 步），刹车较晚（42 步），保持 / 加速在 60 步内未越过。

### 3.2 `cf_grid.mp4` / `cf_contact_sheet.png` — 八格对比

![八格反事实对比（最后一帧）](../outputs/carla_cf/videos/cf_contact_sheet.png)

- 视频：768×848，5 fps，60 帧，12.0 秒
- 静态图：`cf_contact_sheet.png` 就是视频**最后一帧**
- 布局（`counterfactual_demo.py` L468-523）：3 列 × 3 行，共 8 格，每格 256×256，格上方有标签，画布左下角显示 `step t/60`

格子顺序与含义：

| 位置 | 标签 | 内容 |
| --- | --- | --- |
| 第 1 格 | `real observation` | CARLA 真实 `birdeye_wpt`（该时刻真实观测） |
| 第 2 格 | `imagination (factual)` | 按真实动作序列开环想象的 BEV |
| 第 3 格 | `do(brake)` | 每步强制 action_id=2（刹车）的想象 |
| 第 4 格 | `do(hold)` | 强制 action_id=7（保持）的想象 |
| 第 5 格 | `do(accelerate)` | 强制 action_id=12（加速）的想象 |
| 第 6 格 | `do(left)` | 强制 action_id=5（左转）的想象 |
| 第 7 格 | `do(right)` | 强制 action_id=9（右转）的想象 |
| 第 8 格 | `noise floor` | 同动作、重采样潜变量的噪声对照 |

> 注意：这一格视频是**世界模型的想象画面**（解码器输出），不是真实车辆轨迹。它能说明「同一后验起点、换动作后想象如何分叉」，不能说明物理碰撞或轨迹误差。

### 3.3 KPI 表

`outputs/carla_cf/kpi_table.md` / `kpi.csv`：3 个 episode、seed 0–2，平均步数 450.0 ± 16.6，平均回报 986.8 ± 29.8，碰撞率 0.67，出车道率 0.00，到达率 0.33。完整字段说明见 [autonomous_driving.md](autonomous_driving.md) 5.7.6 节。

---

## 4. 分支落盘版（`outputs/carla_cf_branches`）

结构与第 3 节相同（曲线图 / 八格视频 / 接触图 / KPI），差异是**额外保存了逐帧数组**：

- `branches.npz`（38.3 MB，`np.savez_compressed`）
  - `camera`：`uint8 [60, 512, 512, 3]`
  - `real`、`factual`、`do_brake`、`do_hold`、`do_accelerate`、`do_left`、`do_right`、`noise_floor`：`float [60, 128, 128, 3]`，值域 [0, 1]

用途：后续「方向一致性检验」（比较 `do_left` 与 `do_right` 解码 BEV 的质心偏移是否反向）与「动作置换负对照」直接读这个文件即可，无需重跑推理。

这一组与第 3 节的差别（同样 60 步，但 CARLA / Traffic Manager 随机源不同）：

| 项 | `carla_cf` | `carla_cf_branches` |
| --- | ---: | ---: |
| anchor | 47 / 168 / 289 / 411 | 44 / 158 / 273 / 388 |
| 阈值 | 0.4331 | 0.4212 |
| 噪声底终点 | 0.164 | 0.081 |
| 左转终点 / 超阈值步 | 0.403 / 16 | 0.423 / 12 |
| 右转终点 / 超阈值步 | 0.396 / 11 | 0.405 / 11 |
| 加速终点 / 超阈值步 | 0.378 / 未越过 | 0.400 / 45 |
| 刹车终点 / 超阈值步 | 0.439 / 42 | 0.380 / 未越过 |

> 两组不能合并成 6 次实验：同样的 `np.random.seed` 不足以复现，因为 50 辆背景车由 Traffic Manager 的独立随机源驱动。报告里应作为两组独立样本分开呈现；**方向性结论（转向类最早最强）在两组中一致**，但具体终点值与超阈值步会漂移。

---

## 5. 冒烟测试素材（`outputs/carla_cf_smoke`）

用于改脚本后快速验证「CARLA → 世界模型 → 干预 → 出图/出视频」链路，**不用于结论**。

![冒烟版分歧曲线（8 步）](../outputs/carla_cf_smoke/divergence.png)

| 项 | 值 |
| --- | --- |
| 配置 | `--kpi-episodes 1 --anchors 2 --horizon 8` |
| anchor | 15 / 439 |
| 阈值 | 0.0553 |
| KPI | 1 个 episode，447 步，到达 1/1，碰撞 0 |
| 视频 | `videos/cf_grid.mp4`，768×848，5 fps，8 帧，1.6 s |
| 终点分歧 | 左转 0.374、右转 0.353、刹车 0.142、加速 0.137、保持 0.119、噪声底 0.038 |

8 步太短，曲线基本单调上升（对数增长率 0.33–0.45/步），只能说明干预方向有响应，不能外推长期趋势。

---

## 6. 合成环境自检图（`outputs/cf/smoke`，非 CARLA）

![合成小球环境的世界模型反事实四联图](../outputs/cf/smoke/divergence.png)

- 规格：1760×1120（11×7 英寸，160 dpi）
- 来源：`wmcf/counterfactual.py`，输入是 `outputs/runs/smoke/wm.pt`（32×32 图像、4 动作的极小世界模型），环境为带混杂因子的合成小球 `ConfoundedBallEnv`（`confound=True`，`tint_strength=0.45`），**与 CARLA 无关**
- 2×2 四个子图（`counterfactual.py` L338-342）：潜空间 L2 分歧、像素 MSE 分歧、回报绝对差分歧、对真实环境的反事实保真度 MSE；横轴「imagination step」0–3，含 `one_step` 与 `persistent` 两种干预，第一格画噪声底虚线

关键数值（`metrics.json`）：

| 指标 | one_step | persistent |
| --- | ---: | ---: |
| 8 个状态 / 54 条记录 | 24 条 | 30 条 |
| 潜分歧（4 步） | 0.243 → 0.131 | 0.194 → 0.294 |
| 末/首比 | 0.54 | 1.51 |
| 像素分歧 | ~1e-7 | ~1e-6 |
| 回报分歧 | 0.166 → 0.031 | 0.133 → 0.212 |
| 反事实保真度 MSE | 0.095 → 0.146 | 0.083 → 0.146 |

其他：

- `noise_floor_mean = 0.285`，`onset_threshold = 0.570`；两种模式的 `snr_over_noise_floor` 分别为 0.598 / 0.909
- 动作统计：`within_action_cosine = 0.523`，`between_action_cosine = -0.068`，留一质心分类准确率 0.90（随机基线 0.25）
- **`eval_confound = true`**：评估数据也带混杂因子，这张图只能证明「管线与干预实现正确」，不能当作去混杂效果证据

---

## 7. 复现命令

在项目根目录执行（PowerShell，原生 Windows 用 `--platform cpu`）：

```powershell
# 1) 闭环演示：双视角 drive_showcase.mp4 + 每 episode 单视角 mp4
& ".runtime\venv\Scripts\python.exe" -m wmcf.carla.run_demo `
  --cardreamer ".runtime\CarDreamer" --carla-root ".runtime\CARLA_0.9.15" `
  --checkpoint "checkpoints\four_lane.ckpt" --steps 2000 `
  --save-video --logdir "outputs\carla_showcase" --platform cpu

# 2) 反事实：KPI + 分歧曲线 + 八格视频（主结果，60 步）
& ".runtime\venv\Scripts\python.exe" -m wmcf.carla.counterfactual_demo `
  --cardreamer ".runtime\CarDreamer" --carla-root ".runtime\CARLA_0.9.15" `
  --checkpoint "checkpoints\four_lane.ckpt" --kpi-episodes 3 `
  --anchors 4 --horizon 60 --fps 5 --platform cpu --logdir "outputs\carla_cf"

# 3) 冒烟版（约 8 步，1 个 episode）
& ".runtime\venv\Scripts\python.exe" -m wmcf.carla.counterfactual_demo `
  --cardreamer ".runtime\CarDreamer" --carla-root ".runtime\CARLA_0.9.15" `
  --checkpoint "checkpoints\four_lane.ckpt" --kpi-episodes 1 `
  --anchors 2 --horizon 8 --fps 5 --platform cpu --logdir "outputs\carla_cf_smoke"

# 4) 合成环境自检（不需要 CARLA）
& ".runtime\venv\Scripts\python.exe" -m wmcf.counterfactual `
  --config "configs\smoke.yaml" --checkpoint "outputs\runs\smoke\wm.pt"
```

前置条件：CARLA 服务端已启动（端口 2000）且 Town04 已预载，否则 `WorldManager` 会在 20 秒后超时。详见 README 0.4 节。

---

## 8. 报告 / 答辩选材建议

| 场景 | 推荐素材 | 理由 |
| --- | --- | --- |
| 开场演示 | `drive_showcase.mp4` | 双视角 + 动作条，一眼看懂「看什么 → 输出什么」 |
| 方法机制 | `carla_cf/divergence.png` + `cf_contact_sheet.png` | 曲线给量化，九宫格给直觉 |
| 动态说明反事实 | `carla_cf/videos/cf_grid.mp4` | 逐帧展示 8 条分支如何分叉 |
| 复现性说明 | `carla_cf_branches/*` 两张图 | 与主结果对照，体现方差 |
| 流程可靠性 | `carla_cf_smoke/*` | 说明每步改动都经过快速回归 |
| 不建议入报告 | `carla_demo_video/*`、`cf/smoke/divergence.png` | 前者是早期链路记录，后者是合成环境且 `eval_confound=true`，容易被误读为 CARLA 结论 |

这些视频由 `imageio` 逐帧写入，不含音轨；`outputs/` 不入库，提交前请勿误 `git add`。
