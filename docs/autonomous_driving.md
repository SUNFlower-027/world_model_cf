# 智能驾驶场景：把「反事实动作干预」讲成一个驾驶故事

本文档面向报告与后续实验，说明如何用 **CARLA + CarDreamer** 把本项目的「世界模型想象轨迹分歧」放到智能驾驶场景中展示。
内容分三部分：报告叙事、CarDreamer 事实清单与配置要求、可执行的实验路线。

> 结论先行：本项目的主实验仍然是可复现的轻量 RSSM 基线；CarDreamer 提供真实驾驶场景、预训练 checkpoint 和闭环视频，用来回答「这个现象在真实驾驶里是否存在」。两者是**机制验证 + 外部效度**的关系，不是替代关系。

---

## 1. 报告叙事（6 页故事板）

| 页 | 内容 | 素材 |
| --- | --- | --- |
| 1 | 场景：四车道跟车，前车突然减速。世界模型需要想象「自车刹车 / 保持 / 加速」三种未来 | CARLA 截图或 CarDreamer 闭环视频 |
| 2 | 问题：世界模型可能靠场景里的捷径（他车轨迹、道路结构、背景线索）预测未来，而不是真的把自车动作当因果变量 | 一张混淆图：背景线索 vs 动作 |
| 3 | 方法：固定历史，只改动作 `do(a')`，看想象轨迹何时、以多快的速度分开 | 方法示意图（可用 `outputs/cf/*/divergence.png`） |
| 4 | 合成实验：有混淆 vs 去混淆，分歧曲线、噪声底、SNR、反事实真值误差 | 现有 `confounded_vanilla` / `unconfounded` 结果 |
| 5 | 真实场景：CarDreamer checkpoint 在 CARLA 里的闭环表现 + 同一套指标 | 视频 + KPI 表 + 分歧曲线 |
| 6 | 结论：动作敏感性可以作为世界模型的「安全体检」；局限性（仿真、随机性、单一场景） | 总结页 |

讲述主线建议：

> 「我们不是问世界模型预测得准不准，而是问它**有没有把自车动作当成原因**。在驾驶里，这个问题等价于：如果刚才踩的是刹车而不是油门，模型想象出的未来会不会不同？如果不会，规划器就会系统性低估风险。」

---

## 2. 与现有基线的对应关系

| 本项目的诊断 | 驾驶场景中的含义 | CarDreamer / CARLA 中怎么落地 |
| --- | --- | --- |
| 背景色泄漏「即将执行的动作」 | 场景中存在与动作相关、但并非因果的捷径 | BEV 里他车的未来轨迹/意图、道路结构、交通灯状态 |
| `do(a')` 反事实干预 | 刹车 / 保持 / 加速 / 左转 / 右转 | 15 个离散动作（5 转向 × 3 加减速） |
| 分歧起始步 | 模型对动作的「反应延迟」 | 干预后多少步两条想象轨迹开始分开 |
| 分歧增长率 | 风险放大的速度 | 刹车分支与加速分支的轨迹分离速度 |
| 噪声底 / SNR | 分歧是否只是随机采样噪声 | 固定动作、只换后验采样，得到噪声底 |
| 反事实真值 MSE | 想象是否可信 | CARLA 随机性强，建议用固定种子重放或开环未来帧预测替代 |
| 动作方向一致性 | 分歧方向是否符合动作语义 | 刹车分支应更早出现减速/碰撞风险 |
| LOO 动作可分性 | 动作是否真的被使用 | 留一动作分类，越接近 1 越好 |

**关键映射**：现有的「背景色混淆」在驾驶里最自然的对应物是 **BEV 中他车意图/未来轨迹**——如果观测里直接给出了周围车辆的未来轨迹，世界模型就可以不做自车动作的反事实推理。CarDreamer 论文本身也系统研究了观测模态、可观测性和意图共享对安全性的影响，可以直接借用它的实验设置。

---

## 3. CarDreamer 事实清单

以下信息来自官方仓库与论文（2026-11 核对）：

- 论文：*CarDreamer: Open-Source Learning Platform for World Model based Autonomous Driving*，arXiv:2405.09111，IEEE IoT 2024。
- 仿真器：**CARLA 0.9.15**，Gym 接口。
- 世界模型：**DreamerV3（JAX 实现）**，同时提供 DreamerV2；模型与任务解耦。
- 观测：`camera`（512×512）、`birdeye_wpt`（128×128×3，世界模型默认输入）、`collision` 等。
- 动作：离散 **5 转向 × 3 加减速 = 15**，外层用 one-hot 包装。
- 任务（16 个 checkpoint 对应）：
  `four_lane`、`lane_merge`、`left_turn_simple/medium/hard`、`right_turn_raw/simple/medium/hard/fov/sfov`、`roundabout`、`overtake`、`stop_sign`、`traffic_light`、`navigation`。
- 训练资源（官方 README）：单张 4090，显存 **10–20 GB**，另加 **3 GB** 给 CARLA。

### 3.1 预训练 checkpoint

- 地址：<https://huggingface.co/ucd-dare/CarDreamer/tree/main/CarDreamer_checkpoints>
- 每个文件 **370,989,588 字节（约 354 MiB）**，共 16 个。
- 格式：Python **pickle**（`embodied/core/checkpoint.py` 用 `pickle.dumps` 写入），内容主要是 JAX 参数树（numpy 数组）。**不要用 `torch.load`，也不要在不信任的来源上反序列化。**
- 目录约定：

```
CarDreamer_checkpoints/
  four_lane.ckpt
  lane_merge.ckpt
  left_turn_hard.ckpt
  ...
```

下载方式（任选）：

```bash
# 方式一：huggingface-cli
pip install -U huggingface_hub
huggingface-cli download ucd-dare/CarDreamer \
  CarDreamer_checkpoints/four_lane.ckpt --local-dir .
```

```bash
# 方式二：curl（支持断点续传）
curl -L -C - -o CarDreamer_checkpoints/four_lane.ckpt \
  https://huggingface.co/ucd-dare/CarDreamer/resolve/main/CarDreamer_checkpoints/four_lane.ckpt
```

国内网络可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

本项目自带检查工具（见第 5 节），只依赖 `numpy`，不需要 JAX/CARLA：

```bash
python -m wmcf.carla.checkpoints --list
python -m wmcf.carla.checkpoints --download four_lane --dest checkpoints
python -m wmcf.carla.checkpoints --inspect checkpoints/four_lane.ckpt
```

> 工具内置容错 unpickler：checkpoint 里含 `optax` 优化器状态对象，本机未装 JAX/optax 时会自动用占位类（stub）替代，仍可遍历全部 numpy 参数并统计参数量。需要真正加载权重做推理时，才必须在 `cardreamer` 环境里用官方 `embodied.Checkpoint`。

### 3.2 实测 checkpoint 结构（已验证）

2026-11-22 在本机对 `four_lane.ckpt` 实测（`--inspect` + 参数树遍历），结果如下，可直接作为后续复现/干预的对照基准。

**顶层结构**：`step`、`agent`、`replay`、`_timestamp`（保存时间 2024-06-07）。被 stub 的类：`optax.EmptyState`、`optax.ScaleByAdamState`、`optax.ApplyIfFiniteState`（即 Adam 优化器的一阶/二阶动量，占了文件大头）。

**RSSM 配置**（从参数形状反推，非猜测）：

| 参数 | 形状 | 推断含义 |
| --- | --- | --- |
| `rssm/initial` | `(512,)` | 确定性隐状态 `deter = 512` |
| `rssm/img_in/kernel` | `(1039, 512)` | 输入 = 上一帧 stoch(1024) + 动作(15)，隐层 `units = 512` |
| `rssm/img_stats/kernel` | `(512, 1024)` | 先验 logits = 1024 → `stoch = 32, classes = 32` |
| `rssm/obs_stats/kernel` | `(512, 1024)` | 后验 logits = 1024（同上） |
| `rssm/obs_out/kernel` | `(8704, 512)` | 输入 = deter(512) + 图像嵌入(8192) |
| `rssm/gru/kernel` | `(1024, 1536)` | GRU，`3 × deter = 1536` |

**Encoder CNN**：5 层 4×4 stride 卷积，通道 `32→64→128→256→512`。输入 128×128×3 经 5 次减半到 4×4×512，展平为 **8192** 维嵌入（与 `obs_out` 输入吻合）。

**Actor / Critic**：
- Actor：`h0/kernel (1536, 512)`，输入 = deter(512) + stoch(1024) = 1536；输出层 `(512, 15)` → **15 个离散动作**。
- Critic：`h0/kernel (1536, 512)`，价值头 `(512, 255)` → 255 桶 two-hot 价值分布。

**参数量**：模型本体 91 个数组、**31.7M 参数（约 121 MiB fp32）**；另有 181 个优化器状态数组，合计使文件达 354 MiB。

**对后续工作的意义**：
- CarDreamer 的 RSSM 比本仓库轻量基线大（deter 512 vs 128，5 层 CNN vs 4 层），但 **stoch/classes = 32/32 与动作离散化的设计一致**，说明本项目的干预分析范式可以直接迁移。
- 动作维 15、价值 255 桶、latent 特征 1536，是写 5.4 节 JAX 干预代码时必须对齐的硬参数。
- 若只想做「轻量桥接」（5.3 节），只需用 checkpoint 跑 eval 采数据，无需加载这些权重。

---

## 4. 配置要求

### 4.1 硬件

| 用途 | 显存 | 内存 | 磁盘 | 说明 |
| --- | --- | --- | --- | --- |
| 现有轻量基线 | 4 GB（CPU 可跑 smoke） | 8 GB | 1 GB | 已在本机验证 |
| CARLA 仅渲染 + 官方 eval | **8 GB 起**（低画质） | 16 GB | 30 GB | 本机 RTX 4060 Laptop 8 GB 属于「能试、不宽裕」 |
| CarDreamer 训练 | **24 GB（4090）** | 32 GB | 50 GB+ | 官方口径 10–20 GB + 3 GB CARLA |
| 完整 checkpoint 集 | — | — | 约 5.7 GB | 16 × 354 MiB |

本机（RTX 4060 Laptop，8 GB）建议只做：

- 官方 checkpoint 的 **eval-only** 闭环演示；
- CARLA 画质调低、关闭额外传感器；
- 不在本机重训 CarDreamer。

### 4.2 软件

| 组件 | 版本 |
| --- | --- |
| OS | Ubuntu 20.04/22.04（推荐）；Windows 10/11 可跑 CARLA，但官方脚本是 bash |
| Python | 3.10（conda 环境 `cardreamer`） |
| CARLA | 0.9.15（服务端 + PythonAPI） |
| JAX | 0.4.34（CUDA 12） |
| TensorFlow | 2.13（DreamerV3 依赖） |
| 其他 | `carla==0.9.15`、`gym==0.12.5`、`opencv-python`、`moviepy` 等，见 `dreamerv3/requirements.txt` |

安装要点（官方流程）：

```bash
conda create -n cardreamer python=3.10 -y
conda activate cardreamer

# car_dreamer 本体
cd CarDreamer
pip install flit
flit install --symlink

# DreamerV3 依赖
cd dreamerv3
pip install -r requirements.txt
pip install "jax[cuda12_pip]==0.4.34" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

export CARLA_ROOT=/path/to/carla
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla":${PYTHONPATH}
export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export CUSOLVER_PATH=$(dirname $(python -c "import nvidia.cusolver;print(nvidia.cusolver.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CUSOLVER_PATH/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Windows 提示：

- CARLA 服务端用官方 Windows 包，客户端脚本建议在 **WSL2** 里跑；
- 官方 `train_dm3.sh` / `eval_dm3.sh` 是 bash 脚本，直接移植到 PowerShell 需要改写端口、进程管理和 `CUDA_VISIBLE_DEVICES`；
- WSL2 访问 Windows 上的 CARLA 需要把 `localhost` 换成宿主机 IP，或反过来在 WSL 内跑 CARLA（需要 GPU 直通，较麻烦）。

---

## 5. 实验路线

### 5.1 Step 0：素材与基线（现在就能做）

- 用现有 `confounded_vanilla` / `unconfounded` 结果出分歧曲线和指标表；
- 从 CarDreamer 项目页/论文取 1–2 段闭环视频作为「真实场景」引入；
- 本仓库的 checkpoint 工具先确认 checkpoint 可读、任务名正确。

### 5.2 Step 1：CarDreamer 闭环演示（需要 CARLA + GPU）

官方 eval：

```bash
bash eval_dm3.sh 2000 0 ./CarDreamer_checkpoints/four_lane.ckpt \
  --task carla_four_lane \
  --dreamerv3.logdir ./logdir/eval_four_lane
```

跑完后统计指标：

```bash
python dreamerv3/eval_stats.py --logdir ./logdir/eval_four_lane
```

报告素材：

- `camera` 视角视频（前视）；
- `birdeye_wpt` 视角视频（世界模型输入）；
- `collision`、`destination_reached`、`out_of_lane`、`ttc`、`travel_distance` 等 KPI。

建议至少跑：

| 任务 | 用途 |
| --- | --- |
| `carla_four_lane` | 主场景：结构化跟车/换道 |
| `carla_left_turn_hard` | 安全关键：无保护左转 |
| `carla_stop_sign` / `carla_traffic_light` | 规则约束下的动作敏感性 |

每个任务建议 **3 个随机种子**，报告均值 ± 标准差。

### 5.3 Step 2：把驾驶数据接入本项目（推荐的桥）

目标：用同一套 `do(a')` 指标分析驾驶场景，而不是另起炉灶。

1. 在 CarDreamer 的 eval 循环里记录每个 episode：
   - `birdeye_wpt`（uint8，128×128×3）；
   - 执行的 15 维 one-hot 动作（转成 0–14 的索引）；
   - reward、`is_first`、`is_terminal`、collision。
2. 存成 `npz`（每条 episode 一个文件），字段建议：
   `obs [T+1,128,128,3] uint8`、`act [T] int64`、`rew [T] float32`、`done [T] bool`。
3. 用本项目的轻量 RSSM 在该数据上训练：
   - `model.n_actions: 15`
   - `data.img_size: 128`
   - `data.episode_len` 与采样序列长度按显存调整
4. 跑 `counterfactual.py`，把干预动作换成驾驶语义：
   `do(刹车)`、`do(保持)`、`do(加速)`，以及左右转向。
5. 报告里同时给出：
   - 合成环境的机制曲线；
   - 驾驶数据的同指标曲线；
   - CarDreamer checkpoint 的闭环 KPI。

> 注意：当前 `counterfactual.py` 的反事实真值依赖 `ConfoundedBallEnv` 的解析动力学。接入驾驶数据时，真值 MSE 需要改为「固定种子重放」或「开环未来帧预测」，或暂时只报告潜空间/像素分歧与噪声底。这一步是代码主负责的主要改动点。

### 5.4 Step 3（可选，进阶）：直接干预 CarDreamer 的 JAX 世界模型

CarDreamer 的 DreamerV3 是 JAX 实现，内部接口大致为：

- `agent.wm.rssm.obs_step(prev_state, prev_action, embed, is_first)`：用真实观测更新后验；
- `agent.wm.rssm.img_step(prev_state, prev_action)`：只靠动作在潜空间里想象下一步；
- `agent.wm.rssm.get_dist(state)`：得到离散潜变量的分布。

因此可以在真实 rollout 的某个时刻取后验 `(deter, stoch)`，固定它，然后对候选动作序列分别调用 `img_step`，比较 `get_dist` 的 KL/JS 分歧。这条路线能直接回答「CarDreamer 的世界模型是否对自车动作敏感」，但需要：

- 在 `cardreamer` conda 环境里运行；
- 与 checkpoint 的训练配置（`small`/`medium`/`xlarge`）严格对齐；
- 处理 JAX `nj.pure` 的状态传递与随机数；
- 对 CARLA 的非确定性做固定种子/固定交通流处理。

建议把它作为**进阶目标**：先完成 5.3 的轻量桥接，拿到可重复的驾驶数据结果，再决定是否投入 5.4。

### 5.5 本机实测结果（2026-11-22）

环境：Windows 11 + CARLA 0.9.15 + Python 3.10 + JAX 0.4.34（CPU 后端），
权重为官方预训练 `four_lane.ckpt`，仅推理、不训练。入口脚本：
`wmcf/carla/run_demo.py`（闭环演示）与 `wmcf/carla/counterfactual_demo.py`（5.4 路线）。

复现命令（项目根目录）：

```powershell
& ".runtime\venv\Scripts\python.exe" -m wmcf.carla.counterfactual_demo `
  --cardreamer ".runtime\CarDreamer" --carla-root ".runtime\CARLA_0.9.15" `
  --checkpoint "checkpoints\four_lane.ckpt" --kpi-episodes 3 `
  --anchors 4 --horizon 20 --fps 5 --platform cpu --logdir "outputs\carla_cf"
```

#### 运行 A：20 步反事实，3 个种子

| seed | 步数 | 回报 | 碰撞 | 出车道 | 到达终点 | 里程 (m) | 均速 (m/s) | 平均横向偏差 (m) | 最大横向偏差 (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 448 | 1024.0 | 0 | 0 | 1 | 157.3 | 3.526 | 0.918 | 1.505 |
| 1 | 478 | 1088.0 | 0 | 0 | 1 | 164.7 | 3.461 | 0.909 | 1.489 |
| 2 | 460 | 1038.0 | 0 | 0 | 1 | 158.0 | 3.445 | 0.921 | 1.438 |
| 均值 | 462.0 ± 12.3 | 1050.1 ± 27.5 | 0.00 | 0.00 | 1.00 | 160.0 ± 3.4 | 3.478 ± 0.035 | 0.916 ± 0.005 | 1.477 ± 0.028 |

潜空间分歧（4 个干预起点 × 20 步，JS 分歧，单位 nats）：

| 分支 | 终点分歧 | 超阈值起始步 |
| --- | --- | --- |
| do(右转) | 0.426 | 3 |
| do(左转) | 0.413 | 5 |
| do(刹车) | 0.282 | 11 |
| do(加速) | 0.220 | 13 |
| do(保持) | 0.220 | 15 |
| 噪声底 | 0.083 | — |

显著性阈值 0.212（噪声底均值 + 2 倍标准差）。

#### 运行 B：60 步反事实，3 个种子

| seed | 步数 | 回报 | 碰撞 | 出车道 | 到达终点 | 里程 (m) | 均速 (m/s) | 平均横向偏差 (m) | 最大横向偏差 (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 446 | 1028.0 | 0 | 0 | 1 | 157.1 | 3.541 | 0.905 | 1.483 |
| 1 | 432 | 972.5 | 1 | 0 | 0 | 150.1 | 3.482 | 0.923 | 1.481 |
| 2 | 472 | 959.5 | 1 | 0 | 0 | 151.5 | 3.228 | 0.924 | 1.457 |
| 均值 | 450.0 ± 16.6 | 986.8 ± 29.8 | 0.67 | 0.00 | 0.33 | 152.9 ± 3.1 | 3.417 ± 0.136 | 0.917 ± 0.009 | 1.474 ± 0.012 |

潜空间分歧（4 个干预起点 × 60 步）：

| 分支 | 终点分歧 | 超阈值起始步 |
| --- | --- | --- |
| do(刹车) | 0.440 | 42 |
| do(保持) | 0.415 | 未越过 |
| do(左转) | 0.403 | 16 |
| do(右转) | 0.396 | 11 |
| do(加速) | 0.378 | 未越过 |
| 噪声底 | 0.164 | — |

显著性阈值 0.433（60 步时噪声底随时间增大，阈值随之抬高，因此不同 horizon 的终点分歧不可直接横比）。

#### 两次运行的关系与限制

- 两次都是 3 个 episode，但同样 seed 下结果不同：CARLA Traffic Manager 管理 50 辆背景车时使用独立随机源，仅设 `np.random.seed` 不足以复现。因此**运行 A 与运行 B 是两组独立样本，不能合并成 6 次实验**，报告里应分开呈现。
- 运行 A 到达率 3/3，运行 B 到达率 1/3，说明单次 3 episode 的估计方差很大；正式结论建议固定 Traffic Manager 种子，或把样本扩到 10 个 episode 以上再报均值。
- 结论方向在两次运行中一致：**转向类动作最早、最强地改变世界模型潜状态；刹车次之；保持与加速最晚**。
- 该实验是「固定真实观测后验 + 只改动作」的潜空间想象，不是真实反事实轨迹，不能写成物理轨迹误差或碰撞率因果结论。

产物（本地生成，`outputs/` 不入库）：

| 文件 | 说明 |
| --- | --- |
| `outputs/carla_cf/kpi_table.md` / `kpi.csv` | 运行 B 的 KPI 表 |
| `outputs/carla_cf/divergence.png` / `.json` | 60 步分歧曲线与逐点数值 |
| `outputs/carla_cf/videos/cf_grid.mp4` | 8 宫格对比，60 步、12 秒、5 fps |
| `outputs/carla_demo_video/videos/episode_001.mp4` | 前视相机闭环演示，512×512、45 秒 |

### 5.6 演示视频（2026-11-22）

用途：课堂/答辩现场播放，不参与结论统计。

| 项 | 值 |
| --- | --- |
| 文件 | `outputs/carla_showcase/videos/drive_showcase.mp4`（本地生成，`outputs/` 不入库） |
| 规格 | 91.6 秒，1024×580，10 fps |
| 结构 | 左右双视角 + 底部动作条；两段 episode 用 `Episode 2` 卡片连接 |
| Episode 1 | 461 步，回报 1061.4，每步回报 2.302 |
| Episode 2 | 452 步，回报 1037.6，每步回报 2.296 |

画面含义：

- 左：`camera (what driver sees)`，CARLA 真实前视渲染，仅给人看；
- 右：`birdeye_wpt (what world model sees)`，策略实际输入；
- 底部：策略选出的离散动作（加速度 + 转向）及 15 维 one-hot；
- 最下方：`camera + BEV -> encoder -> RSSM latent -> policy -> discrete action`。

配置说明：

- 为让画面更接近人眼，录制时把 `camera.fov` 从 `120` 改为 `90`；该参数只影响录像。`carla_four_lane` 的 `encoder.cnn_keys` 只有 `birdeye_wpt`，策略输入与行为不受影响。
- 该修改位于本地 `.runtime` 配置中，不随仓库分发；重装 CarDreamer 后需重新设置。
- 曾尝试把 `num_vehicles` 从 50 降到 10 以降低视觉车流密度，结果 episode 提前终止、回报降至 281–409，说明策略对车流密度敏感（训练分布为 50 辆），因此演示保留 50 辆。

解读边界：

- 两段是同一策略的独立采样，起点车道、随机路线、背景交通都不同，总回报差异主要来自 episode 长度，不能当作策略优劣或方法对比。
- 两段均在 `time_limit=500` 前结束，回报处于到达终点量级；`run_demo.py` 当前只记录长度与回报，未单独记录 `collision` 标志。
- 视频展示的是「世界模型驱动的闭环行驶」，不是反事实想象，也不是训练过程。

---

## 6. 报告图表清单

| 图/表 | 数据来源 | 说明 |
| --- | --- | --- |
| 场景示意图 | CARLA 截图 | 四车道、前车刹车 |
| 方法示意图 | 手绘/现有管线 | 固定历史 → 三个动作分支 |
| 分歧曲线（合成） | `outputs/cf/*/divergence.png` | 已有 |
| 分歧曲线（驾驶） | 5.5 运行 A/B（`outputs/carla_cf/divergence.png`） | 20 步与 60 步两组，已实测 |
| 噪声底 / SNR | `divergence.json` | 判断分歧是否显著 |
| 闭环 KPI 表 | 5.5 运行 A/B（`kpi_table.md`） | collision / success / 横向偏差 |
| 视频 | `outputs/carla_cf/videos/cf_grid.mp4`、`outputs/carla_showcase/videos/drive_showcase.mp4` | 反事实对比 + 双视角闭环演示，报告现场播放 |

---

## 7. 风险与降级方案

| 风险 | 影响 | 降级方案 |
| --- | --- | --- |
| 没有 8 GB+ 空闲显存 / CARLA 起不来 | 无法闭环演示 | 用官方项目页视频 + 合成实验曲线讲机制，明确标注「视频来自 CarDreamer 官方」 |
| HF 下载失败 | 没有 checkpoint | 用镜像或让有网络的队友下载后共享；先跑通合成基线 |
| checkpoint 与 eval 配置不匹配 | 加载报 shape 错误 | 对齐 `dreamerv3.yaml` 的 `small/medium/xlarge`；记录实际配置 |
| CARLA 非确定性 | 无法得到严格反事实真值 | 固定种子 + 固定交通流；只报告潜空间分歧与开环预测 |
| 8 GB 笔记本跑训练 | OOM | 只做 eval；训练放到 24 GB 工作站/云 |
| Windows 脚本不兼容 | 官方 bash 脚本跑不了 | 用 WSL2 或 Linux 工作站；或把 `eval_dm3.sh` 改写成 PowerShell |

---

## 8. 分工建议（代码主负责视角）

| 角色 | 交付物 |
| --- | --- |
| 代码主负责（你） | checkpoint 工具、eval 包装脚本、驾驶数据导出与格式转换、`n_actions=15` 的模型适配、指标复算 |
| 实验 | 3 个任务 × 3 seed 的 eval；KPI 汇总；异常 run 排查 |
| 报告 | 故事板、视频剪辑、图表排版；把「机制验证 → 外部效度」讲清楚 |
| 指导 | 确认「动作敏感性」指标的定义与阈值；审阅结论边界 |

---

## 9. 一句话总结

> 合成环境回答「机制是否存在」，CarDreamer/CARLA 回答「它在真实驾驶里是否重要」。两者用同一套 `do(a')` 干预和分歧指标串起来，报告就有了从玩具到整车的完整叙事。
