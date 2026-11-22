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

### 5.7 接口与对接参考（后续任务对接用）

> 本节把 CARLA 闭环与反事实干预链路涉及的 6 类接口落到可对接的签名、字段形状与代码位置。
> 归属标注：**A** = 本仓库代码（已入库）；**B** = 上游 CarDreamer / DreamerV3（位于 `.runtime/`，被 `.gitignore` 排除，复刻时需先装好，见 README 0.3）。
> 所有形状与字段均核对自实际源码，不做推测。模型配置为 `dreamerv3.yaml` 的 `defaults + small`（`rssm.deter=512`），见 `wmcf/carla/run_demo.py` L74-78 与 `wmcf/carla/load_check.py` L58-67。

#### 5.7.1 Actor 接口：policy 的输入、输出与循环状态

- 入口：`agent = dreamerv3.Agent(env.obs_space, env.act_space, step, dv3)`（`wmcf/carla/run_demo.py` L137）。
- 调用：`agent.policy(obs, state, mode="eval")`（项目里用 `lambda *a: agent.policy(*a, mode="eval")` 固定 eval，见 `run_demo.py` L242 / `counterfactual_demo.py` L567）。
- 实际实现：`.runtime/CarDreamer/dreamerv3/jaxagent.py:50` `JAXAgent.policy(self, obs, state=None, mode="train")`——把输入搬到 device、必要时初始化 state，再调内层 `self.agent.policy`（`.runtime/CarDreamer/dreamerv3/agent.py:47`）。
- **输入 `obs`**：dict，**batch 维在最前**（Driver 用 `BatchEnv` 包一层，B=1）。关键字段：
  - `birdeye_wpt`：`[B, 128, 128, 3]` uint8——策略唯一使用的图像（`encoder.cnn_keys: "birdeye_wpt"`，`tasks.yaml` L54）；
  - `camera`：`[B, 512, 512, 3]` uint8——只录像/显示，不进 encoder；
  - `is_first` / `is_terminal`：`[B]` bool；`reward`：`[B]` float32。
  - 内层 `Agent.preprocess`（`agent.py` L114-125）会把 uint8 图像除以 255、其余转 float32，并补 `cont = 1 - is_terminal`。
- **输入 `state`（循环状态）**：跨 step 传递，`None` 时 `JAXAgent.policy` 调 `policy_initial` 初始化（`jaxagent.py` L55-56）。实际结构是内层 `Agent.policy_initial` 返回的三元组（`agent.py` L37-42）：
  ```python
  state = (
      (prev_latent, prev_action),   # WorldModel 循环状态（prev_latent 见 5.7.4）
      task_state,                   # task_behavior（Greedy）状态
      expl_state,                   # expl_behavior 状态
  )
  ```
  `prev_action` 形状 `[B, 15]`，即上一步的 one-hot 动作。
- **输出 `(outs, state)`**：`outs["action"]` 是采样后的 **15 维 one-hot**，形状 `[B, 15]` float32（`mode="eval"` 时来自 `task_behavior`，`agent.py` L56-59）；`state` 是更新后的循环状态，需原样传回下一步。
- 最小调用示例（来自 `wmcf/carla/load_check.py`，无需 CARLA）：
  ```python
  state = None
  for t in range(steps):
      obs = {"birdeye_wpt": rng.randint(0, 255, (1, 128, 128, 3), np.uint8),
             "is_first": np.array([t == 0]),
             "is_terminal": np.array([False]),
             "reward": np.zeros(1, np.float32)}
      outs, state = agent.policy(obs, state, mode="eval")
      action = np.asarray(outs["action"])[0]   # shape (15,), one-hot
  ```
- 归属：**B**（DreamerV3）。本项目只构造 + 调用，不改实现。

#### 5.7.2 Driver 接口：policy 如何被调用、action 如何执行

- 入口：`driver = embodied.Driver(env)`（`run_demo.py` L238 / `counterfactual_demo.py` L565）。
- 实现：`.runtime/CarDreamer/dreamerv3/embodied/core/driver.py`。
  - `on_step(fn)`：每步调一次 `fn(trn, inf, worker)`；
  - `on_episode(fn)`：每个 episode 结束调一次 `fn(ep, ep_info, worker)`；
  - `__call__(policy, steps=0, episodes=0)`：循环到满足步数或 episode 数。
- 单步 `_step`（`driver.py` L42-76）的关键顺序：
  1. `obs, info = self._env.step(acts)`——用上一步存下的动作推进 env；
  2. `acts, self._state = policy(obs, self._state)`——调策略（传给 policy 的签名即 `policy(obs, state)`）；
  3. 把 `{**obs, **acts}` 作为 transition 累积进 `self._eps`，把 `info` 累积进 `self._eps_info`；
  4. 若 `obs["is_last"]`，触发该环境的 `on_episode` 回调，`ep` / `ep_info` 是 list 转成的数组 dict。
- 项目用法：
  - `driver.on_episode(per_episode)` 收 KPI / 存 episode 数据（`run_demo.py` L239、`counterfactual_demo.py` L566）；
  - `driver.on_step(lambda tran, info, _: step.increment())` 计步；
  - `policy = lambda *a: agent.policy(*a, mode="eval")` 固定 eval 模式；
  - 主循环：`while step < run_args.steps: driver(policy, steps=100)`（`run_demo.py` L244-245），或 `driver(policy, episodes=1)`（`counterfactual_demo.py` L572）。
- `ep` 的 key 来自 transition（`obs` 键 ∪ `acts` 键），如 `reward`、`action`、`birdeye_wpt`、`camera`、`is_first`、`is_last`；`ep_info` 的 key 来自 env 的 `info`（见 5.7.6）。每个 value 形状为 `(episode_len, ...)`。
- 归属：**B**。

#### 5.7.3 CARLA 环境接口：reset / step / 动作格式

- 构造：`env, env_config = car_dreamer.create_task("carla_four_lane", [f"--env.world.carla_port={port}"])`（`run_demo.py` L82）。`env` 是 `gym.Env` 子类 `CarlaFourLaneEnv`（继承 `CarlaWptEnv → CarlaBaseEnv`）。
- 本项目不直接调底层 env，而是包一层（`run_demo.py` L90-112）：
  ```
  FromGym(env) → OneHotAction / ExpandScalars / TimeLimit / ClipAction → BatchEnv([env])
  ```
  所以 Driver 看到的是包裹后的 env。
- 底层 env 接口（`.runtime/CarDreamer/car_dreamer/carla_base_env.py`）：
  - `reset() -> obs`（L92）：重建世界与 observer，返回首帧 obs dict；
  - `step(action) -> (obs, reward, is_terminal, info)`（L140）：`action` 是 **int ∈ [0, 14]**（gym `Discrete(15)`）；内部 `apply_control` → `world.step()` tick 仿真 → 算 reward / terminal / obs。
- 动作格式的两层转换：
  1. 策略输出 15 维 one-hot → `OneHotAction.step`（`.runtime/CarDreamer/dreamerv3/embodied/core/wrappers.py:140-146`）用 `np.argmax(action["action"])` 转回整数 index，再传给底层 env；
  2. 底层 `get_vehicle_control(index)`（`carla_base_env.py` L105-125）把 index 解码为 `(acc, steer)` 并构造 `carla.VehicleControl`（完整映射见 5.7.5）。
- 观测（`carla_four_lane` 启用 `observation.enabled: [camera, collision, birdeye_wpt]`，`tasks.yaml` L39）：
  - `camera`：`[512, 512, 3]` uint8（`common.yaml` L14-18）；
  - `birdeye_wpt`：`[128, 128, 3]` uint8（`common.yaml` L75-96）——**唯一进 encoder 的图像**；
  - `collision`：碰撞传感器信号；
  - `FromGym` 额外补 `reward`、`is_first`、`is_last`、`is_terminal`（`from_gym.py` L33-39）。
- 终止：`get_terminal_conditions`（`carla_wpt_env.py` L137-146）返回 `{is_collision, time_exceeded, out_of_lane, destination_reached}`，任一为真则终止；`time_limit=500`（`tasks.yaml` 的 `carla_wpt` 锚点 L15）。
- 归属：**B**。本项目只通过 `create_task` + 包装器使用。

#### 5.7.4 RSSM 状态说明：状态结构、更新方法、字段形状

- 句柄获取（`counterfactual_demo.py` L215-217）：
  ```python
  wm = agent.agent.wm          # agent 是 JAXAgent，.agent 是内层 Agent，.wm 是 WorldModel
  rssm = wm.rssm
  encoder = wm.encoder
  decoder = wm.heads["decoder"]
  ```
- 类定义：`.runtime/CarDreamer/dreamerv3/nets.py:23` `class RSSM`。
- 配置（本项目 `defaults + small`）：`deter=512`、`stoch=32`、`classes=32`（`dreamerv3.yaml` defaults L64-69 设 `stoch/classes`，small L272 覆盖 `rssm.deter: 512`；`load_check.py` L67 打印 `deter={dv3.rssm.deter}` 确认）。
- state dict 结构（分类模式 `classes>0`，`nets.py` L44-50），batch 维 `B`：

  | 字段 | 形状 | 含义 |
  | --- | --- | --- |
  | `deter` | `[B, 512]` | GRU 确定性状态 `h_t` |
  | `logit` | `[B, 32, 32]` | 32 组 categorical 的 logits，每组 32 类 |
  | `stoch` | `[B, 32, 32]` | 采样后的 one-hot 离散状态 `z_t` |

- 方法：
  - `initial(bs) -> state`（`nets.py` L44）：返回零/可学习初始状态；单序列推理用 `bs=1`。
  - `obs_step(prev_state, prev_action, embed, is_first) -> (post, prior)`（L105）：先 `img_step` 得先验，再用观测 embed 修正得**后验 post**；`embed = wm.encoder({"birdeye_wpt": image})`。
  - `img_step(prev_state, prev_action) -> prior`（L125）：纯**先验**推进，不看观测，用于想象 rollout。
  - 后验解码：`decoder(state)["birdeye_wpt"].mode()` 得到重建 BEV（`counterfactual_demo.py` L230）。
- 项目典型用法（`counterfactual_demo.py` `_make_fns` / `_analyse`）：
  ```python
  state = rssm.initial(1)                                      # 初始
  post, _ = rssm.obs_step(state, prev_action, embed, is_first) # 用真实观测更新后验
  # post 存入 posts；start = posts[anchor] 作为反事实起点
  prior = rssm.img_step(start, action)                         # 开环想象一步
  frame = decoder(prior)["birdeye_wpt"].mode()
  ```
- 形状自洽校验（来自 `_js_divergence`，L240-248）：`prior["logit"]` 取 batch 0 后为 `[32, 32]`，softmax 沿末轴 → `[32, 32]`，沿末轴求和再均值 → 标量，与 `stoch=32 × classes=32` 一致。
- 注意：`JAXAgent` 把 `jax_transfer_guard` 设为 `disallow`（`jaxagent.py` L148），反事实脚本在 `_analyse` 里改回 `allow`（`counterfactual_demo.py` L274）再建 PRNGKey，否则 host↔device 传输会被拒。
- 归属：**B**。

#### 5.7.5 动作映射表：15 个 action_id 与实际控制量

- 来源：`common.yaml` L165-169（`discrete_acc=[-2, 0, 2]`、`discrete_steer=[-0.6, -0.2, 0, 0.2, 0.6]`）+ `carla_base_env.py` L76-81（维度）/ L105-125（解码为 `VehicleControl`）。
- 维度：`n_acc=3`、`n_steer=5` → `action_space = Discrete(15)`；`action_id = acc_idx * n_steer + steer_idx`，即
  `acc = discrete_acc[id // 5]`，`steer_cfg = discrete_steer[id % 5]`。
- 转 CARLA 控制（L118-125）：`acc>0 → throttle=clip(acc/3, 0, 1), brake=0`；`acc≤0 → throttle=0, brake=clip(-acc/3, 0, 1)`；**`steer = -steer_cfg`**（CARLA 中 steer 正 = 左转，故对配置值取负）。
- 完整映射（与 `counterfactual_demo.INTERVENTIONS` 自洽：刹车=2、保持=7、加速=12、左转=5、右转=9）：

  | action_id | acc_idx | steer_idx | acc (m/s²) | config steer | CARLA steer | 含义 |
  |---:|---:|---:|---:|---:|---:|---|
  | 0 | 0 | 0 | −2 | −0.6 | +0.6 | 刹车 + 左转 |
  | 1 | 0 | 1 | −2 | −0.2 | +0.2 | 刹车 + 轻左 |
  | 2 | 0 | 2 | −2 | 0.0 | 0.0 | 刹车 |
  | 3 | 0 | 3 | −2 | +0.2 | −0.2 | 刹车 + 轻右 |
  | 4 | 0 | 4 | −2 | +0.6 | −0.6 | 刹车 + 右转 |
  | 5 | 1 | 0 | 0 | −0.6 | +0.6 | 保持 + 左转 |
  | 6 | 1 | 1 | 0 | −0.2 | +0.2 | 保持 + 轻左 |
  | 7 | 1 | 2 | 0 | 0.0 | 0.0 | 保持（直行） |
  | 8 | 1 | 3 | 0 | +0.2 | −0.2 | 保持 + 轻右 |
  | 9 | 1 | 4 | 0 | +0.6 | −0.6 | 保持 + 右转 |
  | 10 | 2 | 0 | +2 | −0.6 | +0.6 | 加速 + 左转 |
  | 11 | 2 | 1 | +2 | −0.2 | +0.2 | 加速 + 轻左 |
  | 12 | 2 | 2 | +2 | 0.0 | 0.0 | 加速 |
  | 13 | 2 | 3 | +2 | +0.2 | −0.2 | 加速 + 轻右 |
  | 14 | 2 | 4 | +2 | +0.6 | −0.6 | 加速 + 右转 |

- 策略侧是 15 维 one-hot，`argmax` 即 `action_id`；`OneHotAction` wrapper 已保证 `min=0 / max=1 / sum=1`（`wrappers.py` L142-144）。
- 归属：**B**。

#### 5.7.6 KPI 接口：碰撞 / 终点 / 出车道等指标的获取方式

- 每步 `info` dict 在底层 env 拼装（`carla_base_env.py` L150-156）：`info = {**env_state, **terminal_conds, **obs_info, **reward_info, "action": action}`。
- `carla_four_lane` 继承 `carla_wpt`，关键字段（`carla_wpt_env.py` L117-146 + `base_planner.py` L141-145）：

  | 字段 | 形状 | 类型 | 含义 / 来源 |
  | --- | --- | --- | --- |
  | `is_collision` | `[1]` | bool | `obs["collision"][0] > 0`（`carla_base_env.py` L165-170） |
  | `out_of_lane` | `[1]` | bool | `wpt_dis > terminal.out_lane_thres`（thres=3，`tasks.yaml` L16） |
  | `destination_reached` | `[1]` | bool | `len(waypoints) <= 3`（`carla_wpt_env.py` L134-135） |
  | `time_exceeded` | `[1]` | bool | `_time_step > time_limit`（limit=500） |
  | `terminal` | `[1]` | bool | 上述任一为真 |
  | `episode_timesteps` | `[1]` | int | 终止时的步数（仅终止时存在） |
  | `timesteps` | `[1]` | int | 当前步（env_state，`carla_wpt_env.py` L44） |
  | `speed_norm` | `[1]` | float | 车速模长 ‖v‖（m/s） |
  | `wpt_dis` | `[1]` | float | 到下一路点距离（m） |
  | `ttc` | `[1]` | float | 最小碰撞时间（`TTCCalculator`，`toolkit/carla_manager/utils.py`） |
  | `travel_distance` | `[1]` | float | **单步**行驶增量（m），KPI 聚合时 `.sum()` 得总距离 |
  | `ego_x` / `ego_y` | `[1]` | float | 自车位置 |
  | `speed_parallel` / `speed_perpendicular` | `[1]` | float | 沿 / 垂直路点方向的速度 |
  | `r_waypoints` / `r_speed` / `r_collision` / `r_out_of_lane` | `[1]` | float | 各奖励分量 |
  | `num_completed` / `num_obsolete` | `[1]` | int | planner 统计 |

- 键名前缀：`common.yaml` L2 `eval: False`，故 info 键**不带** `eval_` 前缀，项目 `_episode_kpis` 直接读 `is_collision` 等。若把 `eval` 置 True，所有键会被加 `eval_` 前缀（`carla_base_env.py` L157-159），下游聚合键名需同步改。
- 累积与聚合：Driver 把每步 `info` 按 key 堆成 list，episode 结束时 `ep_info[key]` 是 `(T, 1)` 数组传给 `on_episode` 回调。项目 `_episode_kpis(ep_info)`（`counterfactual_demo.py` L134-167）做四类聚合：
  - `rate(key)`：`arr.max() > 0` → 该 episode 是否发生（碰撞 / 出车道 / 到达 / 超时的 0/1）；
  - `mean(key)`：均值（`speed_norm`、`wpt_dis`）；
  - `total(key)`：求和（`travel_distance` → 总行驶距离）；
  - `max_of(key)`：最大值（`wpt_dis` → 最大横向偏差）。
  结果写入 `kpi.csv` / `kpi_table.md`（`_write_kpi`，L170-210）。
- 另一条输出链路：`run_demo.py` L118-133 用 `embodied.logger.JSONLOutput(logdir, "metrics.jsonl")`，靠 `run.log_keys_sum/mean/max` 的正则（如 `travel_distance|destination_reached|out_of_lane|time_exceeded|is_collision`）筛选写入；字段来源同样是 env info。
- 归属：字段定义在 **B**（CarDreamer env），聚合 / 落盘在 **A**（`counterfactual_demo._episode_kpis/_write_kpi`、`run_demo` 的 logger 配置）。

#### 5.7.7 后续任务对接的接入点

对应 README 0.2 列出的待办，给出可直接复用的接入点：

1. **方向一致性检验**（`do(左转)` vs `do(右转)` 解码 BEV 是否反向偏移）
   - 数据源：`outputs/<logdir>/branches.npz`（`counterfactual_demo.py` L465，`np.savez_compressed`）。
   - 结构：`packed` 含 `camera`（uint8 `[H, 512, 512, 3]`，H=`horizon`）+ 各分支 key（float `[H, 128, 128, 3]`，值域 [0,1]）。分支 key 命名见 `BRANCH_VIDEOS`（L39-48）：文件名去掉 `branch_` 前缀和 `.mp4` 后缀、空格转下划线，即 `real`、`factual`、`do_brake`、`do_hold`、`do_accelerate`、`do_left`、`do_right`、`noise_floor`。
   - 用法：`np.load("branches.npz")["do_left"]` 取帧，比较 `do_left` / `do_right` 相对 `factual` 的质心偏移方向是否相反。
   - 注意：该文件是本地产物，不入库；需先跑 `counterfactual_demo.py` 生成。

2. **动作置换负对照**（打乱动作标签，分歧应回落到噪声底）
   - 接入点：`_analyse` 里的 `rollout(start, act_seq, rng_key)`（`counterfactual_demo.py` L311-321）。把 `factual_acts = actions[anchor:anchor+horizon]`（L336）换成 `np.random.permutation(factual_acts)`（沿时间维打乱），其余不变；得到的曲线应与 `noise` 分支接近。
   - 也可在 `INTERVENTIONS`（L31-37）上做置换：随机重映射 15 个 action_id 后再 rollout。

3. **真实强制动作对照**（真实 CARLA 里强制某动作，与世界模型想象对照）
   - CARLA 无法克隆世界状态做严格反事实，只能近似：把 `driver` 的 policy 换成常动作函数，例如：
     ```python
     def constant_policy(action_id):
         vec = np.zeros(15, np.float32); vec[action_id] = 1.0
         return lambda obs, state: ({"action": vec[None]}, state)
     driver(constant_policy(INTERVENTIONS["do(刹车)"]), episodes=1)
     ```
   - 或在 env 层固定路线 / 清空背景车（`num_vehicles=0`）降低随机性，比较真实轨迹与世界模型 `rollout` 解码 BEV。
   - 注意：`num_vehicles` 降到 0 / 10 会让策略偏离训练分布（见 README 0.5），写结论时要标注分布偏移。

4. **取某时刻 RSSM 后验 / 强制动作开环推演**
   - 已实现：`posts = [逐时刻 obs_step 后的后验]`，`start = posts[anchor]`（L292-300、L335）；`rollout` 循环 `jit_imagine(varibs, key, state, action[None])`（L316）。
   - 想换动作序列：直接改 `act_seq`（`[H, 15]` one-hot），调用 `rollout(start, act_seq, rng_key)` 即可，无需改模型。

5. **加载与设备细节**
   - checkpoint 是 Python pickle（`embodied.Checkpoint`），用 `checkpoint.load(path, keys=["agent"])`（`run_demo.py` L139-142）加载，**不要**用 `torch.load`。
   - 原生 Windows JAX 0.4.34 为 CPU 后端，运行必须传 `--platform cpu`（→ `dreamerv3.jax.platform`）。
   - 反事实推理前把 `jax_transfer_guard` 改回 `allow`（`counterfactual_demo.py` L274），否则 `device_put` 报错。

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
