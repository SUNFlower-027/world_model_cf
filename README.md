# 反事实动作干预下世界模型想象轨迹的分歧分析

> 基于 DreamerV3 / RSSM 的轻量复现。目标不是刷 benchmark，而是回答一个可证伪的问题：
> **当世界模型"想象"未来时，它的预测究竟依赖动作，还是依赖数据里与动作相关的虚假线索？**

本目录是团队项目的代码基线。训练前请先读完第 5 节（环境与配置）和第 6 节（运行流程），
再按第 8 节的实验矩阵开跑。

---

## 0. 当前进度与复刻入口（2026-11-22）

### 0.1 已完成

| 模块 | 状态 | 证据 / 产物 |
| --- | --- | --- |
| 合成环境代码基线 | 已完成 | `wmcf/envs.py`、`models.py`、`train_world_model.py`、`counterfactual.py`、`configs/smoke.yaml` |
| CarDreamer checkpoint | 已完成 | `checkpoints/four_lane.ckpt`（约 354 MiB），纯 JAX 加载后能输出 15 维 one-hot 动作 |
| CARLA + Python 环境 | 已完成 | `.runtime/CARLA_0.9.15`、`.runtime/venv`（Python 3.10）、CarDreamer 以 `--pth-file` 安装 |
| 四车道闭环演示 | 已完成 | 91.6 秒双视角视频，两段 episode 461 / 452 步，回报 1061.4 / 1037.6 |
| 反事实干预 | 已完成 | 3 个 seed 的闭环 KPI + 4 个起点 × 60 步潜空间分歧、噪声底与显著性阈值 |
| 文档与脚本 | 已完成 | commit `805cadd`，细节见 [docs/autonomous_driving.md](docs/autonomous_driving.md) 第 5.5、5.6 节 |

### 0.2 尚未完成

| 事项 | 说明 |
| --- | --- |
| H1 / H2 多种子汇总 | CARLA 的 Traffic Manager 使用独立随机源，同样 seed 结果仍会不同；目前每次只有 3 个 episode |
| 方向一致性检验 | 需要验证 `do(左转)` 与 `do(右转)` 在解码 BEV 中是否反向偏移，现有 `branches.npz` 可直接算 |
| 动作置换负对照 | 打乱动作标签后分歧应回落到噪声底，用于排除“换动作必然分叉” |
| 真实强制动作对照 | CARLA 无法克隆世界状态，只能做固定路线/空交通的近似对照 |

### 0.3 复刻准备

硬件：

| 项 | 要求 | 本机实测 |
| --- | --- | --- |
| 系统 | Windows 10/11 64 位 | Windows，原生运行 |
| GPU | NVIDIA，显存 ≥ 8 GB | RTX 4060 Laptop 8 GB |
| 内存 | ≥ 16 GB | 24 GB |
| 磁盘 | ≥ 30 GB | CARLA 压缩包 7.2 GB，解压后更大 |

软件与资源：

1. **CARLA 0.9.15 Windows 包**：`https://downloads.carlasim.com/Windows/CARLA_0.9.15.zip`，解压到 `.runtime/CARLA_0.9.15`。
2. **DirectX 运行库**：若启动时报 `DirectX Runtime`，用 `directx_Jun2010_redist.exe` 解压后运行 `DXSETUP.exe /silent`；在线版 `dxwebsetup.exe` 可能返回 `-9`。
3. **Python 3.10**：建议用 `uv` 或独立安装，不要污染系统 Python。
4. **Python 依赖**：`carla==0.9.15`、`jax==0.4.34`、`jaxlib==0.4.34`（CPU）、`optax==0.2.2`、`tensorflow==2.13.0`、`tensorflow_probability==0.21.0`、`numpy==1.24.3`、`gym==0.12.5`、`imageio`、`opencv-python`。
5. **CarDreamer 源码**：克隆后在 `.runtime/venv` 中执行 `flit install --pth-file`。
6. **checkpoint**：`python -m wmcf.carla.checkpoints --download four_lane --dest checkpoints`。

目录结构约定（`.runtime/` 与 `checkpoints/` 均被 `.gitignore` 排除）：

```text
world_model_cf/
├── .runtime/
│   ├── CARLA_0.9.15/   # CarlaUE4.exe
│   ├── CarDreamer/     # 源码
│   └── venv/           # Python 3.10 + 依赖
├── checkpoints/
│   └── four_lane.ckpt
└── outputs/            # 视频、KPI、图
```

### 0.4 复刻命令

```powershell
# 1) 启动 CARLA（离屏、低画质、2000 端口）
Start-Process ".runtime\CARLA_0.9.15\CarlaUE4.exe" -ArgumentList "-carla-port=2000 -RenderOffScreen -quality-level=Low -unattended"

# 2) 预载 Town04，避免 WorldManager 的 20 秒超时
& ".runtime\venv\Scripts\python.exe" -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(60.0); c.load_world('Town04'); print('Town04 cached')"

# 3) 闭环演示 + 双视角视频
& ".runtime\venv\Scripts\python.exe" -m wmcf.carla.run_demo --cardreamer ".runtime\CarDreamer" --carla-root ".runtime\CARLA_0.9.15" --checkpoint "checkpoints\four_lane.ckpt" --task carla_four_lane --steps 1000 --platform cpu --logdir "outputs\carla_showcase" --save-video

# 4) 反事实干预：KPI + 分歧曲线 + 宫格视频
& ".runtime\venv\Scripts\python.exe" -m wmcf.carla.counterfactual_demo --cardreamer ".runtime\CarDreamer" --carla-root ".runtime\CARLA_0.9.15" --checkpoint "checkpoints\four_lane.ckpt" --kpi-episodes 3 --anchors 4 --horizon 60 --platform cpu --logdir "outputs\carla_cf"
```

### 0.5 已知坑

| 现象 | 原因 / 处理 |
| --- | --- |
| `load_world("Town04")` 超时 | 首次加载地图缓存超过 20 秒；先执行上面的预载命令 |
| JAX 报找不到 GPU | 原生 Windows 的 JAX 0.4.34 是 CPU 后端；运行脚本必须加 `--platform cpu` |
| 启动弹 `DirectX Runtime` 错误 | 安装离线 DirectX 运行库，见 0.3 |
| 降低 `num_vehicles` 后频繁提前结束 | 策略训练分布是 50 辆；实测降到 10 辆时回报只有 281–409，演示保持 50 |
| 同样 seed 结果不同 | Traffic Manager 的随机源未被固定；KPI 需多 episode 取均值，不能合并不同次的样本 |

### 0.6 接口映射（设计文档 → 代码）

对照设计文档里 5 个必需接口，给出实际代码位置与归属（`A` = 本项目代码，`B` = 上游 CarDreamer，位于 `.runtime/`，不入库）：

| 设计文档接口 | 归属 | 在本仓库？ | 代码位置 |
| --- | --- | --- | --- |
| 原始 CarDreamer Actor 的动作调用 | A/B | ✅ 调用层在 | [run_demo.py](wmcf/carla/run_demo.py) `agent.policy(*a, mode="eval")`；[counterfactual_demo.py](wmcf/carla/counterfactual_demo.py) 同 |
| 当前时刻 RSSM 状态获取 | A/B | ✅ 有 | [counterfactual_demo.py](wmcf/carla/counterfactual_demo.py) 逐真实观测更新后验存入 `posts`，`start = posts[anchor]` 作为反事实起点；RSSM 句柄 `wm.rssm` |
| 多候选动作 imagine | A/B | ✅ 有 | [counterfactual_demo.py](wmcf/carla/counterfactual_demo.py) `imagine_fn`（`rssm.img_step` + decoder）、`rollout`，动作映射见 `INTERVENTIONS` |
| CARLA 动作执行 | B | ⚠️ 底层不在 | 调用层：[run_demo.py](wmcf/carla/run_demo.py) `embodied.Driver(env)`；真正把动作发给车辆的是上游 CarDreamer 的 env（`.runtime/CarDreamer`，被 `.gitignore` 排除） |
| 碰撞 / 终点 / 出车道 KPI 记录 | A | ✅ 有 | [counterfactual_demo.py](wmcf/carla/counterfactual_demo.py) `_episode_kpis` / `_write_kpi`；[run_demo.py](wmcf/carla/run_demo.py) 写 `metrics.jsonl` |

> 设计文档里 `A/B` 表示“接口由本项目调用 / 底层由 CarDreamer 提供”。`CARLA 动作执行` 的底层实现属于运行时依赖，不在本仓库，复刻时需先按 0.3 装好 CarDreamer。

---

## 1. 研究问题与假设

### 1.1 问题

给定同一段历史观测 $x_{\le k}$，固定除动作外的一切条件，把动作从 $a_k$ 改成 $a'$：

$$
\text{do}(a_k = a') \quad\Longrightarrow\quad
\hat{z}^{cf}_{k:t},\ \hat{x}^{cf}_{k:t},\ \hat{r}^{cf}_{k:t}
$$

比较反事实想象轨迹与事实想象轨迹的差异。差异应当：

1. 在动作真正影响环境的方向上出现；
2. 明显高于"同一动作、重采样初始后验"得到的噪声底；
3. 随干预持续时间和预测步数增长，而不是立刻饱和或爆炸；
4. 与真实环境中的反事实轨迹一致（本项目的合成环境提供闭式真值）。

### 1.2 核心假设

| 编号 | 假设 | 预期现象 |
| --- | --- | --- |
| H1 | 若训练数据存在"动作线索泄漏"，模型会走捷径 | 干预后潜轨迹分歧小、与真值误差大 |
| H2 | 打破线索（去混淆数据）后，模型更依赖动作输入 | 分歧更大、方向更一致、CF 保真度更高 |
| H3 | 打乱动作训练出的"无视动作"模型是有效负对照 | 分歧接近噪声底，方向一致性接近随机 |
| H4 | 逆动力学正则能增强潜变量中的动作信息 | 动作方向一致性 / 线性可分性提升 |

H1 与 H2 的对比是本项目的主结果，H3 用于验证指标本身有效，H4 是可选消融。

### 1.3 因果口径（诚实说明）

Pearl 的 do-calculus 主要处理"动作与隐藏混杂因素纠缠"时 $P(Y\mid do(a))$ 能否识别。
在本项目中动作是智能体主动选择的外生输入，所选动作与潜状态之间没有后门路径，
因此 `do(a_k=a')` 的实现方式就是**在 rollout 时把动作钳住**，不做图结构手术。
我们保留 do 记号，是因为它准确命名了概念，也明确了适用范围。

---

## 2. 参考文献与课程对应

| 来源 | 用途 |
| --- | --- |
| DreamerV3, *Mastering Diverse Domains through World Models*, arXiv:2301.04104 | RSSM 结构、KL balancing、free bits、symlog、continue head、隐空间 actor-critic |
| [learn-world-model](https://github.com/datawhalechina/learn-world-model) L02 | 部分可观测、belief state、GRU → MDN-RNN → RSSM、free rollout |
| 同课程 P02（RSSM 动力学） | 编码器 + RSSM 的训练与 prior/posterior rollout 诊断 |
| 同课程 P03（Dreamer 智能体） | 想象式 actor-critic 训练循环 |
| 同课程 P05（评估仪表盘） | 重构、rollout、latent drift 等评估口径 |
| 同课程 P06（反事实动作条件世界模型） | 因果阶梯、反事实 rollout、逆动力学正则、action-influence 指标 |

与 P06 的差别：P06 是教学 notebook，用少量指标演示"动作是否被使用"；
本项目把同一问题扩展成**可重复的对照实验 + 轨迹级分歧曲线 + 真假值校验 + 负对照**，
并显式构造"动作线索泄漏"的数据生成过程来制造捷径。

### 2.1 智能驾驶场景（报告主线）

为了让结论不只停留在合成环境，项目增加一条 **CARLA + CarDreamer** 的外部效度线：

- 场景：四车道跟车，前车减速，比较 `do(刹车)` / `do(保持)` / `do(加速)` 下的想象轨迹；
- 平台：[CarDreamer](https://github.com/ucd-dare/CarDreamer)（CARLA 0.9.15 + DreamerV3 JAX）；
- 预训练权重：[HuggingFace](https://huggingface.co/ucd-dare/CarDreamer/tree/main/CarDreamer_checkpoints)（16 个任务，每个约 354 MiB）。

完整的故事板、硬件/软件配置、实验步骤和风险降级方案见
[docs/autonomous_driving.md](docs/autonomous_driving.md)。

本目录新增的工具（不依赖 CARLA/JAX，可先跑）：

```bash
python -m wmcf.carla.checkpoints --list
python -m wmcf.carla.checkpoints --download four_lane --dest checkpoints
python -m wmcf.carla.checkpoints --inspect checkpoints/four_lane.ckpt
python -m wmcf.carla.summarize_eval --logdir <CarDreamer eval logdir>
```

> 注意：CarDreamer checkpoint 是 Python pickle，不是 PyTorch 权重，**不要**用 `torch.load`。

---

## 3. 方法概览

### 3.1 世界模型

- 观测：$64\times64$ RGB（smoke 配置为 $32\times32$）
- 编码器：4 层 stride-2 卷积 → 256 维嵌入
- RSSM：确定性状态 $h_t$ + 离散随机状态 $z_t$（32 组 × 32 类）

$$
h_t = \mathrm{GRU}(h_{t-1}, [z_{t-1}, a_{t-1}]),\quad
p(z_t\mid h_t),\quad q(z_t\mid h_t, x_t)
$$

- 损失：重构 MSE + reward symlog-MSE + continue BCE + KL balancing
  （$\alpha=0.8$，每类 free bits 1 nat）+ 逆动力学交叉熵
- 想象：只用 prior，不接触真实观测；动力学用 **MAP 潜变量**（确定性）推进，
  分歧则在 **prior 概率向量**上计算，得到连续、无采样噪声的信号；
  另用"重采样初始后验"估计噪声底

### 3.2 合成环境：ConfoundedBall

球在画面中移动，4 个离散动作（左/右/上/下），奖励为到目标的指数距离。

**关键设计：背景色是"动作意图线索"。**

- `confound=true`：第 $t$ 帧的背景色由**即将执行的动作** $a_t$ 决定。
  模型可以只看背景就预测下一帧，不需要读动作输入。
- `confound=false`：背景固定，唯一能预测转移的信息就是动作。

评估时保持 `eval_confound=true`：事实轨迹的线索与动作一致；一旦做
`do(a_k=a')`，输入里的线索仍指向原动作，二者冲突。走捷径的模型会"照着线索想象"，
因此反事实分歧变小、与真实反事实的误差变大。这正是要测量的现象。

真值可用：环境状态就是球坐标，给定 `do(a')` 可以直接仿真出真实反事实画面。

补充：该合成环境没有显式终止，`continue` 目标恒为 1。这是为了获得稳定的长程分歧曲线；
`continue` 头保留是为了与 Dreamer 结构一致，但在本环境中不提供额外信号。

---

## 4. 目录结构与模块职责

```
world_model_cf/
├── README.md                     # 本文件：训练前说明
├── requirements.txt
├── configs/
│   ├── default.yaml              # 正式实验
│   └── smoke.yaml                # CPU 冒烟测试
├── docs/
│   └── autonomous_driving.md     # 智能驾驶场景：叙事、CarDreamer 配置、实验路线
├── wmcf/
│   ├── config.py                 # YAML 配置 + --set 覆盖
│   ├── utils.py                  # seed / device / JSON / CSV
│   ├── envs.py                   # ConfoundedBall 环境与渲染
│   ├── data.py                   # episode 生成、序列采样
│   ├── models.py                 # Encoder / Decoder / RSSM / WorldModel / Actor / Critic
│   ├── train_world_model.py      # 世界模型训练入口
│   ├── train_agent.py            # 想象式 actor-critic（可选）
│   ├── counterfactual.py         # 干预、分歧指标、图表与 JSON
│   └── carla/
│       ├── checkpoints.py        # CarDreamer checkpoint 列表/下载/检查
│       ├── load_check.py         # 纯 JAX 加载 checkpoint 并输出动作自检
│       ├── run_demo.py           # 四车道闭环推理 + 双视角视频
│       ├── counterfactual_demo.py # do(a') 干预、分歧曲线、分支视频
│       └── summarize_eval.py     # CarDreamer metrics.jsonl → 报告表格/图
└── outputs/                      # 运行产物（自动创建）
    ├── runs/<run_name>/          # wm.pt / ac.pt / 日志 / 配置快照
    └── cf/<run_name>/            # metrics.json / divergence_curves.csv / divergence.png
```

**接口约定（代码主负责需要守住）：**

- `WorldModel.observe_seq(obs, actions)`：`obs` 为 `(B, T+1, 3, H, W)`，`actions` 为 `(B, T)`；
  `actions[:, t]` 表示从 `obs_t` 到 `obs_{t+1}` 的动作。
- `WorldModel.imagine_step(h, z, action, sample)`：动作使用环境编号 `0..3`，
  内部加 1 映射到 embedding；`0` 号 embedding 保留给 no-op。
- 配置键名一旦进入实验矩阵不要随意改，否则历史 `wm.pt` 无法复现。
- 输出 JSON/CSV 的字段名是下游画图与写作的契约，改动需同步通知。

---

## 5. 环境配置

### 5.1 依赖

- Python ≥ 3.10
- PyTorch ≥ 2.1（CPU 或 CUDA 均可）
- numpy / matplotlib / PyYAML

```powershell
# 在项目根目录执行
cd world_model_cf
py -m pip install -r requirements.txt
```

如果 `py` 不可用，把命令里的 `py` 换成 `python`。
GPU 机器请先按 PyTorch 官网命令安装对应 CUDA 版本的 wheel，再装其余依赖。

CARLA 线不复用上述 PyTorch 环境，改用独立的 Python 3.10 + JAX 环境，见 0.3。

### 5.2 硬件建议

| 场景 | 配置 | 说明 |
| --- | --- | --- |
| 冒烟 / 改代码 | CPU，`configs/smoke.yaml` | $32\times32$、16 条轨迹、20 步，几分钟内跑完 |
| 正式训练 | 单卡 GPU（≥ 8 GB） | $64\times64$、batch 16、序列长 20、3000 步 |
| 内存 | ≥ 8 GB | 64 条 96 步的 64×64 uint8 轨迹约占 76 MB |
| 磁盘 | ≥ 2 GB | checkpoints + 图表 + 多组 seed |

正式训练在纯 CPU 上不现实；CPU 只用于 smoke 与单元级调试。

### 5.3 训练前检查清单

- [ ] `py -c "import torch, yaml, matplotlib"` 无报错
- [ ] `configs/*.yaml` 中 `data.img_size` 能被 16 整除
- [ ] `data.n_actions = model.n_actions = env.N_ACTIONS = 4`
- [ ] `data.episode_len >= train.seq_len`，且 `eval.intervention_steps[-1] + eval.horizon <= eval_episode_len`
- [ ] 已跑通 smoke 三件套（训练 → 评估 → 出图）
- [ ] 确认 `device`（`auto` / `cpu` / `cuda`）
- [ ] 确定本次 `run_name` 与 `seed`，并记录到实验表
- [ ] 磁盘有空间，`outputs/` 不会被版本控制误提交

---

## 6. 运行流程

所有命令都在 `world_model_cf/` 目录下执行。本节是合成环境线；CARLA 线的命令见 0.4。

### 6.1 冒烟测试（必须最先跑）

```powershell
py -m wmcf.train_world_model --config configs/smoke.yaml
py -m wmcf.train_agent       --config configs/smoke.yaml --checkpoint outputs/runs/smoke/wm.pt
py -m wmcf.counterfactual    --config configs/smoke.yaml --checkpoint outputs/runs/smoke/wm.pt
```

预期：`outputs/runs/smoke/wm.pt`、`outputs/runs/smoke/ac.pt`、
`outputs/cf/smoke/divergence.png` 均生成，无异常。

### 6.2 正式训练

```powershell
py -m wmcf.train_world_model --config configs/default.yaml --set run_name=confounded_vanilla seed=0
```

训练日志看三件事：

1. `recon` 是否持续下降（合成环境最终应明显低于 0.02）；
2. `kl` 是否在 free bits 附近，而不是一路掉到 0（掉到 0 说明 posterior collapse）；
3. `val_loss` 是否跟随 `loss`，若明显背离说明过拟合或数据太少。

### 6.3 智能体（可选）

分歧分析只依赖世界模型。如果报告需要"Dreamer 智能体"部分，再跑：

```powershell
py -m wmcf.train_agent --config configs/default.yaml --checkpoint outputs/runs/confounded_vanilla/wm.pt
```

注意当前实现是 **REINFORCE + λ-return**，梯度不反传世界模型（世界模型冻结）。
这与 Dreamer 的 analytic gradient 有差异，写报告时需要说明。

### 6.4 反事实评估

```powershell
py -m wmcf.counterfactual --config configs/default.yaml --checkpoint outputs/runs/confounded_vanilla/wm.pt --set run_name=confounded_vanilla
```

关键输出：

- `metrics.json`：所有指标与曲线
- `divergence_curves.csv`：逐步均值/方差，供画图
- `divergence.png`：潜空间分歧、像素分歧、奖励分歧、反事实真值误差

---

## 7. 指标定义

设干预时刻为 $k$，预测步为 $i=0,\dots,H-1$，`z_dim = stoch * classes`。

| 指标 | 定义 | 读法 |
| --- | --- | --- |
| 潜空间分歧 $D_i$ | $\lVert p^{cf}(z_{k+i}) - p^{fact}(z_{k+i})\rVert_2 / \sqrt{d_z}$，$p$ 为 prior 概率向量 | 主曲线；越大说明动作改变越多 |
| 像素分歧 | $\mathrm{MSE}(\hat x^{cf}_{k+i}, \hat x^{fact}_{k+i})$ | 可解释性强，但受解码器质量影响 |
| 奖励分歧 | $\lvert \hat r^{cf}_{k+i} - \hat r^{fact}_{k+i}\rvert$ | 检查任务信号是否被动作影响 |
| 起始步 `onset_step` | 首个 $D_i > \max(10^{-4}, 2\cdot\text{noise})$ | 分歧何时变得显著 |
| 增长率 `growth_rate` | $\log D_i$ 对 $i$ 的线性斜率 | 正且适中 = 稳定放大；接近 0 = 饱和 |
| 噪声底 `noise_floor` | 固定事实动作、重采样初始后验 K 次，软轨迹的平均离散度 | 衡量初始状态不确定性；分歧需显著高于它 |
| `snr_over_noise_floor` | 平均分歧 / 噪声底 | 建议 > 2 |
| 反事实保真度 | $\mathrm{MSE}(\hat x^{cf}, x^{cf}_{true})$ | 越小越好，用真实环境仿真 |
| 方向一致性 | 同一动作的 $\Delta z$ 两两余弦均值 | 越接近 1 说明动作签名越稳定 |
| LOO 动作可分性 | 用其他样本的类中心解码动作，`accuracy` | 随机基线 = 0.25 |

两种干预模式：

- `one_step`：只改 $a_k$，后续动作保持事实动作 → 测"单步扰动的自我修正能力"
- `persistent`：$a_k,\dots,a_{k+H-1}$ 全部替换为 $a'$ → 测"持续干预下的长期分歧"

---

## 8. 实验矩阵

每组建议跑 `seed = 0, 1, 2`，报告均值 ± 标准差。

| run_name | confound | action_shuffle | inv_dyn | 作用 |
| --- | --- | --- | --- | --- |
| `confounded_vanilla` | true | 0.0 | 0.1 | 主实验：有线索 + 正常训练 |
| `confounded_shuffled` | true | 0.5 | 0.0 | 负对照：动作被打乱，模型应无视动作 |
| `unconfounded` | false | 0.0 | 0.1 | 去混淆：背景不再泄漏动作 |
| `confounded_no_inv` | true | 0.0 | 0.0 | 消融：去掉逆动力学正则 |
| `confounded_inv_strong` | true | 0.0 | 0.5 | 消融：加强逆动力学 |

命令示例：

```powershell
py -m wmcf.train_world_model --config configs/default.yaml --set run_name=confounded_shuffled train.action_shuffle=0.5 train.inverse_dynamics_coef=0.0 seed=0
py -m wmcf.train_world_model --config configs/default.yaml --set run_name=unconfounded data.confound=false seed=0
py -m wmcf.train_world_model --config configs/default.yaml --set run_name=confounded_no_inv train.inverse_dynamics_coef=0.0 seed=0
py -m wmcf.train_world_model --config configs/default.yaml --set run_name=confounded_inv_strong train.inverse_dynamics_coef=0.5 seed=0
```

评估同样要传 `run_name`：

```powershell
py -m wmcf.counterfactual --config configs/default.yaml --checkpoint outputs/runs/confounded_shuffled/wm.pt --set run_name=confounded_shuffled train.action_shuffle=0.5 train.inverse_dynamics_coef=0.0 seed=0
```

### 8.1 结果解读模板

| 现象 | 解释 | 下一步 |
| --- | --- | --- |
| `confounded_vanilla` 分歧小、CF 误差大 | 模型走背景线索捷径，支持 H1 | 对比 `unconfounded` |
| `unconfounded` 分歧大、CF 误差小 | 模型被迫使用动作，支持 H2 | 检查方向一致性 |
| `confounded_shuffled` 分歧≈噪声底 | 指标有效，负对照成立 | 可写入方法验证 |
| 两组都分歧≈噪声底 | 模型没学会动力学或解码器太差 | 先看 recon，再调容量/步数 |
| 分歧爆炸、CF 误差也爆炸 | 潜空间数值不稳定 | 降低 lr、检查 KL 与梯度裁剪 |

---

## 9. 关键配置说明

| 键 | 默认 | 含义 / 建议 |
| --- | --- | --- |
| `data.img_size` | 64 | 必须被 16 整除；smoke 用 32 |
| `data.n_episodes` | 64 | 轨迹数；过少易过拟合 |
| `data.episode_len` | 96 | 需 ≥ `seq_len` |
| `data.confound` | true | 是否开启背景动作线索（核心自变量） |
| `data.tint_strength` | 0.45 | 线索强度；太弱捷径学不到，太强线索盖过球体 |
| `data.behavior_noise` | 0.15 | 行为策略随机性；影响动作分布覆盖 |
| `data.eval_confound` | true | 评估是否保留线索；一般与训练一致 |
| `model.hidden_dim` | 128 | RSSM 确定性状态维度 |
| `model.stoch` × `model.classes` | 32 × 32 | 离散潜变量组数 × 类别数；显存不足时降到 16×16 |
| `model.kl_free_bits` | 1.0 | 每类 KL 下限，防 posterior collapse |
| `model.kl_balance` | 0.8 | 动力学/表示 KL 的平衡系数 |
| `train.batch_size` | 16 | 显存不足时降为 8 |
| `train.seq_len` | 20 | 训练窗口长度；需 ≤ episode 长度 |
| `train.steps` | 3000 | 正式训练步数；以 recon 收敛为准 |
| `train.action_shuffle` | 0.0 | 负对照开关，>0 时按样本打乱动作 |
| `train.inverse_dynamics_coef` | 0.1 | 逆动力学正则强度；0 表示关闭 |
| `eval.horizon` | 12 | 想象步数；越大越能看长期分歧 |
| `eval.intervention_steps` | [0,2,4] | 干预时刻；需满足 `k + horizon ≤ 轨迹长度` |
| `eval.n_prior_samples` | 16 | 噪声底采样次数；建议 ≥ 8 |

---

## 10. 阶段计划与验收标准

| 阶段 | 交付物 | 验收标准 |
| --- | --- | --- |
| M0 打通流程 | smoke 三件套产物 | 训练/评估/出图均成功，无 NaN |
| M1 世界模型可用 | `wm.pt` + 学习曲线 | `recon` 收敛，`val_loss` 跟随，KL 不塌缩 |
| M2 对照实验 | 4–5 组 × 3 seed 的 `wm.pt` 与日志 | 每组配置快照、seed、日志齐全 |
| M3 分歧分析 | 每组的 `metrics.json` / `divergence.png` | 噪声底、onset、growth、CF 误差均产出 |
| M4 结论与图表 | 汇总表 + 论文级图 | 能回答 H1–H4，负对照成立 |
| M5 可选扩展 | actor-critic 或 Transformer 骨干 | 不影响主结论，作为附录 |

**M1 的硬门槛**：若 `recon` 不下降，后面所有分歧指标都没有意义。先修世界模型，
不要急着跑干预实验。

---

## 11. 风险与排查

| 风险 | 症状 | 处理 |
| --- | --- | --- |
| Posterior collapse | `kl` 长期为 0，重建模糊 | 检查 free bits、提高 `kl_balance`、降低 lr |
| 解码器太弱 | `recon` 高，分歧曲线噪声大 | 增大 `width`/`hidden_dim`，增加步数 |
| 线索过强 | 模型完全不看动作 | 降低 `tint_strength`，或改用 `unconfounded` 对照 |
| 线索太弱 | 两组结果无差异 | 提高 `tint_strength`，检查背景是否真的编码动作 |
| 分歧饱和 | `growth_rate≈0`，曲线贴顶 | 用 onset/CF 保真度补充解释；归一化潜距离有上界是正常的 |
| 随机性淹没信号 | `snr < 1` | 检查先验是否过软；增加 `n_prior_samples`；用 soft 概率分歧 |
| 动作编号错位 | 训练正常但干预无效 | 核对 `actions[:, t]` 对应 `obs_t → obs_{t+1}` 的约定 |
| 序列太短 | 采样报 "no episode is long enough" | 增大 `episode_len` 或减小 `seq_len` |

---

## 12. 团队分工建议（代码主负责视角）

| 角色 | 职责 | 与代码的接口 |
| --- | --- | --- |
| 代码主负责（你） | 架构、训练/评估流水线、指标实现、code review、结果可复现性 | 守住配置键与输出 schema |
| 实验执行 | 按矩阵跑 seed、记录异常、汇总日志 | 只改 `--set`，不改代码；结果放 `outputs/runs/<run_name>/` |
| 分析写作 | 画图、写结论、对照文献 | 消费 `metrics.json` / `divergence_curves.csv` |

协作约定：

1. 任何影响结果的改动（损失、指标、数据生成）都要在 PR 描述里写清楚，并重跑 smoke。
2. 每组实验必须带 `run_name`、`seed`、`config.yaml` 快照，三者缺一不可。
3. `metrics.json` 的字段只增不改；新增字段在 README 第 7 节同步。
4. 结论以多种子均值为准，单次运行的漂亮曲线不作为证据。

---

## 13. 待确认问题

1. 报告是否需要智能体性能（回报曲线）？若需要，actor-critic 要从 REINFORCE 升级为
   反传世界模型的版本吗？
2. 是否需要第二个环境（例如无线索的 DMControl 小任务）来验证结论的外部有效性？
   当前只做合成环境，成本最低。
3. 线索设计是否要更隐蔽（例如线索只以一定概率出现），以避免"过于人为"的批评？
4. 是否把"动作线索"换成"奖励线索"或"未来帧泄漏"，做一组稳健性实验？
5. 团队机器的 GPU 型号与显存是多少？这决定 `stoch × classes` 与 batch 的上限。

---

## 14. 快速命令备忘

```powershell
# 冒烟
py -m wmcf.train_world_model --config configs/smoke.yaml
py -m wmcf.train_agent       --config configs/smoke.yaml --checkpoint outputs/runs/smoke/wm.pt
py -m wmcf.counterfactual    --config configs/smoke.yaml --checkpoint outputs/runs/smoke/wm.pt

# 主实验
py -m wmcf.train_world_model --config configs/default.yaml --set run_name=confounded_vanilla seed=0
py -m wmcf.counterfactual    --config configs/default.yaml --checkpoint outputs/runs/confounded_vanilla/wm.pt

# 去混淆对照
py -m wmcf.train_world_model --config configs/default.yaml --set run_name=unconfounded data.confound=false seed=0
py -m wmcf.counterfactual    --config configs/default.yaml --checkpoint outputs/runs/unconfounded/wm.pt --set run_name=unconfounded data.confound=false

# 负对照
py -m wmcf.train_world_model --config configs/default.yaml --set run_name=confounded_shuffled train.action_shuffle=0.5 train.inverse_dynamics_coef=0.0 seed=0
py -m wmcf.counterfactual    --config configs/default.yaml --checkpoint outputs/runs/confounded_shuffled/wm.pt --set run_name=confounded_shuffled
```
