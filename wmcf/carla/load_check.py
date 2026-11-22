"""不依赖 CARLA 的 CarDreamer checkpoint 加载自检（load_check）。

目的：在没有 CARLA 仿真器、没有 car_dreamer/gym 的机器上，只凭 JAX + optax
验证 ``four_lane.ckpt`` 能被 DreamerV3 正确加载，并对随机合成的 BEV 观测前向
若干步，确认 policy 能输出合法的 15 维 one-hot 动作。

这不是真实驾驶演示（没有 CARLA 就没有真实观测与车辆动力学），而是
「权重完好 + 模型构建 + 前向推理」的冒烟测试。真实四车道闭环请用 run_demo.py。

用法（Python 3.11 + jax[cpu]==0.4.34 + optax==0.2.3 + ruamel.yaml + numpy<2）：
  python -m wmcf.carla.load_check \
      --cardreamer D:/repos/CarDreamer \
      --checkpoint checkpoints/four_lane.ckpt --steps 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cardreamer", required=True, help="CarDreamer 仓库根目录")
    parser.add_argument("--checkpoint", required=True, help="four_lane.ckpt 路径")
    parser.add_argument("--task", default="carla_four_lane")
    parser.add_argument("--steps", type=int, default=5, help="前向步数")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(args.cardreamer).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "dreamerv3"))

    # embodied 的 replay 链在模块顶层 `import tensorflow` / `import reverb`，
    # 但这些名字只在 replay 的函数体内使用，load_check 不会触达。
    # 对未安装的可选重依赖做占位，避免为一次权重自检安装数百 MB 的 TF。
    from unittest.mock import MagicMock

    for _name in ("tensorflow", "reverb"):
        try:
            __import__(_name)
        except ImportError:
            sys.modules[_name] = MagicMock(name=_name)
            print(f"[stub] optional dependency '{_name}' is stubbed for this check")

    import numpy as np
    import ruamel.yaml as yaml

    import embodied
    import dreamerv3

    # --- 1. 配置：defaults + small + 任务覆盖（与官方 eval.py 同序） ---
    model_configs = yaml.YAML(typ="safe").load((root / "dreamerv3" / "dreamerv3.yaml").read_text())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})
    tasks = yaml.YAML(typ="safe").load((root / "car_dreamer" / "configs" / "tasks.yaml").read_text())
    dv3_over = tasks[args.task]["dreamerv3"]
    config = config.update({"dreamerv3": dv3_over})
    dv3 = config.dreamerv3
    # 自检在 CPU 上跑；JAXAgent 从 config.jax.platform 读取后端。
    dv3 = dv3.update({"jax": {"platform": "cpu"}})
    print(f"[config] deter={dv3.rssm.deter} enc.cnn_keys={dv3.encoder.cnn_keys}")

    # --- 2. 观测/动作空间（encoder 只用 birdeye_wpt；动作是 15 维 one-hot） ---
    obs_space = {
        "birdeye_wpt": embodied.Space(np.uint8, (128, 128, 3)),
        "is_first": embodied.Space(bool),
        "is_terminal": embodied.Space(bool),
        "reward": embodied.Space(np.float32),
    }
    action_space = embodied.Space(np.float32, (15,), 0, 1)
    action_space._discrete = True  # 与 OneHotAction wrapper 一致
    act_space = {"action": action_space, "reset": embodied.Space(bool)}

    step = embodied.Counter()
    agent = dreamerv3.Agent(obs_space, act_space, step, dv3)

    # --- 3. 加载 checkpoint ---
    ckpt = embodied.Checkpoint()
    ckpt.agent = agent
    print(f"[load] {args.checkpoint}")
    ckpt.load(args.checkpoint, keys=["agent"])
    print("[load] OK")

    # --- 4. 随机观测前向 N 步 ---
    rng = np.random.RandomState(args.seed)
    batch = 1
    state = None  # JAXAgent.policy 会在 state=None 时初始化 latent/action 状态。
    print(f"[policy] running {args.steps} forward steps on random BEV observations ...")
    for t in range(args.steps):
        obs = {
            "birdeye_wpt": rng.randint(0, 255, (batch, 128, 128, 3), dtype=np.uint8),
            "is_first": np.array([t == 0]),
            "is_terminal": np.array([False]),
            "reward": np.zeros(batch, dtype=np.float32),
        }
        outs, state = agent.policy(obs, state, mode="eval")
        action = np.asarray(outs["action"])[0]
        assert action.shape == (15,), f"unexpected action shape {action.shape}"
        assert np.isclose(action.sum(), 1.0, atol=1e-4), f"not one-hot: sum={action.sum()}"
        print(f"  step {t}: argmax_action={int(action.argmax())} sum={action.sum():.3f}")

    print("[done] checkpoint loads and policy produces valid 15-dim one-hot actions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
