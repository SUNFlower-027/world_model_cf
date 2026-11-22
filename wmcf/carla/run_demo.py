"""CarDreamer 四车道（carla_four_lane）闭环推理演示。

严格遵循官方 ``dreamerv3/eval.py`` 的流程：构建 DreamerV3(JAX) agent、加载
预训练 checkpoint、在 CARLA 的 ``carla_four_lane`` 任务里闭环 rollout，并保存
KPI（metrics.jsonl）与可选的前视视频。

运行前提（缺一不可）：
  1. ``cardreamer`` conda 环境（Python 3.10 + JAX 0.4.34 + car_dreamer + carla）。
  2. CARLA 0.9.15 服务器已启动，默认端口 2000：
       Windows:  CarlaUE4.exe -carla-port=2000 -quality-level=Low
       Linux:    ./CarlaUE4.sh -carla-port=0 -quality-level=Low
  3. 已下载 ``four_lane.ckpt``（见 wmcf.carla.checkpoints --download）。

示例：
  python -m wmcf.carla.run_demo \
      --cardreamer D:/repos/CarDreamer \
      --checkpoint checkpoints/four_lane.ckpt \
      --task carla_four_lane --steps 2000 --save-video

注意：本脚本需要真实 CARLA 仿真器，无法在无 CARLA 的机器上运行；它只做
推理（eval），不进行任何训练。原生 Windows 的 JAX 0.4.34 使用 CPU 后端，
运行时需加 ``--platform cpu``（CARLA 渲染仍可使用 GPU）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cardreamer", required=True, help="CarDreamer 仓库根目录")
    parser.add_argument("--checkpoint", required=True, help="four_lane.ckpt 路径")
    parser.add_argument("--carla-root", default=None, help="CARLA 0.9.15 根目录（可选）")
    parser.add_argument("--task", default="carla_four_lane", help="CarDreamer 任务名")
    parser.add_argument("--port", type=int, default=2000, help="CARLA 服务器端口")
    parser.add_argument("--steps", type=int, default=2000, help="总评估步数")
    parser.add_argument("--logdir", default="outputs/carla_demo", help="日志/视频输出目录")
    parser.add_argument("--save-video", action="store_true", help="导出 camera 视角 mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--platform",
        choices=("cpu", "gpu"),
        default=None,
        help="JAX 后端；原生 Windows 的 JAX 0.4.34 请使用 cpu",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # --- 1. 路径准备（必须在 import carla / car_dreamer / dreamerv3 之前） ---
    cardreamer_root = Path(args.cardreamer).resolve()
    if not (cardreamer_root / "dreamerv3").is_dir():
        raise FileNotFoundError(f"{cardreamer_root} 下没有 dreamerv3/，--cardreamer 路径有误")
    # CarDreamer 把 embodied / dreamerv3 作为顶层包，需要把 dreamerv3 目录加入 sys.path
    sys.path.insert(0, str(cardreamer_root))
    sys.path.insert(0, str(cardreamer_root / "dreamerv3"))
    if args.carla_root:
        # 等价于官方 export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla":...
        sys.path.insert(0, str(Path(args.carla_root) / "PythonAPI" / "carla"))

    import numpy as np
    import ruamel.yaml as yaml

    import embodied
    import car_dreamer
    import dreamerv3
    from embodied.envs import from_gym

    # --- 2. 配置：defaults + small + 任务 env_config（与官方 eval.py 一致） ---
    yaml_path = cardreamer_root / "dreamerv3" / "dreamerv3.yaml"
    model_configs = yaml.YAML(typ="safe").load(yaml_path.read_text())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})

    # 端口等覆盖走 env.world.* 嵌套键（见 car_dreamer/configs/common.yaml）
    task_argv = [f"--env.world.carla_port={args.port}"]
    env, env_config = car_dreamer.create_task(args.task, task_argv)
    config = config.update(env_config)
    config = config.update({"dreamerv3.seed": args.seed, "dreamerv3.logdir": args.logdir})
    if args.platform:
        config = config.update({"dreamerv3.jax.platform": args.platform})
    dv3 = config.dreamerv3

    # --- 3. 环境包装：FromGym -> OneHot/TimeLimit -> BatchEnv ---
    def wrap_env(env, config):
        wargs = config.wrapper
        env = embodied.wrappers.InfoWrapper(env)
        for name, space in env.act_space.items():
            if name == "reset":
                continue
            elif space.discrete:
                env = embodied.wrappers.OneHotAction(env, name)
            elif wargs.discretize:
                env = embodied.wrappers.DiscretizeAction(env, name, wargs.discretize)
            else:
                env = embodied.wrappers.NormalizeAction(env, name)
        env = embodied.wrappers.ExpandScalars(env)
        if wargs.length:
            env = embodied.wrappers.TimeLimit(env, wargs.length, wargs.reset)
        for name, space in env.act_space.items():
            if not space.discrete:
                env = embodied.wrappers.ClipAction(env, name)
        return env

    env = from_gym.FromGym(env)
    env = wrap_env(env, dv3)
    env = embodied.BatchEnv([env], parallel=False)

    # --- 4. 构建 agent 并加载 checkpoint ---
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
        ],
    )

    dv3 = dv3.update(
        {
            "run.log_keys_sum": "(travel_distance|destination_reached|out_of_lane|time_exceeded|is_collision|timesteps)",
            "run.log_keys_mean": "(travel_distance|ttc|speed_norm|wpt_dis)",
            "run.log_keys_max": "(travel_distance|ttc|speed_norm|wpt_dis)",
            "run.steps": float(args.steps),
        }
    )

    print("Observation space:", env.obs_space)
    print("Action space:", env.act_space)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, dv3)

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint.load(args.checkpoint, keys=["agent"])

    run_args = embodied.Config(
        **dv3.run,
        logdir=dv3.logdir,
        batch_steps=dv3.batch_size * dv3.batch_length,
    )

    # --- 5. 闭环 rollout ---
    video_dir = Path(args.logdir) / "videos"
    if args.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    episode_count = [0]

    def per_episode(ep, ep_info, worker):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        logger.add({"length": length, "score": score}, prefix="episode")
        episode_count[0] += 1
        print(f"[episode {episode_count[0]}] steps={length} return={score:.1f}")
        if args.save_video and "camera" in ep:
            try:
                import imageio.v2 as imageio

                frames = [f for f in ep["camera"]]
                # 相机首帧到达前会插入全黑占位帧，去掉片头黑帧
                while frames and not np.asarray(frames[0]).any():
                    frames.pop(0)
                out = video_dir / f"episode_{episode_count[0]:03d}.mp4"
                imageio.mimsave(out, frames, fps=10)
                print(f"[video] {out}")
            except Exception as exc:  # 视频是可选产物，失败不影响 KPI
                print(f"[warn] 视频保存失败：{exc}")
        logger.write()

    driver = embodied.Driver(env)
    driver.on_episode(per_episode)
    driver.on_step(lambda tran, info, _: step.increment())

    policy = lambda *a: agent.policy(*a, mode="eval")
    print("Start closed-loop rollout ...")
    while step < run_args.steps:
        driver(policy, steps=100)
    logger.write()
    print(f"Done. KPI -> {Path(args.logdir) / 'metrics.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
