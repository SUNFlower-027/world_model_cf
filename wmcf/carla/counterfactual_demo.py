"""在 CARLA 四车道场景中对 CarDreamer 世界模型做反事实动作干预。

对应 ``docs/autonomous_driving.md`` 的 Step 3（5.4 节）：先让预训练策略在
``carla_four_lane`` 里真实闭环若干 episode，拿到 KPI；再取最后一个 episode 的
``birdeye_wpt`` 与动作序列，逐时刻用 RSSM 后验编码真实观测，固定该后验后
分别以 ``do(刹车/保持/加速/左转/右转)`` 在潜空间里想象 H 步，与"按真实动作
想象的轨迹"比较分歧，并与"同动作、重采样潜变量"的噪声底对照。

输出：
  outputs/<logdir>/kpi.csv / kpi_table.md     闭环 KPI 表
  outputs/<logdir>/divergence.png / .json     干预分歧曲线与指标
  outputs/<logdir>/videos/cf_grid.mp4         真实 BEV / 事实想象 / 各干预分支对比
  outputs/<logdir>/videos/cf_contact_sheet.png

说明：这里只做推理与干预，不训练任何模型。原生 Windows 的 JAX 请用
``--platform cpu``（CARLA 渲染仍可用 GPU）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# 动作索引 -> (加速度, 转向)：acc = discrete_acc[a // 5]，steer = -discrete_steer[a % 5]
# discrete_acc = [-2, 0, 2]，discrete_steer = [-0.6, -0.2, 0, 0.2, 0.6]
# CARLA 中正 steer 为左转，因此 steer 索引 0 为左转、4 为右转。
INTERVENTIONS = {
    "do(刹车)": 2,
    "do(保持)": 7,
    "do(加速)": 12,
    "do(左转)": 5,
    "do(右转)": 9,
}

BRANCH_VIDEOS = {
    "real": ("branch_real.mp4", "real observation"),
    "factual": ("branch_factual.mp4", "imagination (factual)"),
    "do(刹车)": ("branch_do_brake.mp4", "do(brake) imagination"),
    "do(保持)": ("branch_do_hold.mp4", "do(hold) imagination"),
    "do(加速)": ("branch_do_accelerate.mp4", "do(accelerate) imagination"),
    "do(左转)": ("branch_do_left.mp4", "do(left) imagination"),
    "do(右转)": ("branch_do_right.mp4", "do(right) imagination"),
    "noise": ("branch_noise_floor.mp4", "noise floor imagination"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cardreamer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--carla-root", default=None)
    parser.add_argument("--task", default="carla_four_lane")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kpi-episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=20, help="想象步数")
    parser.add_argument("--anchors", type=int, default=4, help="每条 episode 的干预起点个数")
    parser.add_argument("--fps", type=int, default=5, help="对比视频帧率")
    parser.add_argument("--logdir", default="outputs/carla_cf")
    return parser.parse_args()


def _build_stack(args):
    """构建环境与 agent，并加载 checkpoint（与 run_demo.py 保持一致）。"""
    import numpy as np
    import ruamel.yaml as yaml

    import embodied
    import car_dreamer
    import dreamerv3
    from embodied.envs import from_gym

    cardreamer_root = Path(args.cardreamer).resolve()
    yaml_path = cardreamer_root / "dreamerv3" / "dreamerv3.yaml"
    model_configs = yaml.YAML(typ="safe").load(yaml_path.read_text())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})

    task_argv = [f"--env.world.carla_port={args.port}"]
    env, env_config = car_dreamer.create_task(args.task, task_argv)
    config = config.update(env_config)
    config = config.update({"dreamerv3.seed": args.seed, "dreamerv3.logdir": args.logdir})
    if args.platform:
        config = config.update({"dreamerv3.jax.platform": args.platform})
    dv3 = config.dreamerv3

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

    step = embodied.Counter()
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, dv3)
    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint.load(args.checkpoint, keys=["agent"])
    return env, agent, dv3


# --------------------------------------------------------------------- 指标
def _flat(info: dict, key: str):
    import numpy as np

    if key not in info:
        return None
    return np.asarray(info[key], dtype=np.float64).reshape(len(info[key]), -1)


def _episode_kpis(info: dict) -> dict:
    import numpy as np

    def rate(key):
        arr = _flat(info, key)
        return None if arr is None else int(bool(arr.max()))

    def mean(key, cap=None):
        arr = _flat(info, key)
        if arr is None:
            return None
        values = arr[:, 0]
        if cap is not None:
            values = values[values < cap]
        return float(values.mean()) if len(values) else None

    def total(key):
        arr = _flat(info, key)
        return None if arr is None else float(arr[:, 0].sum())

    def max_of(key):
        arr = _flat(info, key)
        return None if arr is None else float(arr[:, 0].max())

    return {
        "collision": rate("is_collision"),
        "out_of_lane": rate("out_of_lane"),
        "destination": rate("destination_reached"),
        "time_exceeded": rate("time_exceeded"),
        "travel_distance_m": total("travel_distance"),
        "mean_speed_ms": mean("speed_norm"),
        "mean_wpt_dis_m": mean("wpt_dis"),
        "max_wpt_dis_m": max_of("wpt_dis"),
    }


def _write_kpi(rows: list[dict], logdir: Path) -> None:
    import numpy as np

    fields = [
        "episode", "seed", "length", "score", "collision", "out_of_lane",
        "destination", "time_exceeded", "travel_distance_m", "mean_speed_ms",
        "mean_wpt_dis_m", "max_wpt_dis_m",
    ]
    with (logdir / "kpi.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value):
        return "-" if value is None else f"{value:.4g}" if isinstance(value, float) else str(value)

    lines = [
        "# CARLA 四车道闭环 KPI",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[k]) for k in fields) + " |")

    def agg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return None if not vals else (float(np.mean(vals)), float(np.std(vals)))

    lines += ["", "## 跨 episode 汇总（均值 ± 标准差）", "", "| 指标 | 值 |", "| --- | --- |"]
    for key in ("length", "score", "travel_distance_m", "mean_speed_ms",
                "mean_wpt_dis_m", "max_wpt_dis_m"):
        stat = agg(key)
        if stat:
            lines.append(f"| {key} | {stat[0]:.3f} ± {stat[1]:.3f} |")
    for key in ("collision", "out_of_lane", "destination", "time_exceeded"):
        vals = [r[key] for r in rows if r[key] is not None]
        if vals:
            lines.append(f"| {key} 率 | {np.mean(vals):.2f} |")
    (logdir / "kpi_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[kpi] {logdir / 'kpi_table.md'}")


# ------------------------------------------------------- 世界模型内部 rollout
def _make_fns(agent, jax, jnp, nj):
    wm = agent.agent.wm
    rssm = wm.rssm
    decoder = wm.heads["decoder"]

    # ninjax 的 pure 函数通过 context 管理参数，不把 varibs/rng 作为形参。
    def initial_fn():
        return rssm.initial(1)

    def obs_fn(state, prev_action, image, is_first):
        embed = wm.encoder({"birdeye_wpt": image})
        post, _ = rssm.obs_step(state, prev_action, embed, is_first)
        return post

    def imagine_fn(state, action):
        prior = rssm.img_step(state, action)
        frame = decoder(prior)["birdeye_wpt"].mode()
        return {"state": prior, "frame": frame, "logit": prior["logit"]}

    return (
        nj.jit(nj.pure(initial_fn)),
        nj.jit(nj.pure(obs_fn)),
        nj.jit(nj.pure(imagine_fn)),
    )


def _js_divergence(p, q):
    import numpy as np

    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    kl_p = np.sum(p * (np.log(p) - np.log(m)), axis=-1)
    kl_q = np.sum(q * (np.log(q) - np.log(m)), axis=-1)
    return float(0.5 * kl_p.mean() + 0.5 * kl_q.mean())


def _softmax(logits):
    import numpy as np

    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _analyse(args, agent, ep, logdir: Path):
    import imageio.v2 as imageio
    import jax
    import jax.numpy as jnp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from dreamerv3 import ninjax as nj

    # JAXAgent 把 transfer guard 设成了 disallow，内部推理结束后要建 PRNGKey，
    # 这里恢复为 allow，避免隐式 host-to-device 传输被拒。
    jax.config.update("jax_transfer_guard", "allow")

    jit_initial, jit_obs, jit_imagine = _make_fns(agent, jax, jnp, nj)
    varibs = agent.varibs

    images = np.asarray(ep["birdeye_wpt"], dtype=np.float32) / 255.0
    actions = np.asarray(ep["action"], dtype=np.float32)
    total = len(actions) - 1
    if total <= args.horizon + 10:
        print(f"[warn] episode 太短（{total} 步），跳过反事实分析")
        return None

    # 用真实观测逐时刻更新后验：prev_action[t] 是产生 obs[t] 的动作
    prev_actions = np.zeros((1, total + 1, actions.shape[-1]), np.float32)
    prev_actions[0, 1:] = actions[:total]
    is_first = np.zeros((1, total + 1), np.float32)
    is_first[0, 0] = 1.0

    key = jax.random.PRNGKey(args.seed)
    state, varibs = jit_initial(varibs, key)
    posts = []
    for t in range(total + 1):
        obs = jax.device_put(images[t][None])
        act = jax.device_put(prev_actions[:, t])
        first = jax.device_put(is_first[:, t])
        state, varibs = jit_obs(varibs, key, state, act, obs, first)
        posts.append(state)
    print(f"[cf] 已编码 {total + 1} 个真实观测后验")

    # 避开起步和终点附近，取 episode 中段 10%~90% 的均匀采样作为干预起点
    lo = max(15, int(0.10 * total))
    hi = min(total - args.horizon - 1, int(0.90 * total))
    if hi < lo:
        lo, hi = 15, max(15, total - args.horizon - 1)
    anchors = sorted({int(x) for x in np.linspace(lo, hi, args.anchors) if lo <= x <= hi})
    print(f"[cf] 干预起点: {anchors}, horizon={args.horizon}")

    def rollout(start, act_seq, rng_key):
        state = start
        key = rng_key
        logits, frames = [], []
        for action in act_seq:
            out, _ = jit_imagine(varibs, key, state, jax.device_put(action[None]))
            state = out["state"]
            logits.append(np.asarray(jax.device_get(out["logit"]))[0])
            frames.append(np.asarray(jax.device_get(out["frame"]))[0])
            key = jax.random.fold_in(key, 1)
        return np.stack(logits), np.clip(np.stack(frames), 0.0, 1.0)

    def onehot(index):
        vec = np.zeros(actions.shape[-1], np.float32)
        vec[index] = 1.0
        return vec

    curves = {"factual": [], "noise": []}
    for name in INTERVENTIONS:
        curves[name] = []
    video_branch = None
    video_anchor = None

    for anchor in anchors:
        start = posts[anchor]
        factual_acts = actions[anchor:anchor + args.horizon]
        f_logits, f_frames = rollout(start, factual_acts, jax.random.fold_in(key, 101 + anchor))
        n_logits, n_frames = rollout(
            start, factual_acts, jax.random.fold_in(key, 10007 + anchor)
        )
        instant = {"factual": f_logits, "noise": n_logits}
        branch_frames = {"real": images[anchor + 1:anchor + 1 + args.horizon],
                         "factual": f_frames, "noise": n_frames}
        for name, index in INTERVENTIONS.items():
            acts = np.stack([onehot(index)] * args.horizon)
            c_logits, c_frames = rollout(start, acts, jax.random.fold_in(key, 7919 + anchor))
            instant[name] = c_logits
            branch_frames[name] = c_frames

        f_probs = np.stack([_softmax(x) for x in f_logits])
        n_probs = np.stack([_softmax(x) for x in n_logits])
        curves["factual"].append([0.0] * args.horizon)
        curves["noise"].append([_js_divergence(n_probs[t], f_probs[t]) for t in range(args.horizon)])
        for name in INTERVENTIONS:
            probs = np.stack([_softmax(x) for x in instant[name]])
            curves[name].append([_js_divergence(probs[t], f_probs[t]) for t in range(args.horizon)])

        if video_branch is None:
            video_branch = branch_frames
            video_anchor = anchor

    summary = {}
    for name, seq in curves.items():
        arr = np.asarray(seq, dtype=np.float64)
        summary[name] = {"mean": arr.mean(0).tolist(), "std": arr.std(0).tolist()}

    noise_mean = np.asarray(summary["noise"]["mean"])
    noise_std = np.asarray(summary["noise"]["std"])
    threshold = float((noise_mean + 2.0 * noise_std).max())
    metrics = {"horizon": args.horizon, "anchors": anchors, "threshold": threshold,
               "curves": summary, "onset": {}, "growth": {}}
    for name in INTERVENTIONS:
        curve = np.asarray(summary[name]["mean"])
        hits = np.where(curve > threshold)[0]
        metrics["onset"][name] = int(hits[0] + 1) if len(hits) else -1
        x = np.arange(len(curve), dtype=np.float64)
        metrics["growth"][name] = float(np.polyfit(x, np.log(np.maximum(curve, 1e-6)), 1)[0])

    (logdir / "divergence.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 分歧曲线 ----
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    x = np.arange(1, args.horizon + 1)
    for name in INTERVENTIONS:
        mean = np.asarray(summary[name]["mean"])
        std = np.asarray(summary[name]["std"])
        ax.plot(x, mean, label=name, lw=1.8)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    noise = np.asarray(summary["noise"]["mean"])
    ax.plot(x, noise, "--", color="0.4", lw=1.2, label="噪声底（同动作重采样）")
    ax.axhline(threshold, color="0.6", ls=":", lw=1, label="显著性阈值")
    ax.set_xlabel("干预后想象步数")
    ax.set_ylabel("潜空间 JS 分歧（nats）")
    ax.set_title("CARLA 四车道：do(a') 反事实想象轨迹分歧")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(logdir / "divergence.png")
    plt.close(fig)
    print(f"[cf] {logdir / 'divergence.png'}")

    # ---- 对比视频 ----
    if video_branch:
        _save_grid_video(video_branch, logdir, args.horizon, args.fps)
        camera = np.asarray(ep.get("camera", []), dtype=np.uint8)
        if len(camera):
            _save_branch_videos(
                camera, video_branch, video_anchor, logdir, args.horizon, args.fps
            )

    return metrics


def _save_branch_videos(camera, branches: dict, anchor: int, logdir: Path,
                        horizon: int, fps: int = 5) -> None:
    """每个分支一段视频：左侧真实相机参考，右侧该分支的世界模型想象 BEV。"""
    import cv2
    import imageio.v2 as imageio
    import numpy as np

    size, bar, foot = 512, 24, 22
    video_dir = logdir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    camera_slice = np.asarray(
        camera[anchor + 1:anchor + 1 + horizon], dtype=np.uint8
    )
    packed = {"camera": camera_slice}

    for name, frames in branches.items():
        if name not in BRANCH_VIDEOS:
            continue
        filename, title = BRANCH_VIDEOS[name]
        ascii_name = filename[len("branch_"):-len(".mp4")].replace("_", " ")
        frames = np.asarray(frames)
        packed[ascii_name.replace(" ", "_")] = frames
        if np.issubdtype(frames.dtype, np.floating):
            frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
        out = []
        for t in range(horizon):
            canvas = np.zeros((size + bar + foot, size * 2, 3), np.uint8)
            cam = camera[min(anchor + 1 + t, len(camera) - 1)]
            bev = frames[min(t, len(frames) - 1)]
            canvas[bar:bar + size, :size] = cv2.resize(
                cam, (size, size), interpolation=cv2.INTER_AREA)
            canvas[bar:bar + size, size:] = cv2.resize(
                bev, (size, size), interpolation=cv2.INTER_NEAREST)
            cv2.putText(canvas, "camera (real reference)", (6, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{title} (BEV)", (size + 6, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"left: factual real camera | right: world-model {ascii_name} | step {t + 1}/{horizon}",
                        (6, size + bar + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 200, 200), 1, cv2.LINE_AA)
            out.append(canvas)
        path = video_dir / filename
        imageio.mimsave(path, out, fps=fps)
        print(f"[cf] {path}")

    np.savez_compressed(logdir / "branches.npz", **packed)


def _save_grid_video(branches: dict, logdir: Path, horizon: int, fps: int = 5) -> None:
    import cv2
    import imageio.v2 as imageio
    import numpy as np

    labels = {
        "real": "real observation",
        "factual": "imagination (factual)",
        "do(刹车)": "do(brake)",
        "do(保持)": "do(hold)",
        "do(加速)": "do(accelerate)",
        "do(左转)": "do(left)",
        "do(右转)": "do(right)",
        "noise": "noise floor",
    }
    order = ["real", "factual"] + [n for n in INTERVENTIONS if n in branches]
    if "noise" in branches:
        order.append("noise")
    tiles = []
    for name in order:
        frames = np.asarray(branches[name])
        if frames.dtype != np.uint8:
            frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
        tiles.append((name, frames))

    scale = 2
    tile_h = 128 * scale
    label_h = 22
    cols, rows = 3, int(np.ceil(len(tiles) / 3))
    frame_h = rows * (tile_h + label_h)
    frame_w = cols * tile_h

    out = []
    for t in range(horizon):
        canvas = np.zeros((frame_h, frame_w, 3), np.uint8)
        for i, (name, frames) in enumerate(tiles):
            r, c = divmod(i, cols)
            img = frames[min(t, len(frames) - 1)]
            img = cv2.resize(img, (tile_h, tile_h), interpolation=cv2.INTER_NEAREST)
            y = r * (tile_h + label_h) + label_h
            canvas[y:y + tile_h, c * tile_h:(c + 1) * tile_h] = img
            cv2.putText(
                canvas, labels.get(name, name), (c * tile_h + 4, r * (tile_h + label_h) + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )
        cv2.putText(
            canvas, f"step {t + 1}/{horizon}", (6, frame_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )
        out.append(canvas)

    video_dir = logdir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(video_dir / "cf_grid.mp4", out, fps=fps)
    imageio.imwrite(video_dir / "cf_contact_sheet.png", out[-1])
    print(f"[cf] {video_dir / 'cf_grid.mp4'}")


def main() -> int:
    args = _parse_args()

    cardreamer_root = Path(args.cardreamer).resolve()
    if not (cardreamer_root / "dreamerv3").is_dir():
        raise FileNotFoundError(f"{cardreamer_root} 下没有 dreamerv3/")
    sys.path.insert(0, str(cardreamer_root))
    sys.path.insert(0, str(cardreamer_root / "dreamerv3"))
    if args.carla_root:
        sys.path.insert(0, str(Path(args.carla_root) / "PythonAPI" / "carla"))

    import numpy as np

    import embodied

    env, agent, _ = _build_stack(args)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed + i for i in range(args.kpi_episodes)]
    rows: list[dict] = []
    last: dict = {}

    def per_episode(ep, ep_info, worker):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        row = {"episode": len(rows) + 1, "seed": seeds[min(len(rows), len(seeds) - 1)],
               "length": length, "score": score}
        row.update(_episode_kpis(ep_info))
        rows.append(row)
        last["ep"] = {"birdeye_wpt": np.asarray(ep["birdeye_wpt"]),
                      "action": np.asarray(ep["action"]),
                      "camera": np.asarray(ep.get("camera", []), dtype=np.uint8)}
        print(f"[episode {row['episode']}] seed={row['seed']} steps={length} "
              f"return={score:.1f} collision={row['collision']} "
              f"dest={row['destination']} lane={row['out_of_lane']}")
        if len(rows) < len(seeds):
            np.random.seed(seeds[len(rows)])

    driver = embodied.Driver(env)
    driver.on_episode(per_episode)
    policy = lambda *a: agent.policy(*a, mode="eval")

    start = time.time()
    np.random.seed(seeds[0])
    for _ in range(args.kpi_episodes):
        driver(policy, episodes=1)
    print(f"[kpi] {args.kpi_episodes} 个 episode 用时 {time.time() - start:.1f}s")
    _write_kpi(rows, logdir)

    if "ep" not in last:
        print("[warn] 没有收到完整 episode，跳过反事实分析")
        return 1

    _analyse(args, agent, last["ep"], logdir)
    print(f"[done] 结果目录：{logdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
