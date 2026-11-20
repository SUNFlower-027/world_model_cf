"""Counterfactual action interventions and imagined-trajectory divergence analysis.

For every held-out trajectory and intervention step k we roll the prior forward under
``do(a_k = a')`` and compare against the factual rollout. Three families of numbers are
reported:

* latent / pixel / reward divergence curves between factual and counterfactual imagination
* a stochastic noise floor (same action, resampled latents) that divergence must exceed
* counterfactual fidelity against the true environment, which is available in closed form
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import load_config
from .data import episode_to_tensors, generate_episodes
from .envs import ConfoundedBallEnv
from .models import WorldModel, symexp
from .utils import append_csv, resolve_device, set_seed, write_json


# ------------------------------------------------------------------ imagination
@torch.no_grad()
def encode_prefix(wm: WorldModel, obs: torch.Tensor, actions: torch.Tensor):
    out = wm.observe_seq(obs, actions)
    return out["h"][:, -1], out["z"][:, -1]


@torch.no_grad()
def imagine(
    wm: WorldModel,
    h: torch.Tensor,
    z: torch.Tensor,
    actions: torch.Tensor,
    sample: bool = False,
) -> dict[str, torch.Tensor]:
    """Roll out the prior. ``sample=False`` gives the deterministic MAP trajectory.

    ``soft`` holds the flattened prior probabilities, which give a smooth divergence
    signal; ``z`` holds the sampled one-hot latent that actually drives the dynamics.
    """

    latents, soft, frames, rewards = [], [], [], []
    for action in actions:
        h, z, logits = wm.imagine_step(h, z, action, sample=sample)
        latents.append(z)
        soft.append(torch.softmax(logits, dim=-1).flatten(1))
        frames.append(wm.decode(h, z))
        rewards.append(symexp(wm.reward_head(torch.cat([h, z], dim=-1)).squeeze(-1)))
    return {
        "z": torch.stack(latents),        # (H, B, z_dim)
        "soft": torch.stack(soft),        # (H, B, z_dim)
        "frames": torch.stack(frames),    # (H, B, 3, S, S)
        "reward": torch.stack(rewards),   # (H, B)
    }


def ground_truth_rollout(env: ConfoundedBallEnv, pos: np.ndarray, actions) -> np.ndarray:
    """Simulate ``do(a_k..a_{k+H-1})`` in the real environment, rendering true frames."""

    actions = [int(a) for a in actions]
    frames = []
    for index, action in enumerate(actions):
        pos, _, _ = env.step(pos, action)
        next_cue = actions[index + 1] if index + 1 < len(actions) else None
        frames.append(env.render(pos, cue=next_cue))
    return np.stack(frames, axis=0).astype(np.float32)


# --------------------------------------------------------------------- analysis
def _aggregate(records: list[dict], key: str):
    values = np.stack([record[key] for record in records])
    return values.mean(axis=0), values.std(axis=0)


def _onset(curve: np.ndarray, threshold: float) -> int:
    hits = np.where(curve > threshold)[0]
    return int(hits[0]) if len(hits) else -1


def _growth(curve: np.ndarray) -> float:
    x = np.arange(len(curve), dtype=np.float64)
    y = np.log(np.maximum(curve, 1e-6))
    if np.allclose(y, y[0]):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _action_statistics(records: list[dict]) -> dict:
    """How consistent and decodable the latent effect of an intervention is."""

    deltas: dict[int, list[np.ndarray]] = defaultdict(list)
    for record in records:
        if record["mode"] != "persistent":
            continue
        delta = record["delta_final"]
        norm = np.linalg.norm(delta) + 1e-8
        deltas[int(record["alt"])].append(delta / norm)

    within = []
    for vectors in deltas.values():
        if len(vectors) < 2:
            continue
        matrix = np.stack(vectors)
        similarity = matrix @ matrix.T
        upper = similarity[np.triu_indices(len(vectors), k=1)]
        within.append(float(upper.mean()))

    centroids = {
        action: np.stack(vectors).mean(axis=0)
        for action, vectors in deltas.items()
        if vectors
    }
    for action in centroids:
        centroids[action] = centroids[action] / (np.linalg.norm(centroids[action]) + 1e-8)

    actions = sorted(centroids)
    between = []
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            between.append(float(centroids[actions[i]] @ centroids[actions[j]]))

    correct, total = 0, 0
    for record in records:
        if record["mode"] != "persistent":
            continue
        delta = record["delta_final"]
        delta = delta / (np.linalg.norm(delta) + 1e-8)
        scores = {action: float(delta @ centroids[action]) for action in actions}
        prediction = max(scores, key=scores.get)
        correct += int(prediction == int(record["alt"]))
        total += 1

    return {
        "within_action_cosine": float(np.mean(within)) if within else None,
        "between_action_cosine": float(np.mean(between)) if between else None,
        "loo_centroid_accuracy": (correct / total) if total else None,
        "chance_accuracy": 1.0 / max(1, len(actions)),
        "n_deltas": total,
    }


def analyze(cfg, wm: WorldModel, episodes: list[dict], device: torch.device) -> dict:
    horizon = int(cfg.eval.horizon)
    intervention_steps = [int(step) for step in cfg.eval.intervention_steps]
    modes = [str(mode) for mode in cfg.eval.modes]
    n_prior_samples = max(2, int(cfg.eval.n_prior_samples))
    n_actions = int(wm.n_actions)
    z_dim = wm.z_dim
    max_states = int(cfg.eval.n_states)

    env = ConfoundedBallEnv(
        img_size=int(wm.img_size),
        confound=bool(cfg.data.eval_confound),
        tint_strength=float(cfg.data.tint_strength),
    )

    records: list[dict] = []
    noise_curves: list[np.ndarray] = []
    n_states = 0

    for episode_index, episode in enumerate(episodes):
        if n_states >= max_states:
            break
        obs_full, act_full = episode_to_tensors(episode, device)
        for k in intervention_steps:
            if n_states >= max_states:
                break
            if k + horizon > act_full.shape[0]:
                continue

            h, z = encode_prefix(
                wm, obs_full[: k + 1].unsqueeze(0), act_full[:k].unsqueeze(0)
            )
            factual_actions = act_full[k : k + horizon]
            factual = imagine(wm, h, z, factual_actions, sample=False)

            # 噪声底：固定事实动作，只改变初始后验采样，观察软轨迹的离散度
            prefix_obs = obs_full[: k + 1].unsqueeze(0)
            prefix_act = act_full[:k].unsqueeze(0)
            stochastic = []
            for _ in range(n_prior_samples):
                h_sample, z_sample = encode_prefix(wm, prefix_obs, prefix_act)
                sample = imagine(wm, h_sample, z_sample, factual_actions, sample=False)
                stochastic.append(sample["soft"].squeeze(1).cpu().numpy())
            stack = np.stack(stochastic)  # (K, H, z_dim)
            mean = stack.mean(axis=0)
            noise = np.sqrt(((stack - mean) ** 2).sum(axis=-1)) / np.sqrt(z_dim)
            noise_curves.append(noise.mean(axis=0))

            for mode in modes:
                for alt in range(n_actions):
                    cf_actions = factual_actions.clone()
                    if mode == "one_step":
                        cf_actions[0] = alt
                    else:
                        cf_actions[:] = alt
                    if torch.equal(cf_actions, factual_actions):
                        continue

                    counterfactual = imagine(wm, h, z, cf_actions, sample=False)

                    latent_diff = counterfactual["soft"].squeeze(1) - factual["soft"].squeeze(1)
                    latent_curve = latent_diff.norm(dim=-1).cpu().numpy() / np.sqrt(z_dim)
                    pixel_curve = (
                        (counterfactual["frames"] - factual["frames"])
                        .pow(2)
                        .flatten(2)
                        .mean(-1)
                        .squeeze(1)
                        .cpu()
                        .numpy()
                    )
                    reward_curve = (
                        (counterfactual["reward"] - factual["reward"])
                        .abs()
                        .squeeze(1)
                        .cpu()
                        .numpy()
                    )

                    gt_frames = ground_truth_rollout(
                        env, episode["pos"][k].copy(), cf_actions.cpu().numpy()
                    )
                    gt_tensor = torch.as_tensor(gt_frames, dtype=torch.float32, device=device)
                    gt_mse = (
                        (counterfactual["frames"].squeeze(1) - gt_tensor)
                        .pow(2)
                        .flatten(1)
                        .mean(-1)
                        .cpu()
                        .numpy()
                    )

                    records.append(
                        {
                            "episode": episode_index,
                            "k": k,
                            "mode": mode,
                            "alt": alt,
                            "latent": latent_curve,
                            "pixel": pixel_curve,
                            "reward": reward_curve,
                            "gt_mse": gt_mse,
                            "delta_final": latent_diff[-1].cpu().numpy(),
                            "cf_final": counterfactual["soft"][-1, 0].cpu().numpy(),
                        }
                    )
            n_states += 1

    noise_floor = float(np.mean([curve.mean() for curve in noise_curves])) if noise_curves else 0.0
    threshold = max(1e-4, 2.0 * noise_floor)

    summary: dict = {
        "n_states": n_states,
        "n_records": len(records),
        "noise_floor_mean": noise_floor,
        "onset_threshold": threshold,
        "per_mode": {},
        "curves": {},
        "action_statistics": _action_statistics(records),
    }

    for mode in modes:
        subset = [record for record in records if record["mode"] == mode]
        if not subset:
            continue
        latent_mean, latent_std = _aggregate(subset, "latent")
        pixel_mean, pixel_std = _aggregate(subset, "pixel")
        reward_mean, reward_std = _aggregate(subset, "reward")
        gt_mean, gt_std = _aggregate(subset, "gt_mse")

        summary["per_mode"][mode] = {
            "n_records": len(subset),
            "latent_divergence_mean": latent_mean.tolist(),
            "latent_divergence_std": latent_std.tolist(),
            "onset_step": _onset(latent_mean, threshold),
            "growth_rate": _growth(latent_mean),
            "final_over_first": float(latent_mean[-1] / (latent_mean[0] + 1e-8)),
            "pixel_divergence_mean": pixel_mean.tolist(),
            "reward_divergence_mean": reward_mean.tolist(),
            "counterfactual_gt_mse_mean": gt_mean.tolist(),
            "snr_over_noise_floor": float(latent_mean.mean() / (noise_floor + 1e-8)),
        }
        summary["curves"][mode] = {
            "latent_mean": latent_mean.tolist(),
            "latent_std": latent_std.tolist(),
            "pixel_mean": pixel_mean.tolist(),
            "pixel_std": pixel_std.tolist(),
            "reward_mean": reward_mean.tolist(),
            "reward_std": reward_std.tolist(),
            "gt_mse_mean": gt_mean.tolist(),
            "gt_mse_std": gt_std.tolist(),
        }

    summary["noise_curves"] = (
        np.stack(noise_curves).tolist() if noise_curves else []
    )
    summary["_records"] = records
    return summary


# ---------------------------------------------------------------------- outputs
def write_curves_csv(path: Path, summary: dict) -> None:
    path.unlink(missing_ok=True)
    for mode, curves in summary["curves"].items():
        for step in range(len(curves["latent_mean"])):
            append_csv(
                path,
                {
                    "mode": mode,
                    "step": step,
                    "latent_mean": curves["latent_mean"][step],
                    "latent_std": curves["latent_std"][step],
                    "pixel_mean": curves["pixel_mean"][step],
                    "reward_mean": curves["reward_mean"][step],
                    "gt_mse_mean": curves["gt_mse_mean"][step],
                },
            )


def plot_curves(path: Path, summary: dict) -> None:
    modes = list(summary["curves"])
    if not modes:
        return
    colors = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        (axes[0, 0], "latent_mean", "latent_std", "Latent divergence (L2 / sqrt(dim))"),
        (axes[0, 1], "pixel_mean", "pixel_std", "Pixel divergence (MSE)"),
        (axes[1, 0], "reward_mean", "reward_std", "Reward divergence (|diff|)"),
        (axes[1, 1], "gt_mse_mean", "gt_mse_std", "Counterfactual fidelity (MSE vs true env)"),
    ]
    for axis, key, std_key, title in panels:
        for index, mode in enumerate(modes):
            curve = np.asarray(summary["curves"][mode][key])
            std = np.asarray(summary["curves"][mode][std_key])
            steps = np.arange(len(curve))
            axis.plot(steps, curve, label=mode, color=colors[index % len(colors)])
            axis.fill_between(
                steps, curve - std, curve + std, alpha=0.18, color=colors[index % len(colors)]
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("imagination step")
        axis.grid(alpha=0.3)
    axes[0, 0].axhline(
        summary["noise_floor_mean"], linestyle="--", color="gray", label="noise floor"
    )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Counterfactual action intervention: imagined-trajectory divergence")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="path to wm.pt")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.set)
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    wm_cfg = checkpoint["cfg"]
    wm = WorldModel(wm_cfg).to(device)
    wm.load_state_dict(checkpoint["model"])
    wm.eval()

    episodes = generate_episodes(
        img_size=int(wm_cfg.data.img_size),
        n_episodes=int(cfg.data.eval_episodes),
        episode_len=int(cfg.data.eval_episode_len),
        policy_noise=float(cfg.data.eval_behavior_noise),
        seed=int(cfg.seed) + 10000,
        confound=bool(cfg.data.eval_confound),
        tint_strength=float(cfg.data.tint_strength),
    )

    summary = analyze(cfg, wm, episodes, device)

    out_dir = Path(cfg.eval.out_dir) / str(cfg.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = summary.pop("_records")
    summary["checkpoint"] = str(args.checkpoint)
    summary["run_name"] = str(cfg.run_name)
    summary["eval_confound"] = bool(cfg.data.eval_confound)
    summary["horizon"] = int(cfg.eval.horizon)
    summary["intervention_steps"] = [int(step) for step in cfg.eval.intervention_steps]

    write_json(out_dir / "metrics.json", summary)
    write_curves_csv(out_dir / "divergence_curves.csv", summary)
    plot_curves(out_dir / "divergence.png", summary)

    print(f"[cf] states={summary['n_states']} records={summary['n_records']}")
    print(f"[cf] noise floor={summary['noise_floor_mean']:.4f}")
    for mode, stats in summary["per_mode"].items():
        print(
            f"[cf] {mode:10s} onset={stats['onset_step']:2d} "
            f"growth={stats['growth_rate']:+.3f} "
            f"mean_div={np.mean(stats['latent_divergence_mean']):.4f} "
            f"snr={stats['snr_over_noise_floor']:.2f} "
            f"gt_mse={np.mean(stats['counterfactual_gt_mse_mean']):.4f}"
        )
    print(f"[cf] leaked action statistics: {summary['action_statistics']}")
    print(f"[cf] outputs at {out_dir}")


if __name__ == "__main__":
    main()
