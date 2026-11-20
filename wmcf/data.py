"""Episode generation and sequence sampling."""

from __future__ import annotations

import numpy as np
import torch

from .envs import ConfoundedBallEnv


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    return (np.clip(frame, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def generate_episodes(
    img_size: int,
    n_episodes: int,
    episode_len: int,
    policy_noise: float,
    seed: int,
    confound: bool,
    tint_strength: float = 0.45,
) -> list[dict]:
    """Roll out the behaviour policy and store frames, actions, rewards and true positions."""

    rng = np.random.default_rng(seed)
    env = ConfoundedBallEnv(img_size=img_size, confound=confound, tint_strength=tint_strength)
    episodes: list[dict] = []

    for _ in range(n_episodes):
        pos = env.reset(rng)
        obs_list: list[np.ndarray] = []
        act_list: list[int] = []
        rew_list: list[float] = []
        pos_list: list[np.ndarray] = [pos.copy()]

        for _ in range(episode_len):
            action = env.behavior_action(pos, policy_noise, rng)
            # 注意：当前帧的线索编码的是“即将执行的动作”，这是走捷径的来源
            obs_list.append(_to_uint8(env.render(pos, cue=action)))
            pos, reward, _done = env.step(pos, action)
            act_list.append(action)
            rew_list.append(reward)
            pos_list.append(pos.copy())

        obs_list.append(_to_uint8(env.render(pos, cue=None)))

        episodes.append(
            {
                "obs": np.stack(obs_list, axis=0),
                "act": np.asarray(act_list, dtype=np.int64),
                "rew": np.asarray(rew_list, dtype=np.float32),
                "pos": np.stack(pos_list, axis=0).astype(np.float32),
            }
        )

    return episodes


class SequenceBuffer:
    """Fixed-size replay buffer of variable-length episodes."""

    def __init__(self, episodes: list[dict], seq_len: int):
        self.episodes = [ep for ep in episodes if len(ep["act"]) >= seq_len]
        if not self.episodes:
            lengths = [len(ep["act"]) for ep in episodes]
            raise ValueError(
                f"no episode is long enough for seq_len={seq_len}; lengths={lengths[:10]}..."
            )
        self.seq_len = seq_len

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        seq_len = self.seq_len
        obs_batch, act_batch, rew_batch = [], [], []

        for _ in range(batch_size):
            episode = self.episodes[int(rng.integers(0, len(self.episodes)))]
            max_start = len(episode["act"]) - seq_len
            start = int(rng.integers(0, max_start + 1))
            obs_batch.append(episode["obs"][start : start + seq_len + 1])
            act_batch.append(episode["act"][start : start + seq_len])
            rew_batch.append(episode["rew"][start : start + seq_len])

        obs = torch.as_tensor(np.stack(obs_batch), dtype=torch.float32, device=device) / 255.0
        act = torch.as_tensor(np.stack(act_batch), dtype=torch.long, device=device)
        rew = torch.as_tensor(np.stack(rew_batch), dtype=torch.float32, device=device)
        cont = torch.ones_like(rew)
        return {"obs": obs, "act": act, "rew": rew, "cont": cont}


def episode_to_tensors(episode: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    obs = torch.as_tensor(episode["obs"], dtype=torch.float32, device=device) / 255.0
    act = torch.as_tensor(episode["act"], dtype=torch.long, device=device)
    return obs, act
