"""Train the world model (encoder + RSSM + heads) on synthetic pixel trajectories."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .config import load_config, save_config
from .data import SequenceBuffer, generate_episodes
from .models import WorldModel
from .utils import append_csv, resolve_device, set_seed, write_json


def _maybe_shuffle_actions(
    actions: torch.Tensor, probability: float
) -> torch.Tensor:
    """Negative control: destroy action-observation alignment for some samples."""

    if probability <= 0.0:
        return actions
    mask = torch.rand(actions.shape[0], device=actions.device) < probability
    if not bool(mask.any()):
        return actions
    permutation = torch.argsort(torch.rand_like(actions, dtype=torch.float), dim=1)
    shuffled = torch.gather(actions, 1, permutation)
    out = actions.clone()
    out[mask] = shuffled[mask]
    return out


def build_world_model(cfg, device: torch.device) -> WorldModel:
    model = WorldModel(cfg).to(device)
    return model


def make_dataset(cfg, device: torch.device) -> SequenceBuffer:
    episodes = generate_episodes(
        img_size=int(cfg.data.img_size),
        n_episodes=int(cfg.data.n_episodes),
        episode_len=int(cfg.data.episode_len),
        policy_noise=float(cfg.data.behavior_noise),
        seed=int(cfg.seed),
        confound=bool(cfg.data.confound),
        tint_strength=float(cfg.data.tint_strength),
    )
    return SequenceBuffer(episodes, seq_len=int(cfg.train.seq_len))


def save_checkpoint(path: Path, model: WorldModel, cfg, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="dotted key=value overrides")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.set)
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))

    out_dir = Path(cfg.train.out_dir) / str(cfg.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    print(f"[wm] device={device} run={cfg.run_name} out={out_dir}")
    print(
        f"[wm] data: {cfg.data.n_episodes} episodes x {cfg.data.episode_len} steps, "
        f"img={cfg.data.img_size}, confound={cfg.data.confound}"
    )

    buffer = make_dataset(cfg, device)
    model = build_world_model(cfg, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg.train.lr), eps=float(cfg.train.eps)
    )

    rng = np.random.default_rng(int(cfg.seed))
    val_batch = buffer.sample(int(cfg.train.batch_size), rng, device)

    log_path = out_dir / "train_log.csv"
    if log_path.exists():
        log_path.unlink()

    steps = int(cfg.train.steps)
    log_every = max(1, int(cfg.train.log_every))
    ckpt_every = max(1, int(cfg.train.ckpt_every))
    started = time.time()

    for step in range(1, steps + 1):
        batch = buffer.sample(int(cfg.train.batch_size), rng, device)
        batch["act"] = _maybe_shuffle_actions(batch["act"], float(cfg.train.action_shuffle))

        total, metrics = model.compute_losses(batch)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
        optimizer.step()

        if step % log_every == 0 or step == 1 or step == steps:
            with torch.no_grad():
                _, val_metrics = model.compute_losses(val_batch)
            row = {
                "step": step,
                "loss": float(metrics["loss"]),
                "recon": float(metrics["recon"]),
                "reward": float(metrics["reward"]),
                "cont": float(metrics["cont"]),
                "kl": float(metrics["kl"]),
                "inv": float(metrics["inv"]),
                "val_loss": float(val_metrics["loss"]),
                "val_recon": float(val_metrics["recon"]),
                "seconds": round(time.time() - started, 2),
            }
            append_csv(log_path, row)
            print(
                f"[wm] step {step:5d}/{steps} loss={row['loss']:.4f} "
                f"recon={row['recon']:.4f} kl={row['kl']:.2f} "
                f"inv={row['inv']:.3f} val={row['val_loss']:.4f}"
            )

        if step % ckpt_every == 0 or step == steps:
            save_checkpoint(out_dir / "wm.pt", model, cfg, step)

    write_json(
        out_dir / "summary.json",
        {
            "run_name": str(cfg.run_name),
            "steps": steps,
            "device": str(device),
            "confound": bool(cfg.data.confound),
            "action_shuffle": float(cfg.train.action_shuffle),
            "inverse_dynamics_coef": float(cfg.train.inverse_dynamics_coef),
            "checkpoint": str(out_dir / "wm.pt"),
            "seconds": round(time.time() - started, 2),
        },
    )
    print(f"[wm] done, checkpoint at {out_dir / 'wm.pt'}")


if __name__ == "__main__":
    main()
