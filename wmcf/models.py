"""Encoder, decoder, categorical RSSM, world-model heads, and actor-critic."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributions as dists
import torch.nn.functional as F


# --------------------------------------------------------------------------- utils
def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1.0)


def mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ELU(),
        nn.Linear(hidden, hidden),
        nn.ELU(),
        nn.Linear(hidden, out_dim),
    )


def categorical_sample(
    logits: torch.Tensor,
    straight_through: bool = False,
    sample: bool = True,
) -> torch.Tensor:
    """Sample ``(B, C, K)`` logits and return flattened one-hot latents ``(B, C*K)``."""

    num_classes = logits.shape[-1]
    if not sample:
        index = logits.argmax(dim=-1)
        onehot = F.one_hot(index, num_classes).float()
    elif straight_through:
        onehot = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
    else:
        probs = F.softmax(logits, dim=-1)
        flat = probs.reshape(-1, num_classes)
        index = torch.multinomial(flat, num_samples=1)
        onehot = F.one_hot(index.squeeze(-1), num_classes).float().view_as(probs)
    return onehot.flatten(1)


def categorical_kl(post_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
    """KL(q || p) per category, shape ``(..., C)``."""

    log_q = F.log_softmax(post_logits, dim=-1)
    log_p = F.log_softmax(prior_logits, dim=-1)
    return (log_q.exp() * (log_q - log_p)).sum(dim=-1)


# ----------------------------------------------------------------------- networks
class ConvEncoder(nn.Module):
    def __init__(self, img_size: int, in_ch: int = 3, embed_dim: int = 256, width: int = 32):
        super().__init__()
        if img_size % 16 != 0:
            raise ValueError("img_size must be divisible by 16")
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 4, 2, 1),
            nn.ELU(),
            nn.Conv2d(width, width * 2, 4, 2, 1),
            nn.ELU(),
            nn.Conv2d(width * 2, width * 4, 4, 2, 1),
            nn.ELU(),
            nn.Conv2d(width * 4, width * 8, 4, 2, 1),
            nn.ELU(),
            nn.Flatten(),
        )
        feat_dim = width * 8 * (img_size // 16) ** 2
        self.fc = nn.Sequential(nn.Linear(feat_dim, embed_dim), nn.ELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))


class ConvDecoder(nn.Module):
    def __init__(self, in_dim: int, img_size: int, out_ch: int = 3, width: int = 32):
        super().__init__()
        if img_size % 16 != 0:
            raise ValueError("img_size must be divisible by 16")
        self.width = width
        self.init_hw = img_size // 16
        self.fc = nn.Sequential(
            nn.Linear(in_dim, width * 8 * self.init_hw**2),
            nn.ELU(),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(width * 8, width * 4, 4, 2, 1),
            nn.ELU(),
            nn.ConvTranspose2d(width * 4, width * 2, 4, 2, 1),
            nn.ELU(),
            nn.ConvTranspose2d(width * 2, width, 4, 2, 1),
            nn.ELU(),
            nn.ConvTranspose2d(width, out_ch, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x).view(-1, self.width * 8, self.init_hw, self.init_hw)
        return self.net(h)


class RSSM(nn.Module):
    """Categorical RSSM: deterministic h plus stochastic one-hot z."""

    def __init__(
        self,
        n_actions: int = 4,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        stoch: int = 32,
        classes: int = 32,
        action_embed_dim: int = 32,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.stoch = stoch
        self.classes = classes
        self.z_dim = stoch * classes
        self.hidden_dim = hidden_dim

        # index 0 is a reserved "no-op" used at t=0, env action a maps to a+1
        self.act_embed = nn.Embedding(n_actions + 1, action_embed_dim)
        self.gru = nn.GRUCell(self.z_dim + action_embed_dim, hidden_dim)
        self.prior_net = mlp(hidden_dim, hidden_dim, self.z_dim)
        self.post_net = mlp(hidden_dim + embed_dim, hidden_dim, self.z_dim)

    def initial_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.z_dim, device=device)
        return h, z

    def prior_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.prior_net(h).view(-1, self.stoch, self.classes)

    def posterior_logits(self, h: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        return self.post_net(torch.cat([h, embed], dim=-1)).view(
            -1, self.stoch, self.classes
        )


class Actor(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = mlp(state_dim, hidden_dim, n_actions)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z], dim=-1))

    def distribution(self, h: torch.Tensor, z: torch.Tensor):
        return dists.Categorical(logits=self.forward(h, z))


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = mlp(state_dim, hidden_dim, 1)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z], dim=-1)).squeeze(-1)


# --------------------------------------------------------------------- world model
class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg.model
        self.img_size = int(cfg.data.img_size)
        self.n_actions = int(model_cfg.n_actions)
        self.hidden_dim = int(model_cfg.hidden_dim)
        self.stoch = int(model_cfg.stoch)
        self.classes = int(model_cfg.classes)
        self.z_dim = self.stoch * self.classes

        self.kl_free_bits = float(model_cfg.kl_free_bits)
        self.kl_balance = float(model_cfg.kl_balance)
        self.inv_coef = float(cfg.train.inverse_dynamics_coef)

        self.encoder = ConvEncoder(
            self.img_size, embed_dim=int(model_cfg.embed_dim), width=int(model_cfg.width)
        )
        self.decoder = ConvDecoder(
            self.hidden_dim + self.z_dim, self.img_size, width=int(model_cfg.width)
        )
        self.rssm = RSSM(
            n_actions=self.n_actions,
            embed_dim=int(model_cfg.embed_dim),
            hidden_dim=self.hidden_dim,
            stoch=self.stoch,
            classes=self.classes,
        )
        state_dim = self.hidden_dim + self.z_dim
        self.reward_head = mlp(state_dim, self.hidden_dim, 1)
        self.cont_head = mlp(state_dim, self.hidden_dim, 1)
        self.inv_head = mlp(2 * self.z_dim, self.hidden_dim, self.n_actions)

    # ------------------------------------------------------------------ core
    def observe_seq(self, obs: torch.Tensor, actions: torch.Tensor) -> dict:
        """Encode a window with the posterior.

        Args:
            obs: ``(B, T+1, 3, H, W)`` frames in ``[0, 1]``.
            actions: ``(B, T)`` env actions; ``actions[:, t]`` leads from obs_t to obs_{t+1}.
        """

        batch, num_frames = obs.shape[:2]
        num_actions = actions.shape[1]
        if num_frames != num_actions + 1:
            raise ValueError(f"expected T+1 frames for T actions, got {num_frames} and {num_actions}")

        embed = self.encoder(obs.reshape(batch * num_frames, *obs.shape[2:])).reshape(
            batch, num_frames, -1
        )
        device = obs.device
        h, z = self.rssm.initial_state(batch, device)

        hs, zs, posts, priors = [], [], [], []
        for step in range(num_frames):
            if step == 0:
                action_index = torch.zeros(batch, dtype=torch.long, device=device)
            else:
                action_index = actions[:, step - 1] + 1
            h = self.rssm.gru(
                torch.cat([z, self.rssm.act_embed(action_index)], dim=-1), h
            )
            prior = self.rssm.prior_logits(h)
            post = self.rssm.posterior_logits(h, embed[:, step])
            z = categorical_sample(post, straight_through=True)
            hs.append(h)
            zs.append(z)
            posts.append(post)
            priors.append(prior)

        return {
            "h": torch.stack(hs, dim=1),
            "z": torch.stack(zs, dim=1),
            "post": torch.stack(posts, dim=1),
            "prior": torch.stack(priors, dim=1),
            "embed": embed,
        }

    def compute_losses(self, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs, actions, rewards = batch["obs"], batch["act"], batch["rew"]
        cont = batch["cont"]
        out = self.observe_seq(obs, actions)

        batch_size, num_frames = obs.shape[:2]
        num_steps = num_frames - 1

        flat_h = out["h"].reshape(batch_size * num_frames, -1)
        flat_z = out["z"].reshape(batch_size * num_frames, -1)
        recon = self.decoder(torch.cat([flat_h, flat_z], dim=-1)).reshape(
            batch_size, num_frames, *obs.shape[2:]
        )
        loss_recon = F.mse_loss(recon, obs)

        state = torch.cat([out["h"][:, :num_steps], out["z"][:, :num_steps]], dim=-1)
        pred_reward = self.reward_head(state).squeeze(-1)
        loss_reward = F.mse_loss(pred_reward, symlog(rewards))
        pred_cont = self.cont_head(state).squeeze(-1)
        loss_cont = F.binary_cross_entropy_with_logits(pred_cont, cont)

        kl_dyn = categorical_kl(out["post"].detach(), out["prior"])
        kl_rep = categorical_kl(out["post"], out["prior"].detach())
        kl = self.kl_balance * kl_dyn + (1.0 - self.kl_balance) * kl_rep
        kl = torch.clamp(kl, min=self.kl_free_bits)
        loss_kl = kl.sum(dim=-1).mean()

        if self.inv_coef > 0.0:
            z_now = out["z"][:, :-1]
            z_next = out["z"][:, 1:]
            action_logits = self.inv_head(torch.cat([z_now, z_next], dim=-1))
            loss_inv = F.cross_entropy(
                action_logits.reshape(-1, self.n_actions), actions.reshape(-1)
            )
        else:
            loss_inv = torch.zeros((), device=obs.device)

        total = loss_recon + loss_reward + loss_cont + loss_kl + self.inv_coef * loss_inv
        with torch.no_grad():
            metrics = {
                "loss": total.detach(),
                "recon": loss_recon.detach(),
                "reward": loss_reward.detach(),
                "cont": loss_cont.detach(),
                "kl": loss_kl.detach(),
                "inv": loss_inv.detach(),
            }
        return total, metrics

    # ------------------------------------------------------------- imagination
    def decode(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([h, z], dim=-1))

    def imagine_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor | int,
        sample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the prior one step. ``sample=False`` uses the MAP latent.

        Returns ``(h, z, prior_logits)`` so callers can also inspect the predictive
        distribution instead of a single sampled latent.
        """

        if not torch.is_tensor(action):
            action = torch.full((h.shape[0],), int(action), dtype=torch.long, device=h.device)
        elif action.dim() == 0:
            action = action.expand(h.shape[0])
        action_index = action.long() + 1
        h = self.rssm.gru(torch.cat([z, self.rssm.act_embed(action_index)], dim=-1), h)
        logits = self.rssm.prior_logits(h)
        z = categorical_sample(logits, sample=sample)
        return h, z, logits
