"""Parameter-sharing MAPPO with centralized critic (CTDE)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli

from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json, set_seed


class SharedActor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MeanPoolCritic(nn.Module):
    """Permutation-invariant centralized critic (variable agent count)."""

    def __init__(self, obs_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.v = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (n_agents, obs_dim) or (B, n_agents, obs_dim)
        if obs.dim() == 2:
            h = self.phi(obs)
            g = h.mean(dim=0, keepdim=True)
            return self.v(g).squeeze(-1).squeeze(0)
        h = self.phi(obs)
        g = h.mean(dim=1)
        return self.v(g).squeeze(-1)


# Backward-compatible alias
CentralizedCritic = MeanPoolCritic


@dataclass
class MAPPOTrainer:
    cfg: dict[str, Any]
    device: str = "cpu"

    def __post_init__(self) -> None:
        mappo = self.cfg.get("mappo", {})
        self.lr = float(mappo.get("lr", 3e-4))
        self.gamma = float(mappo.get("gamma", 0.99))
        self.gae_lambda = float(mappo.get("gae_lambda", 0.95))
        self.clip_range = float(mappo.get("clip_range", 0.2))
        self.ent_coef = float(mappo.get("ent_coef", 0.01))
        self.vf_coef = float(mappo.get("vf_coef", 0.5))
        self.n_steps = int(self.cfg.get("ppo", {}).get("n_steps", 1024))
        self.batch_size = int(self.cfg.get("ppo", {}).get("batch_size", 256))
        self.n_epochs = int(self.cfg.get("ppo", {}).get("n_epochs", 5))
        self.total_timesteps = int(mappo.get("total_timesteps", 300000))
        self.parameter_sharing = bool(mappo.get("parameter_sharing", True))
        self.centralized_critic = bool(mappo.get("centralized_critic", True))

    def make_env(self, split: str = "train", max_sensors: int | None = None) -> TraceDrivenCampusEnv:
        # Reconstructor + graph loaded from cfg["environment"] inside the env.
        return TraceDrivenCampusEnv(
            cfg=self.cfg,
            split=split,
            max_sensors=max_sensors,
            multi_agent=True,
        )

    def train(
        self,
        *,
        split: str = "train",
        max_sensors: int | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        set_seed(int(self.cfg.get("seed", 42)))
        env = self.make_env(split, max_sensors=max_sensors)
        obs_dim = env.observation_space.shape[-1]
        n_agents = env.n_sensors
        actor = SharedActor(obs_dim).to(self.device)
        critic = MeanPoolCritic(obs_dim).to(self.device)
        opt = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=self.lr)

        root = repo_root()
        ckpt_dir = ensure_dir(
            checkpoint_dir or root / self.cfg.get("training", {}).get("checkpoint_dir", "outputs/experiments") / "mappo"
        )
        metrics: list[dict[str, Any]] = []
        global_step = 0
        obs, _ = env.reset()

        def compute_gae(rewards, values, dones, last_val):
            adv = np.zeros_like(rewards)
            last_gae = 0.0
            for t in reversed(range(len(rewards))):
                next_val = last_val if t == len(rewards) - 1 else values[t + 1]
                next_non_terminal = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
                adv[t] = last_gae
            return adv, adv + values

        while global_step < self.total_timesteps:
            obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

            for _ in range(self.n_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                logits = actor(obs_t)
                dist = Bernoulli(logits=logits)
                actions = dist.sample()
                logp = dist.log_prob(actions)

                value = critic(obs_t)

                next_obs, reward, terminated, truncated, _ = env.step(actions.cpu().numpy().astype(int))
                done = terminated or truncated

                obs_buf.append(obs.copy())
                act_buf.append(actions.cpu().numpy().copy())
                logp_buf.append(logp.detach().cpu().numpy())
                rew_buf.append(float(reward))
                val_buf.append(float(value.item()))
                done_buf.append(float(done))

                obs = next_obs
                global_step += 1
                if done:
                    obs, _ = env.reset()
                if global_step >= self.total_timesteps:
                    break

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                last_val = float(critic(obs_t).item())

            adv, ret = compute_gae(np.asarray(rew_buf), np.asarray(val_buf), np.asarray(done_buf), last_val)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            obs_arr = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=self.device)
            act_arr = torch.as_tensor(np.asarray(act_buf), dtype=torch.float32, device=self.device)
            old_logp = torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=self.device)
            adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
            ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

            t_steps, n_agents, obs_d = obs_arr.shape
            flat_obs = obs_arr.reshape(t_steps * n_agents, obs_d)
            flat_act = act_arr.reshape(-1)
            flat_logp_old = old_logp.reshape(-1)
            agent_adv = np.repeat(adv, n_agents)
            agent_adv = (agent_adv - agent_adv.mean()) / (agent_adv.std() + 1e-8)
            agent_adv_t = torch.as_tensor(agent_adv, dtype=torch.float32, device=self.device)

            idx = np.arange(t_steps * n_agents)
            for _ in range(self.n_epochs):
                np.random.shuffle(idx)
                for start in range(0, len(idx), self.batch_size):
                    mb = idx[start : start + self.batch_size]
                    logits = actor(flat_obs[mb])
                    dist = Bernoulli(logits=logits)
                    logp = dist.log_prob(flat_act[mb])
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(logp - flat_logp_old[mb])
                    pg1 = ratio * agent_adv_t[mb]
                    pg2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * agent_adv_t[mb]
                    policy_loss = -torch.min(pg1, pg2).mean()

                    mb_steps = np.unique(mb // n_agents)
                    values = critic(obs_arr[mb_steps])  # (B,)
                    value_loss = ((ret_t[mb_steps] - values) ** 2).mean()

                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                    opt.step()

            ep_reward = float(np.sum(rew_buf))
            metrics.append({"step": global_step, "rollout_reward": ep_reward})
            print(f"[mappo] step={global_step} rollout_reward={ep_reward:.3f}")

        torch.save(
            {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "cfg": self.cfg,
                "critic_type": "mean_pool",
                "obs_dim": obs_dim,
                "n_agents_train": n_agents,
            },
            ckpt_dir / "final_model.pt",
        )
        save_json({"metrics": metrics}, ckpt_dir / "metrics.json")
        return {"checkpoint": str(ckpt_dir / "final_model.pt"), "metrics": metrics}


def run_mappo_training(
    cfg: dict[str, Any] | None = None,
    *,
    device: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    root = repo_root()
    cfg = cfg or load_yaml(root / "configs" / "rl.yaml")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("[mappo] CUDA unavailable; falling back to CPU")
        device = "cpu"
    print(f"[mappo] device={device}")
    trainer = MAPPOTrainer(cfg, device=device)
    out = trainer.train(**kwargs)
    out["device"] = device
    return out
