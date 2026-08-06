"""Simple PPO trainer for centralized communication controller."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli

from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json, set_seed


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)


@dataclass
class PPOTrainer:
    cfg: dict[str, Any]
    device: str = "cpu"

    def __post_init__(self) -> None:
        ppo = self.cfg.get("ppo", {})
        self.lr = float(ppo.get("lr", 3e-4))
        self.gamma = float(ppo.get("gamma", 0.99))
        self.gae_lambda = float(ppo.get("gae_lambda", 0.95))
        self.clip_range = float(ppo.get("clip_range", 0.2))
        self.ent_coef = float(ppo.get("ent_coef", 0.01))
        self.vf_coef = float(ppo.get("vf_coef", 0.5))
        self.max_grad_norm = float(ppo.get("max_grad_norm", 0.5))
        self.n_steps = int(ppo.get("n_steps", 2048))
        self.batch_size = int(ppo.get("batch_size", 256))
        self.n_epochs = int(ppo.get("n_epochs", 10))
        self.total_timesteps = int(ppo.get("total_timesteps", 200000))

    def make_env(self, split: str = "train", max_sensors: int | None = None) -> TraceDrivenCampusEnv:
        return TraceDrivenCampusEnv(
            cfg=self.cfg,
            split=split,
            max_sensors=max_sensors,
            multi_agent=False,
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
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        model = ActorCritic(obs_dim, act_dim).to(self.device)
        opt = optim.Adam(model.parameters(), lr=self.lr)

        root = repo_root()
        ckpt_dir = ensure_dir(
            checkpoint_dir or root / self.cfg.get("training", {}).get("checkpoint_dir", "outputs/experiments") / "ppo"
        )
        metrics: list[dict[str, Any]] = []
        global_step = 0
        obs, _ = env.reset()
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        def compute_gae(rewards, values, dones, last_val):
            adv = np.zeros_like(rewards)
            last_gae = 0.0
            for t in reversed(range(len(rewards))):
                next_val = last_val if t == len(rewards) - 1 else values[t + 1]
                next_non_terminal = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
                adv[t] = last_gae
            returns = adv + values
            return adv, returns

        while global_step < self.total_timesteps:
            obs_buf.clear()
            act_buf.clear()
            logp_buf.clear()
            rew_buf.clear()
            val_buf.clear()
            done_buf.clear()

            for _ in range(self.n_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).flatten()
                with torch.no_grad():
                    logits, value = model(obs_t.unsqueeze(0))
                    dist = Bernoulli(logits=logits)
                    action = dist.sample()
                    logp = dist.log_prob(action).sum()
                action_np = action.cpu().numpy().astype(int)
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated

                obs_buf.append(obs.copy())
                act_buf.append(action_np.copy())
                logp_buf.append(float(logp.item()))
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
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).flatten()
                _, last_val = model(obs_t.unsqueeze(0))
                last_val = float(last_val.item())

            adv, ret = compute_gae(
                np.asarray(rew_buf),
                np.asarray(val_buf),
                np.asarray(done_buf),
                last_val,
            )
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            obs_arr = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=self.device).view(len(obs_buf), -1)
            act_arr = torch.as_tensor(np.asarray(act_buf), dtype=torch.float32, device=self.device)
            old_logp = torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=self.device)
            adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
            ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

            n = len(obs_buf)
            idx = np.arange(n)
            for _ in range(self.n_epochs):
                np.random.shuffle(idx)
                for start in range(0, n, self.batch_size):
                    mb = idx[start : start + self.batch_size]
                    logits, values = model(obs_arr[mb])
                    dist = Bernoulli(logits=logits)
                    logp = dist.log_prob(act_arr[mb]).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                    ratio = torch.exp(logp - old_logp[mb])
                    pg1 = ratio * adv_t[mb]
                    pg2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv_t[mb]
                    policy_loss = -torch.min(pg1, pg2).mean()
                    value_loss = ((ret_t[mb] - values) ** 2).mean()
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    opt.step()

            ep_reward = float(np.sum(rew_buf))
            metrics.append({"step": global_step, "rollout_reward": ep_reward})
            print(f"[ppo] step={global_step} rollout_reward={ep_reward:.3f}")

            eval_every = int(self.cfg.get("training", {}).get("eval_every_steps", 10000))
            if global_step % eval_every < self.n_steps:
                eval_r = self.evaluate(model, split="val", max_sensors=max_sensors)
                metrics.append({"step": global_step, "eval_reward": eval_r})

        torch.save({"model_state": model.state_dict(), "cfg": self.cfg}, ckpt_dir / "final_model.pt")
        save_json({"metrics": metrics}, ckpt_dir / "metrics.json")
        return {"checkpoint": str(ckpt_dir / "final_model.pt"), "metrics": metrics}

    @torch.no_grad()
    def evaluate(
        self,
        model: ActorCritic,
        *,
        split: str = "val",
        max_sensors: int | None = None,
        max_steps: int = 500,
    ) -> float:
        env = self.make_env(split, max_sensors=max_sensors)
        obs, _ = env.reset()
        total = 0.0
        for _ in range(max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).flatten()
            logits, _ = model(obs_t.unsqueeze(0))
            action = (torch.sigmoid(logits) > 0.5).int().cpu().numpy().astype(int)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                obs, _ = env.reset()
        return total


def run_ppo_training(
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
        print("[ppo] CUDA unavailable; falling back to CPU")
        device = "cpu"
    print(f"[ppo] device={device}")
    trainer = PPOTrainer(cfg, device=device)
    out = trainer.train(**kwargs)
    out["device"] = device
    return out
