"""Boosted MAPPO: residual heuristic actor + BC warm-start + train without shield.

Addresses the failure mode where the safety shield steals credit and the policy
collapses to always-SKIP. The actor is initialized near a strong semantic expert
and fine-tuned with reshaped rewards and optional soft TX-budget matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli

from campus_senserl.data.cohort import load_cohort
from campus_senserl.environment.communication_model import SKIP, TRANSMIT
from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv
from campus_senserl.evaluation.rl_policy_eval import evaluate_policy, load_mappo_policy, make_final_env
from campus_senserl.rl.expert_policy import SemanticExpertPolicy, heuristic_logits_torch
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json, set_seed


class ResidualSharedActor(nn.Module):
    """π(a|s) with residual expert bias: logit = f_θ(s) + α · h(s)."""

    def __init__(self, obs_dim: int, hidden: int = 128, residual_init: float = 1.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # Start near expert; α is learnable so RL can reduce reliance
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))
        # Slight TX-favoring bias so Bernoulli does not start at p≈0.5 collapse to 0
        nn.init.constant_(self.net[-1].bias, 0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        neural = self.net(x).squeeze(-1)
        return neural + self.residual_scale * heuristic_logits_torch(x)


class MeanPoolCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.v = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 2:
            h = self.phi(obs)
            g = h.mean(dim=0, keepdim=True)
            return self.v(g).squeeze(-1).squeeze(0)
        h = self.phi(obs)
        g = h.mean(dim=1)
        return self.v(g).squeeze(-1)


def _per_agent_rewards(
    info: dict[str, Any],
    actions: np.ndarray,
    *,
    w_tx: float,
    w_err: float,
    w_aoi: float,
    w_miss: float,
    w_event: float,
    w_budget: float,
    target_tx_rate: float,
    max_aoi: float,
) -> np.ndarray:
    """Dense local reward so each sensor gets credit for its own TX/SKIP."""
    n = len(actions)
    local = np.asarray(info.get("local_available"), dtype=bool)
    gt = np.asarray(info["ground_truth"] if "ground_truth" in info else np.full(n, np.nan), dtype=float)
    # env puts reconstruction in info
    recon = np.asarray(info["reconstruction"], dtype=float)
    aoi = np.asarray(info.get("aoi", np.zeros(n)), dtype=float)
    if aoi.shape[0] != n and "aoi" not in info:
        aoi = np.zeros(n, dtype=float)

    missed = np.asarray(info.get("missed_events", np.zeros(n, dtype=bool)), dtype=bool)
    detected = np.asarray(info.get("detected_events", np.zeros(n, dtype=bool)), dtype=bool)
    final = np.asarray(info.get("final_actions", actions), dtype=int)

    rew = np.zeros(n, dtype=np.float32)
    # Soft budget: encourage mean TX rate near target among available
    avail_n = max(int(local.sum()), 1)
    tx_rate = float((final[local] == TRANSMIT).mean()) if local.any() else 0.0
    budget_pen = -w_budget * abs(tx_rate - target_tx_rate)

    for i in range(n):
        if not local[i]:
            rew[i] = 0.0
            continue
        tx = float(final[i] == TRANSMIT)
        err = 0.0
        if np.isfinite(gt[i]) and np.isfinite(recon[i]) and not tx:
            err = abs(gt[i] - recon[i]) / 400.0
        elif np.isfinite(gt[i]) and np.isfinite(recon[i]) and tx:
            err = 0.0  # delivered truth
        aoi_n = float(aoi[i]) / max(max_aoi, 1.0) if np.isfinite(aoi[i]) else 0.0
        r = (
            -w_tx * tx
            - w_err * err
            - w_aoi * aoi_n
            - w_miss * float(missed[i])
            + w_event * float(detected[i])
            + budget_pen
        )
        rew[i] = r
    return rew


@dataclass
class BoostedMAPPOTrainer:
    cfg: dict[str, Any]
    device: str = "cpu"

    def __post_init__(self) -> None:
        m = self.cfg.get("mappo", {})
        r = self.cfg.get("reward", {})
        b = self.cfg.get("boosted", {})
        self.lr = float(m.get("lr", 3e-4))
        self.gamma = float(m.get("gamma", 0.99))
        self.gae_lambda = float(m.get("gae_lambda", 0.95))
        self.clip_range = float(m.get("clip_range", 0.2))
        self.ent_coef = float(m.get("ent_coef", 0.05))
        self.vf_coef = float(m.get("vf_coef", 0.5))
        self.n_steps = int(m.get("n_steps", self.cfg.get("ppo", {}).get("n_steps", 1024)))
        self.batch_size = int(m.get("batch_size", self.cfg.get("ppo", {}).get("batch_size", 512)))
        self.n_epochs = int(m.get("n_epochs", self.cfg.get("ppo", {}).get("n_epochs", 4)))
        self.total_timesteps = int(m.get("total_timesteps", 150000))
        self.bc_steps = int(b.get("bc_steps", 20000))
        self.bc_lr = float(b.get("bc_lr", 1e-3))
        self.expert_mix = float(b.get("expert_mix_start", 0.5))
        self.expert_mix_end = float(b.get("expert_mix_end", 0.05))
        self.target_tx_rate = float(b.get("target_tx_rate", 0.28))
        self.w_tx = float(r.get("w_tx", 0.35))
        self.w_err = float(r.get("w_err", 2.0))
        self.w_aoi = float(r.get("w_aoi", 1.0))
        self.w_miss = float(r.get("w_miss", 8.0))
        self.w_event = float(r.get("w_event", 4.0))
        self.w_budget = float(r.get("w_budget", 1.5))
        self.eval_every = int(self.cfg.get("training", {}).get("eval_every_steps", 10000))
        self.max_aoi = float(self.cfg.get("env", {}).get("max_aoi", 8))

    def make_env(self, split: str = "train") -> TraceDrivenCampusEnv:
        cfg = dict(self.cfg)
        env_cfg = dict(cfg.get("environment", {}))
        env_cfg["graph_fail_fast"] = True
        env_cfg.setdefault("cohort", "final")
        env_cfg.setdefault("graph", "hybrid")
        cfg["environment"] = env_cfg
        # Critical: train WITHOUT shield so the policy owns TX decisions
        sensors = load_cohort(str(env_cfg["cohort"]))
        return TraceDrivenCampusEnv(
            cfg=cfg,
            split=split,
            sensor_ids=sensors,
            multi_agent=True,
        )

    def _bc_pretrain(self, actor: ResidualSharedActor, env: TraceDrivenCampusEnv) -> list[dict]:
        expert = SemanticExpertPolicy()
        opt = optim.Adam(actor.parameters(), lr=self.bc_lr)
        logs = []
        obs, _ = env.reset()
        steps = 0
        buf_obs: list[np.ndarray] = []
        buf_act: list[np.ndarray] = []
        while steps < self.bc_steps:
            local = env.local_available[env._t]
            expert_a = expert.act(obs, local_available=local)
            buf_obs.append(obs.copy())
            buf_act.append(expert_a.copy())
            # Roll env with expert so state distribution matches deployment
            obs, _, term, trunc, _ = env.step(expert_a)
            steps += 1
            if term or trunc:
                obs, _ = env.reset()
            if len(buf_obs) >= 256 or steps >= self.bc_steps:
                x = torch.as_tensor(np.asarray(buf_obs), dtype=torch.float32, device=self.device)
                y = torch.as_tensor(np.asarray(buf_act), dtype=torch.float32, device=self.device)
                # flatten agents
                t, n, d = x.shape
                logits = actor(x.reshape(t * n, d))
                loss = F.binary_cross_entropy_with_logits(logits, y.reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                logs.append({"bc_step": steps, "bc_loss": float(loss.item())})
                if steps % 2000 == 0:
                    print(f"[bc] step={steps} loss={loss.item():.4f} alpha={float(actor.residual_scale):.3f}")
                buf_obs, buf_act = [], []
        return logs

    def train(self, *, checkpoint_dir: str | Path | None = None) -> dict[str, Any]:
        set_seed(int(self.cfg.get("seed", 42)))
        env = self.make_env("train")
        obs_dim = int(env.observation_space.shape[-1])
        n_agents = env.n_sensors
        actor = ResidualSharedActor(obs_dim).to(self.device)
        critic = MeanPoolCritic(obs_dim).to(self.device)

        print(f"[boosted] n_agents={n_agents} obs_dim={obs_dim} device={self.device}")
        print(f"[boosted] shield=OFF during train; BC steps={self.bc_steps}")
        bc_logs = self._bc_pretrain(actor, env)

        opt = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=self.lr)
        root = repo_root()
        ckpt_dir = ensure_dir(
            checkpoint_dir
            or root / "results" / "rl_final" / "mappo_boosted" / f"seed_{self.cfg.get('seed', 42)}"
        )
        metrics: list[dict[str, Any]] = list(bc_logs)
        global_step = 0
        obs, _ = env.reset()
        expert = SemanticExpertPolicy()
        best_val = -1e18
        best_path = ckpt_dir / "best_model.pt"

        def compute_gae(rewards, values, dones, last_val):
            adv = np.zeros_like(rewards, dtype=np.float64)
            last_gae = 0.0
            for t in reversed(range(len(rewards))):
                next_val = last_val if t == len(rewards) - 1 else values[t + 1]
                next_non_terminal = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
                adv[t] = last_gae
            return adv, adv + values

        while global_step < self.total_timesteps:
            frac = global_step / max(self.total_timesteps, 1)
            mix = self.expert_mix * (1 - frac) + self.expert_mix_end * frac
            obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
            # per-agent reward buffer for GAE at agent level
            agent_rew_buf: list[np.ndarray] = []

            for _ in range(self.n_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                logits = actor(obs_t)
                dist = Bernoulli(logits=logits)
                actions = dist.sample()
                logp = dist.log_prob(actions)
                value = critic(obs_t)

                act_np = actions.cpu().numpy().astype(int)
                local = env.local_available[env._t]
                # DAgger-style mix: sometimes follow expert during early RL
                if np.random.random() < mix:
                    act_np = expert.act(obs, local_available=local)
                    with torch.no_grad():
                        logp = Bernoulli(logits=logits).log_prob(
                            torch.as_tensor(act_np, dtype=torch.float32, device=self.device)
                        )

                next_obs, team_r, terminated, truncated, info = env.step(act_np)
                # Attach GT for per-agent err (internal env field)
                info = dict(info)
                info["ground_truth"] = env.ground_truth[env._t - 1] if env._t > 0 else env.ground_truth[0]
                info["aoi"] = env.server_state.aoi.copy() if env.server_state is not None else np.zeros(n_agents)
                agent_r = _per_agent_rewards(
                    info,
                    act_np,
                    w_tx=self.w_tx,
                    w_err=self.w_err,
                    w_aoi=self.w_aoi,
                    w_miss=self.w_miss,
                    w_event=self.w_event,
                    w_budget=self.w_budget,
                    target_tx_rate=self.target_tx_rate,
                    max_aoi=self.max_aoi,
                )
                done = terminated or truncated

                obs_buf.append(obs.copy())
                act_buf.append(act_np.copy())
                logp_buf.append(logp.detach().cpu().numpy())
                # team scalar for critic = mean agent reward
                rew_buf.append(float(agent_r.mean()))
                agent_rew_buf.append(agent_r.copy())
                val_buf.append(float(value.item()))
                done_buf.append(float(done))

                obs = next_obs
                global_step += 1
                if done:
                    obs, _ = env.reset()
                if global_step >= self.total_timesteps:
                    break

            with torch.no_grad():
                last_val = float(critic(torch.as_tensor(obs, dtype=torch.float32, device=self.device)).item())

            adv, ret = compute_gae(np.asarray(rew_buf), np.asarray(val_buf), np.asarray(done_buf), last_val)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            obs_arr = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=self.device)
            act_arr = torch.as_tensor(np.asarray(act_buf), dtype=torch.float32, device=self.device)
            old_logp = torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=self.device)
            ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

            t_steps, n_ag, obs_d = obs_arr.shape
            flat_obs = obs_arr.reshape(t_steps * n_ag, obs_d)
            flat_act = act_arr.reshape(-1)
            flat_logp_old = old_logp.reshape(-1)

            # Per-agent advantages from local rewards (better credit)
            agent_rew = np.asarray(agent_rew_buf)  # (T, N)
            # Broadcast team GAE shape onto agents, scaled by local reward deviation
            agent_adv = np.repeat(adv[:, None], n_ag, axis=1)
            local_center = agent_rew - agent_rew.mean(axis=1, keepdims=True)
            agent_adv = agent_adv + 0.5 * local_center
            agent_adv = agent_adv.reshape(-1)
            agent_adv = (agent_adv - agent_adv.mean()) / (agent_adv.std() + 1e-8)
            agent_adv_t = torch.as_tensor(agent_adv, dtype=torch.float32, device=self.device)

            idx = np.arange(t_steps * n_ag)
            ent_now = self.ent_coef * (1.0 - 0.7 * frac)  # anneal entropy
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
                    mb_steps = np.unique(mb // n_ag)
                    values = critic(obs_arr[mb_steps])
                    value_loss = ((ret_t[mb_steps] - values) ** 2).mean()
                    loss = policy_loss + self.vf_coef * value_loss - ent_now * entropy
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                    opt.step()

            mean_r = float(np.mean(rew_buf))
            tx_frac = float(np.mean(np.asarray(act_buf) == TRANSMIT))
            row = {
                "step": global_step,
                "mean_reward_per_step": mean_r,
                "rollout_tx_frac": tx_frac,
                "expert_mix": mix,
                "residual_scale": float(actor.residual_scale.detach().cpu()),
                "ent_coef": ent_now,
            }
            metrics.append(row)
            print(
                f"[boosted] step={global_step} mean_r={mean_r:.4f} tx_frac={tx_frac:.3f} "
                f"mix={mix:.2f} alpha={row['residual_scale']:.3f}"
            )

            # Periodic validation (no shield — measures true policy)
            if global_step % self.eval_every < self.n_steps or global_step >= self.total_timesteps:
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "cfg": self.cfg,
                        "critic_type": "mean_pool",
                        "actor_type": "residual_heuristic",
                        "obs_dim": obs_dim,
                        "n_agents_train": n_agents,
                    },
                    ckpt_dir / "final_model.pt",
                )
                try:
                    val_env = make_final_env(
                        split="val",
                        cfg=self.cfg,
                        multi_agent=True,
                    )
                    act_fn = load_mappo_policy(ckpt_dir / "final_model.pt", device=self.device)
                    # Monkey-patch loader can't know ResidualSharedActor — evaluate inline
                    def _act(o, *, local_available=None):
                        with torch.no_grad():
                            x = torch.as_tensor(o, dtype=torch.float32, device=self.device)
                            if x.ndim == 1:
                                x = x.reshape(n_agents, obs_dim)
                            logits = actor(x)
                            a = (torch.sigmoid(logits) > 0.5).long().cpu().numpy().reshape(-1)
                            if local_available is not None:
                                a = np.where(local_available, a, SKIP)
                            return a.astype(int)

                    vm = evaluate_policy(val_env, _act, max_steps=None, seed=int(self.cfg.get("seed", 42)))
                    row_v = {
                        "step": global_step,
                        "val_tx_reduction": vm["transmission_reduction_pct"],
                        "val_mae": vm["mae_skipped"],
                        "val_recall": vm["event_recall"],
                        "val_f1": vm["event_f1"],
                        "val_aoi": vm["mean_aoi"],
                        "val_events": vm["n_true_events"],
                    }
                    metrics.append(row_v)
                    print(
                        f"[boosted] VAL red={vm['transmission_reduction_pct']:.1f}% "
                        f"mae={vm['mae_skipped']:.2f} recall={vm['event_recall']:.3f} "
                        f"aoi={vm['mean_aoi']:.2f}"
                    )
                    # Score: high reduction + recall + low mae
                    score = (
                        vm["transmission_reduction_pct"]
                        + 50.0 * (vm["event_recall"] if np.isfinite(vm["event_recall"]) else 0)
                        - 0.5 * (vm["mae_skipped"] if np.isfinite(vm["mae_skipped"]) else 50)
                    )
                    if score > best_val:
                        best_val = score
                        torch.save(
                            {
                                "actor": actor.state_dict(),
                                "critic": critic.state_dict(),
                                "cfg": self.cfg,
                                "critic_type": "mean_pool",
                                "actor_type": "residual_heuristic",
                                "obs_dim": obs_dim,
                                "n_agents_train": n_agents,
                                "val": row_v,
                            },
                            best_path,
                        )
                        print(f"[boosted] new best → {best_path}")
                except Exception as exc:
                    print(f"[boosted] val eval failed: {exc}")

        # Ensure residual actor is loadable via dedicated key
        final = {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "cfg": self.cfg,
            "critic_type": "mean_pool",
            "actor_type": "residual_heuristic",
            "obs_dim": obs_dim,
            "n_agents_train": n_agents,
        }
        torch.save(final, ckpt_dir / "final_model.pt")
        save_json({"metrics": metrics, "best_val_score": best_val}, ckpt_dir / "metrics.json")
        return {"checkpoint": str(ckpt_dir / "final_model.pt"), "best": str(best_path), "metrics": metrics}


def run_boosted_mappo(
    cfg: dict[str, Any] | None = None,
    *,
    device: str | None = None,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = repo_root()
    cfg = cfg or load_yaml(root / "configs" / "rl_learn.yaml")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[boosted] device={device}")
    trainer = BoostedMAPPOTrainer(cfg, device=device)
    out = trainer.train(checkpoint_dir=checkpoint_dir)
    out["device"] = device
    return out
