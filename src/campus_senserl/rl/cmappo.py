"""KL-regularized BC-initialized Constrained MAPPO.

Goal: improve BC toward expert-quality MAE/AoI while keeping high TX reduction.

L = E[C_tx] + Σ λ_k * hinge(J_k - ε_k) + β KL(π || π_BC)

- Dual variables have floors so AoI/MAE penalties cannot die
- Checkpoints saved ONLY when val constraints are feasible
- Among feasible, prefer lower MAE then higher TX reduction
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli

from campus_senserl.data.cohort import load_cohort
from campus_senserl.environment.communication_model import SKIP, TRANSMIT
from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv
from campus_senserl.evaluation.rl_policy_eval import evaluate_policy, make_final_env
from campus_senserl.rl.mappo_boosted import MeanPoolCritic, ResidualSharedActor
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json, set_seed


def bernoulli_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """KL(Bern(σ(p)) || Bern(σ(q))) mean over batch."""
    p = torch.sigmoid(logits_p).clamp(1e-6, 1 - 1e-6)
    q = torch.sigmoid(logits_q).clamp(1e-6, 1 - 1e-6)
    kl = p * (torch.log(p) - torch.log(q)) + (1 - p) * (torch.log(1 - p) - torch.log(1 - q))
    return kl.mean()


@dataclass
class ConstraintSpec:
    eps_miss: float = 0.015  # recall >= 0.985
    eps_mae: float = 9.0
    eps_aoi: float = 3.5
    lr_lambda: float = 0.08
    lambda_max: float = 25.0
    lambda_min_e: float = 0.5
    lambda_min_m: float = 1.0
    lambda_min_a: float = 1.5


class ConstrainedMAPPOTrainer:
    def __init__(self, cfg: dict[str, Any], device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        m = cfg.get("mappo", {})
        c = cfg.get("constraints", {})
        k = cfg.get("kl", {})
        self.lr = float(m.get("lr", 3e-5))
        self.gamma = float(m.get("gamma", 0.99))
        self.gae_lambda = float(m.get("gae_lambda", 0.95))
        self.clip_range = float(m.get("clip_range", 0.15))
        self.ent_coef = float(m.get("ent_coef", 0.01))
        self.vf_coef = float(m.get("vf_coef", 0.5))
        self.n_steps = int(m.get("n_steps", 1024))
        self.batch_size = int(m.get("batch_size", 512))
        self.n_epochs = int(m.get("n_epochs", 3))
        self.total_timesteps = int(m.get("total_timesteps", 60000))
        self.eval_every = int(cfg.get("training", {}).get("eval_every_steps", 8000))
        self.use_kl = bool(k.get("enabled", True))
        self.use_constraints = bool(c.get("enabled", True))
        self.kl_beta = float(k.get("beta", 0.05)) if self.use_kl else 0.0
        self.kl_beta_end = float(k.get("beta_end", 0.02)) if self.use_kl else 0.0
        self.freeze_residual = bool(k.get("freeze_residual_scale", True))
        self.constraints = ConstraintSpec(
            eps_miss=float(c.get("eps_miss", 0.015)),
            eps_mae=float(c.get("eps_mae", 9.0)),
            eps_aoi=float(c.get("eps_aoi", 3.5)),
            lr_lambda=float(c.get("lr_lambda", 0.08)),
            lambda_max=float(c.get("lambda_max", 25.0)),
            lambda_min_e=float(c.get("lambda_min_e", 0.5)),
            lambda_min_m=float(c.get("lambda_min_m", 1.0)),
            lambda_min_a=float(c.get("lambda_min_a", 1.5)),
        )
        if self.use_constraints:
            self.lam_e = max(1.0, self.constraints.lambda_min_e)
            self.lam_m = max(2.0, self.constraints.lambda_min_m)
            self.lam_a = max(2.0, self.constraints.lambda_min_a)
        else:
            self.lam_e = self.lam_m = self.lam_a = 0.0
            # Disable dual floors when constraints are ablated
            self.constraints.lambda_min_e = 0.0
            self.constraints.lambda_min_m = 0.0
            self.constraints.lambda_min_a = 0.0
            self.constraints.lr_lambda = 0.0

    def make_env(self, split: str = "train") -> TraceDrivenCampusEnv:
        cfg = dict(self.cfg)
        env_cfg = dict(cfg.get("environment", {}))
        env_cfg["graph_fail_fast"] = True
        env_cfg.setdefault("cohort", "final")
        env_cfg.setdefault("graph", "hybrid")
        cfg["environment"] = env_cfg
        sensors = load_cohort(str(env_cfg["cohort"]))
        return TraceDrivenCampusEnv(cfg=cfg, split=split, sensor_ids=sensors, multi_agent=True)

    def _step_costs(self, info: dict[str, Any], actions: np.ndarray) -> dict[str, float]:
        local = np.asarray(info["local_available"], dtype=bool)
        final = np.asarray(info.get("final_actions", actions), dtype=int)
        gt = np.asarray(info.get("ground_truth", np.full(len(actions), np.nan)), dtype=float)
        recon = np.asarray(info["reconstruction"], dtype=float)
        aoi_raw = np.asarray(info.get("aoi_raw", info.get("aoi_state", np.zeros(len(actions)))), dtype=float)
        missed = np.asarray(info.get("missed_events", np.zeros(len(actions), dtype=bool)), dtype=bool)
        true_e = np.asarray(info.get("true_events", np.zeros(len(actions), dtype=bool)), dtype=bool)

        c_tx = float((final[local] == TRANSMIT).mean()) if local.any() else 0.0
        n_te = int(true_e[local].sum()) if local.any() else 0
        j_miss = float(missed[local].sum() / n_te) if n_te > 0 else 0.0
        errs = []
        for i in range(len(actions)):
            if not local[i] or not np.isfinite(gt[i]) or not np.isfinite(recon[i]):
                continue
            if int(final[i]) != TRANSMIT:
                errs.append(abs(gt[i] - recon[i]))
        j_mae = float(np.mean(errs)) if errs else 0.0
        j_aoi = float(aoi_raw[local].mean()) if local.any() else 0.0
        return {"c_tx": c_tx, "j_miss": j_miss, "j_mae": j_mae, "j_aoi": j_aoi}

    def _clip_lams(self) -> None:
        c = self.constraints
        self.lam_e = float(np.clip(self.lam_e, c.lambda_min_e, c.lambda_max))
        self.lam_m = float(np.clip(self.lam_m, c.lambda_min_m, c.lambda_max))
        self.lam_a = float(np.clip(self.lam_a, c.lambda_min_a, c.lambda_max))

    def train(
        self,
        *,
        bc_checkpoint: str | Path,
        checkpoint_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        set_seed(int(self.cfg.get("seed", 42)))
        env = self.make_env("train")
        obs_dim = int(env.observation_space.shape[-1])
        n_agents = env.n_sensors

        actor = ResidualSharedActor(obs_dim).to(self.device)
        critic = MeanPoolCritic(obs_dim).to(self.device)
        bc_actor = ResidualSharedActor(obs_dim).to(self.device)

        ckpt = torch.load(bc_checkpoint, map_location=self.device, weights_only=False)
        actor.load_state_dict(ckpt["actor"])
        bc_actor.load_state_dict(ckpt["actor"])
        bc_actor.eval()
        for p in bc_actor.parameters():
            p.requires_grad_(False)
        if "critic" in ckpt:
            try:
                critic.load_state_dict(ckpt["critic"])
            except Exception:
                pass

        if self.freeze_residual:
            actor.residual_scale.requires_grad_(False)

        params = [p for p in list(actor.parameters()) + list(critic.parameters()) if p.requires_grad]
        opt = optim.Adam(params, lr=self.lr)

        root = repo_root()
        ckpt_dir = ensure_dir(
            checkpoint_dir
            or root / "results" / "rl_final" / "cmappo_kl" / f"seed_{self.cfg.get('seed', 42)}"
        )
        metrics: list[dict[str, Any]] = []
        global_step = 0
        obs, _ = env.reset()
        best_key = None  # (mae, -reduction) lower better
        best_path = ckpt_dir / "best_model.pt"
        best_val: dict[str, Any] | None = None

        print(f"[kl-cmappo] loaded BC {bc_checkpoint}")
        print(
            f"[kl-cmappo] use_kl={self.use_kl} use_constraints={self.use_constraints} "
            f"eps_miss={self.constraints.eps_miss} eps_mae={self.constraints.eps_mae} "
            f"eps_aoi={self.constraints.eps_aoi} beta={self.kl_beta}→{self.kl_beta_end} lr={self.lr}"
        )

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
            beta = self.kl_beta * (1 - frac) + self.kl_beta_end * frac
            obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
            cost_acc = {"c_tx": [], "j_miss": [], "j_mae": [], "j_aoi": []}

            for _ in range(self.n_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                logits = actor(obs_t)
                dist = Bernoulli(logits=logits)
                actions = dist.sample()
                logp = dist.log_prob(actions)
                value = critic(obs_t)
                act_np = actions.cpu().numpy().astype(int)

                next_obs, _, terminated, truncated, info = env.step(act_np)
                info = dict(info)
                info["ground_truth"] = env.ground_truth[env._t - 1] if env._t > 0 else env.ground_truth[0]
                costs = self._step_costs(info, act_np)

                if self.use_constraints:
                    viol_e = max(0.0, costs["j_miss"] - self.constraints.eps_miss)
                    viol_m = max(0.0, (costs["j_mae"] - self.constraints.eps_mae) / 10.0)
                    viol_a = max(0.0, (costs["j_aoi"] - self.constraints.eps_aoi) / 4.0)
                    lag = costs["c_tx"] + self.lam_e * viol_e + self.lam_m * viol_m + self.lam_a * viol_a
                else:
                    # Unconstrained TX-minimizing reward (ordinary MAPPO / KL-only ablations)
                    lag = costs["c_tx"]
                reward = -float(lag)

                done = terminated or truncated
                obs_buf.append(obs.copy())
                act_buf.append(act_np.copy())
                logp_buf.append(logp.detach().cpu().numpy())
                rew_buf.append(reward)
                val_buf.append(float(value.item()))
                done_buf.append(float(done))
                for k in cost_acc:
                    cost_acc[k].append(costs[k])

                obs = next_obs
                global_step += 1
                if done:
                    obs, _ = env.reset()
                if global_step >= self.total_timesteps:
                    break

            mean_miss = float(np.mean(cost_acc["j_miss"]))
            mean_mae = float(np.mean(cost_acc["j_mae"]))
            mean_aoi = float(np.mean(cost_acc["j_aoi"]))
            if self.use_constraints:
                self.lam_e += self.constraints.lr_lambda * (mean_miss - self.constraints.eps_miss)
                self.lam_m += self.constraints.lr_lambda * ((mean_mae - self.constraints.eps_mae) / 10.0)
                self.lam_a += self.constraints.lr_lambda * ((mean_aoi - self.constraints.eps_aoi) / 4.0)
                self._clip_lams()
            else:
                self.lam_e = self.lam_m = self.lam_a = 0.0

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
            agent_adv = np.repeat(adv, n_ag)
            agent_adv = (agent_adv - agent_adv.mean()) / (agent_adv.std() + 1e-8)
            agent_adv_t = torch.as_tensor(agent_adv, dtype=torch.float32, device=self.device)

            idx = np.arange(t_steps * n_ag)
            for _ in range(self.n_epochs):
                np.random.shuffle(idx)
                for start in range(0, len(idx), self.batch_size):
                    mb = idx[start : start + self.batch_size]
                    logits = actor(flat_obs[mb])
                    with torch.no_grad():
                        bc_logits = bc_actor(flat_obs[mb])
                    dist = Bernoulli(logits=logits)
                    logp = dist.log_prob(flat_act[mb])
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(logp - flat_logp_old[mb])
                    pg1 = ratio * agent_adv_t[mb]
                    pg2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * agent_adv_t[mb]
                    policy_loss = -torch.min(pg1, pg2).mean()
                    kl = bernoulli_kl(logits, bc_logits)
                    mb_steps = np.unique(mb // n_ag)
                    values = critic(obs_arr[mb_steps])
                    value_loss = ((ret_t[mb_steps] - values) ** 2).mean()
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy + beta * kl
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(params, 0.5)
                    opt.step()

            row = {
                "step": global_step,
                "mean_tx": float(np.mean(cost_acc["c_tx"])),
                "mean_miss": mean_miss,
                "mean_mae": mean_mae,
                "mean_aoi": mean_aoi,
                "lam_e": self.lam_e,
                "lam_m": self.lam_m,
                "lam_a": self.lam_a,
                "kl_beta": beta,
            }
            metrics.append(row)
            print(
                f"[kl-cmappo] step={global_step} tx={row['mean_tx']:.3f} miss={mean_miss:.3f} "
                f"mae={mean_mae:.2f} aoi={mean_aoi:.2f} "
                f"lam=({self.lam_e:.2f},{self.lam_m:.2f},{self.lam_a:.2f}) beta={beta:.3f}"
            )

            if global_step % self.eval_every < self.n_steps or global_step >= self.total_timesteps:
                payload = {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "cfg": self.cfg,
                    "critic_type": "mean_pool",
                    "actor_type": "residual_heuristic",
                    "obs_dim": obs_dim,
                    "n_agents_train": n_agents,
                    "method": "cmappo_kl",
                    "lambdas": {"e": self.lam_e, "m": self.lam_m, "a": self.lam_a},
                }
                torch.save(payload, ckpt_dir / "final_model.pt")

                def _act(o, *, local_available=None, _actor=actor, _n=n_agents, _d=obs_dim):
                    with torch.no_grad():
                        x = torch.as_tensor(o, dtype=torch.float32, device=self.device)
                        if x.ndim == 1:
                            x = x.reshape(_n, _d)
                        a = (torch.sigmoid(_actor(x)) > 0.5).long().cpu().numpy().reshape(-1)
                        if local_available is not None:
                            a = np.where(local_available, a, SKIP)
                        return a.astype(int)

                try:
                    val_env = make_final_env(
                        split="val", cfg=self.cfg, multi_agent=True
                    )
                    vm = evaluate_policy(val_env, _act, max_steps=None, seed=int(self.cfg.get("seed", 42)))
                    aoi = float(vm.get("mean_aoi_raw", vm["mean_aoi"]))
                    mae = float(vm["mae_skipped"]) if np.isfinite(vm["mae_skipped"]) else 99.0
                    rec = float(vm["event_recall"]) if np.isfinite(vm["event_recall"]) else 0.0
                    red = float(vm["transmission_reduction_pct"])
                    if self.use_constraints:
                        feasible = (
                            rec >= 1.0 - self.constraints.eps_miss - 1e-6
                            and mae <= self.constraints.eps_mae + 0.05
                            and aoi <= self.constraints.eps_aoi + 0.05
                            and red >= 70.0
                        )
                    else:
                        # Ablations without constraints: keep best MAE among reasonable TX
                        feasible = red >= 50.0 and rec >= 0.90
                    vrow = {
                        "step": global_step,
                        "val_tx_reduction": red,
                        "val_mae": mae,
                        "val_recall": rec,
                        "val_precision": vm["event_precision"],
                        "val_f1": vm["event_f1"],
                        "val_aoi_raw": aoi,
                        "feasible": feasible,
                        "fp": vm["fp"],
                    }
                    metrics.append(vrow)
                    print(
                        f"[kl-cmappo] VAL red={red:.1f}% mae={mae:.2f} rec={rec:.4f} "
                        f"aoi_raw={aoi:.2f} feasible={feasible} fp={vm['fp']}"
                    )
                    if feasible:
                        # Prefer lower MAE, then higher reduction
                        key = (mae, -red)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_val = vrow
                            torch.save({**payload, "val": vrow}, best_path)
                            print(f"[kl-cmappo] FEASIBLE best → {best_path}")
                except Exception as exc:
                    print(f"[kl-cmappo] val failed: {exc}")

        # If never feasible, keep final but mark
        save_json(
            {
                "metrics": metrics,
                "best_val": best_val,
                "had_feasible": best_val is not None,
            },
            ckpt_dir / "metrics.json",
        )
        return {
            "checkpoint": str(ckpt_dir / "final_model.pt"),
            "best": str(best_path) if best_val is not None else str(ckpt_dir / "final_model.pt"),
            "best_val": best_val,
            "metrics": metrics,
        }


def run_constrained_mappo(
    cfg: dict[str, Any] | None = None,
    *,
    bc_checkpoint: str | Path,
    device: str | None = None,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = repo_root()
    cfg = cfg or load_yaml(root / "configs" / "rl_cmappo.yaml")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = ConstrainedMAPPOTrainer(cfg, device=device)
    out = trainer.train(bc_checkpoint=bc_checkpoint, checkpoint_dir=checkpoint_dir)
    out["device"] = device
    return out
