import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Generator, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.vector import SyncVectorEnv

from environment import DroneEnv
from agent import ActorCritic


@dataclass
class PPOConfig:
    """
    Configuration parameters for PPO training.
    """
    seed: int = 1
    total_timesteps: int = 200_000
    num_envs: int = 8
    num_steps: int = 128

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4

    update_epochs: int = 8
    minibatch_size: int = 512
    hidden_size: int = 128

    eval_every_updates: int = 10
    eval_episodes: int = 8
    device: str = "auto"


class RolloutBuffer:
    """
    Buffer to store transitions for PPO updates.
    """

    def __init__(self, obs_dim: int, num_steps: int, num_envs: int, device: torch.device):
        """
        Initialize the rollout buffer.

        Args:
            obs_dim (int): Dimension of observations.
            num_steps (int): Number of steps per environment per rollout.
            num_envs (int): Number of parallel environments.
            device (torch.device): Computation device.
        """
        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, 3), device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.shoot_mask = torch.zeros((num_steps, num_envs, 1), device=device)

        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)
        self.ptr = 0
        self.T = num_steps
        self.N = num_envs

    def add(self, obs: torch.Tensor, actions: torch.Tensor, logprobs: torch.Tensor, 
            rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, shoot_mask: torch.Tensor):
        """
        Add a transition to the buffer.
        """
        t = self.ptr
        self.obs[t].copy_(obs)
        self.actions[t].copy_(actions)
        self.logprobs[t].copy_(logprobs)
        self.rewards[t].copy_(rewards)
        self.dones[t].copy_(dones)
        self.values[t].copy_(values)
        self.shoot_mask[t].copy_(shoot_mask)
        self.ptr += 1

    def compute_returns_advantages(self, last_value: torch.Tensor, gamma: float, lam: float):
        """
        Compute GAE (Generalized Advantage Estimation).

        Args:
            last_value (torch.Tensor): Value estimate of the next state after rollout.
            gamma (float): Discount factor.
            lam (float): GAE smoothing parameter.
        """
        last_gae = torch.zeros((self.N,), device=self.obs.device)
        for t in reversed(range(self.T)):
            next_nonterminal = 1.0 - self.dones[t]
            next_value = last_value if (t == self.T - 1) else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values
        adv = self.advantages
        self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

    def minibatches(self, mb_size: int) -> Generator:
        """
        Yield minibatches of data for PPO updates.

        Args:
            mb_size (int): Size of each minibatch.

        Yields:
            tuple: Tensors for obs, actions, logprobs, advantages, returns, values, and mask.
        """
        T, N = self.T, self.N
        B = T * N

        obs = self.obs.reshape(B, -1)
        actions = self.actions.reshape(B, -1)
        logprobs = self.logprobs.reshape(B)
        adv = self.advantages.reshape(B)
        rets = self.returns.reshape(B)
        vals = self.values.reshape(B)
        shoot_mask = self.shoot_mask.reshape(B, 1)

        idx = torch.randperm(B, device=obs.device)
        for start in range(0, B, mb_size):
            mb = idx[start:start + mb_size]
            yield obs[mb], actions[mb], logprobs[mb], adv[mb], rets[mb], vals[mb], shoot_mask[mb]


def make_env(seed: int, idx: int):
    """
    Factory function for creating environment instances.
    """
    def thunk():
        return DroneEnv(render_mode=None, seed=seed + idx)
    return thunk


def pick_device(device_pref: str = "auto") -> torch.device:
    """
    Select the computation device.

    Args:
        device_pref (str): Preference ("auto", "cpu", "cuda", "mps").

    Returns:
        torch.device: The selected device.
    """
    if device_pref != "auto":
        return torch.device(device_pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def csv_write_row(path: str, row: Dict[str, Any]):
    """Append a row to a CSV file."""
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def save_checkpoint(path: str, payload: Dict[str, Any]):
    """Save training checkpoint."""
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    """Load training checkpoint."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def act_deterministic(policy: ActorCritic, obs_t: torch.Tensor, shoot_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get deterministic actions for evaluation.

    Args:
        policy (ActorCritic): The policy network.
        obs_t (torch.Tensor): Observation tensor.
        shoot_mask (torch.Tensor): Shoot availability mask.

    Returns:
        tuple: Action tensor and value tensor.
    """
    mu_t, _, mu_s, _, shoot_logit, value = policy.forward(obs_t)
    thrust = torch.sigmoid(mu_t)
    steer = torch.tanh(mu_s)
    shoot = (torch.sigmoid(shoot_logit) > 0.5).float() * shoot_mask
    action = torch.cat([thrust, steer, shoot], dim=-1)
    return action, value


@torch.no_grad()
def evaluate(policy: ActorCritic, device: torch.device, episodes: int = 8) -> Tuple[float, float, float, List[int]]:
    """
    Run evaluation episodes.

    Args:
        policy (ActorCritic): The policy network.
        device (torch.device): Computation device.
        episodes (int): Number of episodes to run.

    Returns:
        tuple: Mean score, std score, mean length, list of termination codes.
    """
    env = DroneEnv(render_mode=None, seed=999)
    scores = []
    lengths = []
    term_codes = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        total = 0.0
        steps = 0
        last_code = 0

        while not (done or trunc):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            can_shoot = 1.0 if obs[4] <= 1e-6 else 0.0
            shoot_mask = torch.tensor([[can_shoot]], dtype=torch.float32, device=device)

            action, _ = act_deterministic(policy, obs_t, shoot_mask)
            obs, r, done, trunc, info = env.step(action.squeeze(0).cpu().numpy().astype(np.float32))
            total += float(r)
            steps += 1
            if isinstance(info, dict) and "term_code" in info:
                last_code = int(info["term_code"])

        scores.append(total)
        lengths.append(steps)
        term_codes.append(last_code)

    env.close()
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(lengths)), term_codes


def plot_from_csv(csv_path: str, out_png_path: str):
    """
    Generate training plots from CSV log.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return

    upd = np.array([int(r["update"]) for r in rows])
    sps = np.array([float(r["sps"]) for r in rows])

    ep_ema = np.array([float(r["ep_return_ema"]) for r in rows])
    ep_len_ema = np.array([float(r["ep_len_ema"]) for r in rows])

    eval_mean = np.array([float(r["eval_mean"]) if r["eval_mean"] else np.nan for r in rows])
    best_eval = np.array([float(r["best_eval"]) for r in rows])

    wall = np.array([float(r["term_wall"]) for r in rows])
    obst = np.array([float(r["term_obstacle"]) for r in rows])
    frnd = np.array([float(r["term_friendly"]) for r in rows])
    enem = np.array([float(r["term_enemy"]) for r in rows])
    tout = np.array([float(r["term_timeout"]) for r in rows])

    hit_rate = np.array([float(r["shot_hit_rate"]) for r in rows])

    fig = plt.figure(figsize=(10, 10))

    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(upd, ep_ema, label="Train episode return (EMA)")
    ax1.plot(upd, eval_mean, label="Eval return (deterministic)")
    ax1.plot(upd, best_eval, label="Best eval so far")
    ax1.set_ylabel("Return")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(upd, ep_len_ema, label="Train episode length (EMA)")
    ax2.set_ylabel("Ep length")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(upd, wall, label="wall")
    ax3.plot(upd, obst, label="obstacle")
    ax3.plot(upd, frnd, label="friendly")
    ax3.plot(upd, enem, label="enemy")
    ax3.plot(upd, tout, label="timeout")
    ax3.set_ylabel("Ends/update")
    ax3.legend(ncol=3)
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(upd, sps, label="SPS")
    ax4.plot(upd, hit_rate, label="Shot hit rate")
    ax4.set_xlabel("Update")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=170)
    plt.close(fig)


def main():
    """
    Main training loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=150_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default="drone_ppo")

    parser.add_argument("--resume", type=str, default="", help="Path to *_ckpt.pt to resume")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|mps|cuda")
    args = parser.parse_args()

    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.steps,
        lr=args.lr,
        hidden_size=args.hidden,
        seed=args.seed,
        device=args.device,
    )

    ensure_dir(args.run_dir)
    weights_path = os.path.join(args.run_dir, f"{args.run_name}_weights.pt")
    ckpt_path    = os.path.join(args.run_dir, f"{args.run_name}_ckpt.pt")
    log_path     = os.path.join(args.run_dir, f"{args.run_name}_log.csv")
    plot_path    = os.path.join(args.run_dir, f"{args.run_name}_curves.png")

    device = pick_device(cfg.device)
    print("Device:", device)

    tmp = DroneEnv(render_mode=None, seed=cfg.seed)
    obs_dim = int(tmp.observation_space.shape[0])
    tmp.close()
    print("Obs dim:", obs_dim)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    policy = ActorCritic(obs_dim, hidden=cfg.hidden_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)

    start_update = 1
    total_seen_steps = 0
    best_eval = -1e18

    if args.resume:
        ckpt = load_checkpoint(args.resume, device)
        policy.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_update = int(ckpt["update"]) + 1
        total_seen_steps = int(ckpt["total_seen_steps"])
        best_eval = float(ckpt.get("best_eval", best_eval))
        print(f"Resumed from {args.resume} at update {start_update} (steps seen={total_seen_steps})")

    envs = SyncVectorEnv([make_env(cfg.seed, i) for i in range(cfg.num_envs)])
    obs, _ = envs.reset(seed=cfg.seed)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    buffer = RolloutBuffer(obs_dim, cfg.num_steps, cfg.num_envs, device=device)

    batch_size = cfg.num_envs * cfg.num_steps
    num_updates = cfg.total_timesteps // batch_size
    print(f"Batch/update: {batch_size} | Target updates: {num_updates} | Starting at: {start_update}")

    ep_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(cfg.num_envs, dtype=np.int32)

    ema_return = -100.0
    ema_len = 200.0
    ema_beta = 0.90

    start_time = time.time()

    for update in range(start_update, num_updates + 1):
        buffer.ptr = 0

        ended_eps_returns: List[float] = []
        ended_eps_lengths: List[int] = []

        term_counts = np.zeros(6, dtype=np.int32)
        shots_fired = 0
        shots_hit = 0
        shots_hit_friendly = 0
        shots_missed = 0

        for _ in range(cfg.num_steps):
            cooldown_norm = obs_t[:, 4:5]
            shoot_mask = (cooldown_norm <= 1e-6).float()

            with torch.no_grad():
                actions, logp, values = policy.act(obs_t, shoot_mask)

            next_obs, rewards, terms, truncs, infos = envs.step(actions.cpu().numpy().astype(np.float32))
            dones = np.logical_or(terms, truncs)

            if isinstance(infos, dict):
                tc = infos.get("term_code", None)
                if tc is not None:
                    tc = np.asarray(tc, dtype=np.int32)
                    for code in tc:
                        if 0 <= int(code) <= 5:
                            term_counts[int(code)] += 1

                shots_fired += int(np.sum(np.asarray(infos.get("shot_fired", 0), dtype=np.int32)))
                shots_hit += int(np.sum(np.asarray(infos.get("shot_hit", 0), dtype=np.int32)))
                shots_hit_friendly += int(np.sum(np.asarray(infos.get("shot_hit_friendly", 0), dtype=np.int32)))
                shots_missed += int(np.sum(np.asarray(infos.get("shot_missed", 0), dtype=np.int32)))

            ep_returns += rewards.astype(np.float32)
            ep_lengths += 1
            if np.any(dones):
                done_idx = np.where(dones)[0]
                for i in done_idx:
                    ended_eps_returns.append(float(ep_returns[i]))
                    ended_eps_lengths.append(int(ep_lengths[i]))
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0

            buffer.add(
                obs=obs_t,
                actions=actions,
                logprobs=logp,
                rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
                dones=torch.tensor(dones.astype(np.float32), dtype=torch.float32, device=device),
                values=values,
                shoot_mask=shoot_mask,
            )

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            last_value = policy.get_value(obs_t)

        buffer.compute_returns_advantages(last_value, cfg.gamma, cfg.gae_lambda)

        policy.train()
        mb_count = max(1, batch_size // cfg.minibatch_size)

        approx_kl_sum = 0.0
        clipfrac_sum = 0.0
        ent_sum = 0.0
        pg_sum = 0.0
        v_sum = 0.0

        for _ in range(cfg.update_epochs):
            for mb_obs, mb_actions, mb_logp_old, mb_adv, mb_rets, mb_vals_old, mb_shoot_mask in buffer.minibatches(cfg.minibatch_size):
                new_logp, entropy, new_value = policy.evaluate_actions(mb_obs, mb_actions, mb_shoot_mask)

                log_ratio = new_logp - mb_logp_old
                ratio = torch.exp(log_ratio)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                v_pred = new_value
                v_pred_clip = mb_vals_old + torch.clamp(v_pred - mb_vals_old, -cfg.clip_coef, cfg.clip_coef)
                v_loss = 0.5 * torch.max((v_pred - mb_rets) ** 2, (v_pred_clip - mb_rets) ** 2).mean()

                ent = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = (ratio - 1.0 - log_ratio).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()

                approx_kl_sum += approx_kl
                clipfrac_sum += clipfrac
                ent_sum += ent.item()
                pg_sum += pg_loss.item()
                v_sum += v_loss.item()

        total_seen_steps += batch_size
        sps = int(total_seen_steps / (time.time() - start_time + 1e-8))
        denom = cfg.update_epochs * mb_count

        mean_step_r = float(buffer.rewards.mean().item())
        ep_count = len(ended_eps_returns)

        if ep_count > 0:
            mean_ep_r = float(np.mean(ended_eps_returns))
            mean_ep_len = float(np.mean(ended_eps_lengths))
            ema_return = ema_beta * ema_return + (1 - ema_beta) * mean_ep_r
            ema_len = ema_beta * ema_len + (1 - ema_beta) * mean_ep_len

        shot_hit_rate = (shots_hit / max(1, shots_fired))
        friendly_rate = (shots_hit_friendly / max(1, shots_fired))

        eval_mean = ""
        eval_std = ""
        eval_len = ""
        if (update % cfg.eval_every_updates) == 0 or update == num_updates:
            policy.eval()
            m, s, L, codes = evaluate(policy, device, episodes=cfg.eval_episodes)
            eval_mean = f"{m:.6f}"
            eval_std = f"{s:.6f}"
            eval_len = f"{L:.2f}"
            best_eval = max(best_eval, m)

            torch.save(policy.state_dict(), weights_path)
            save_checkpoint(ckpt_path, {
                "model_state": policy.state_dict(),
                "optim_state": optimizer.state_dict(),
                "update": update,
                "total_seen_steps": total_seen_steps,
                "best_eval": best_eval,
            })

        row = {
            "update": update,
            "seen_steps": total_seen_steps,
            "sps": sps,
            "mean_step_reward": f"{mean_step_r:.6f}",
            "episodes_ended": ep_count,
            "ep_return_ema": f"{ema_return:.6f}",
            "ep_len_ema": f"{ema_len:.6f}",

            "term_wall": int(term_counts[1]),
            "term_obstacle": int(term_counts[2]),
            "term_friendly": int(term_counts[3]),
            "term_enemy": int(term_counts[4]),
            "term_timeout": int(term_counts[5]),

            "shots_fired": shots_fired,
            "shots_hit": shots_hit,
            "shots_hit_friendly": shots_hit_friendly,
            "shots_missed": shots_missed,
            "shot_hit_rate": f"{shot_hit_rate:.6f}",
            "friendly_fire_rate": f"{friendly_rate:.6f}",

            "kl": f"{(approx_kl_sum/denom):.6f}",
            "entropy": f"{(ent_sum/denom):.6f}",
            "pg_loss": f"{(pg_sum/denom):.6f}",
            "v_loss": f"{(v_sum/denom):.6f}",

            "eval_mean": eval_mean,
            "eval_std": eval_std,
            "eval_len": eval_len,
            "best_eval": f"{best_eval:.6f}",
        }
        csv_write_row(log_path, row)

        term_str = f"W{term_counts[1]} O{term_counts[2]} F{term_counts[3]} E{term_counts[4]} T{term_counts[5]}"
        shot_str = f"shots {shots_fired} hit {shots_hit} ff {shots_hit_friendly} hr {shot_hit_rate:.2f}"
        eval_str = f" eval {float(eval_mean):+.1f}±{float(eval_std):.1f} len {float(eval_len):.0f} best {best_eval:+.1f}" if eval_mean else ""
        print(
            f"upd {update:03d}/{num_updates} | steps {total_seen_steps:7d} | sps {sps:4d} | "
            f"step_r {mean_step_r:+.3f} | eps {ep_count:2d} emaR {ema_return:+.1f} emaL {ema_len:5.0f} | "
            f"ends {term_str} | {shot_str} | kl {(approx_kl_sum/denom):.4f}{eval_str}"
        )

    envs.close()

    try:
        plot_from_csv(log_path, plot_path)
        print("Saved plot:", plot_path)
    except Exception as e:
        print("Plotting failed:", e)

    print("Done.")
    print("Weights:", weights_path)
    print("Checkpoint:", ckpt_path)
    print("Log CSV:", log_path)


if __name__ == "__main__":
    main()