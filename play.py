import argparse
import time
import numpy as np
import torch

from environment import DroneEnv
from agent import ActorCritic

# Optional pygame (only used for event pump when render_mode="human")
try:
    import pygame
except ImportError:
    pygame = None


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def deterministic_action(policy: ActorCritic, obs: np.ndarray, device):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    # cooldown is obs[4] in your codebase
    can_shoot = 1.0 if obs[4] <= 1e-6 else 0.0
    shoot_mask = torch.tensor([[can_shoot]], dtype=torch.float32, device=device)

    mu_t, _, mu_s, _, shoot_logit, _ = policy.forward(obs_t)
    thrust = torch.sigmoid(mu_t)
    steer = torch.tanh(mu_s)
    shoot = (torch.sigmoid(shoot_logit) > 0.5).float() * shoot_mask

    action = torch.cat([thrust, steer, shoot], dim=-1)
    return action.squeeze(0).cpu().numpy().astype(np.float32)


@torch.no_grad()
def stochastic_action(policy: ActorCritic, obs: np.ndarray, device):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    can_shoot = 1.0 if obs[4] <= 1e-6 else 0.0
    shoot_mask = torch.tensor([[can_shoot]], dtype=torch.float32, device=device)
    action, _, _ = policy.act(obs_t, shoot_mask)
    return action.squeeze(0).cpu().numpy().astype(np.float32)


def term_code_to_text(code: int) -> str:
    return {
        0: "none",
        1: "wall",
        2: "obstacle",
        3: "friendly",
        4: "enemy",
        5: "timeout",
    }.get(int(code), f"code{code}")


def pump_pygame_events() -> bool:
    """
    Returns False if user requested quit, True otherwise.
    """
    if pygame is None:
        return True

    # If pygame isn't initialized by env yet, don't touch it.
    if not pygame.get_init():
        return True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/drone_ppo_weights.pt")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions (more chaotic)")
    parser.add_argument("--fps", type=int, default=60, help="Render FPS cap")
    args = parser.parse_args()

    device = pick_device()

    # Note: If your DroneEnv has a strict internal counter that stops physics
    # after 1000 steps, you might need to modify DroneEnv as well.
    # But usually, just ignoring the reset here works.
    env = DroneEnv(render_mode="human", seed=args.seed)
    obs_dim = int(env.observation_space.shape[0])

    policy = ActorCritic(obs_dim, hidden=128).to(device)
    try:
        policy.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")
    except FileNotFoundError:
        print(f"Error: Could not find weights file: {args.weights}")
        return

    policy.eval()

    obs, _ = env.reset()

    total_steps = 0
    segment = 1
    seg_return = 0.0

    print("Playing continuously (Infinite Horizon).")
    print(" - TIMEOUTS are ignored (game continues)")
    print(" - Resets only on CRASH/DEATH")
    print("Press ESC/Q or close the window to quit.")

    # show first frame immediately
    try:
        env.render()
    except Exception:
        pass

    dt = 1.0 / max(1, int(args.fps))
    next_t = time.perf_counter()

    running = True
    while running:
        # pygame event pump (prevents 'not responding' + allows quitting)
        if not pump_pygame_events():
            break

        act = (
            stochastic_action(policy, obs, device)
            if args.stochastic
            else deterministic_action(policy, obs, device)
        )

        obs, r, terminated, truncated, info = env.step(act)
        seg_return += float(r)
        total_steps += 1

        # render every step
        try:
            env.render()
        except Exception:
            pass

        # --- CHANGE IS HERE ---
        # We simply ignore 'truncated' (timeout).
        # We only print a log if it happens, but we DO NOT reset.
        if truncated and not terminated:
            # Optional: Uncomment the line below if you want to see when 1000 steps pass
            # print(f"[Info] Passed step limit at {total_steps}, continuing...")
            pass 

        # If crash/terminated: reset (DON'T break) so window stays open
        if terminated:
            code = info.get("term_code", 0) if isinstance(info, dict) else 0
            print(
                f"[segment {segment}] CRASH reset at {total_steps} steps | "
                f"return={seg_return:.2f} | reason={term_code_to_text(code)}"
            )
            obs, _ = env.reset()
            segment += 1
            seg_return = 0.0

        # FPS cap
        now = time.perf_counter()
        if now < next_t:
            time.sleep(next_t - now)
        next_t = max(next_t + dt, time.perf_counter())

    env.close()


if __name__ == "__main__":
    main()