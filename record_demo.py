import argparse
import time
import numpy as np
import torch
import pygame
import imageio

from environment import DroneEnv
from agent import ActorCritic

# --- Config ---
FPS = 60
OUTPUT_VIDEO = "agent_gameplay.mp4"

def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def get_action(policy, obs, device, stochastic=False):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    can_shoot = 1.0 if obs[4] <= 1e-6 else 0.0
    shoot_mask = torch.tensor([[can_shoot]], dtype=torch.float32, device=device)

    if stochastic:
        action, _, _ = policy.act(obs_t, shoot_mask)
    else:
        mu_t, _, mu_s, _, shoot_logit, _ = policy.forward(obs_t)
        thrust = torch.sigmoid(mu_t)
        steer = torch.tanh(mu_s)
        shoot = (torch.sigmoid(shoot_logit) > 0.5).float() * shoot_mask
        action = torch.cat([thrust, steer, shoot], dim=-1)

    return action.squeeze(0).cpu().numpy().astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/drone_ppo_weights.pt")
    parser.add_argument("--record", action="store_true", help="Save video to mp4")
    parser.add_argument("--no-lidar", action="store_true", help="Hide LiDAR rays")
    parser.add_argument("--stochastic", action="store_true", help="Use random sampling")
    parser.add_argument("--time-limit", type=int, default=30, help="Seconds to record")
    args = parser.parse_args()

    device = pick_device()
    env = DroneEnv(render_mode="human")
    
    # Load Policy
    obs_dim = int(env.observation_space.shape[0])
    policy = ActorCritic(obs_dim, hidden=128).to(device)
    try:
        policy.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights: {args.weights}")
    except FileNotFoundError:
        print("Weights file not found!")
        return
    policy.eval()

    obs, _ = env.reset()
    frames = []
    start_time = time.time()
    
    print("Running simulation...")
    if args.record:
        print(f"Recording to {OUTPUT_VIDEO} (Limit: {args.time_limit}s)...")

    running = True
    while running:
        # Handle window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        # Get Action
        action = get_action(policy, obs, device, stochastic=args.stochastic)
        obs, _, terminated, _, _ = env.step(action)

        if terminated:
            obs, _ = env.reset()

        # --- VISUALS ---
        # Hack to hide LiDAR: clear the list before rendering
        if args.no_lidar:
            env.lidar_rays = []
            
        env.render()

        # --- RECORDING ---
        if args.record:
            # Capture the screen surface
            # Pygame surface is (Width, Height, RGB). 
            # Video expects (Height, Width, RGB).
            # We transpose axes 0 and 1.
            frame = pygame.surfarray.array3d(env.screen)
            frame = frame.transpose([1, 0, 2])
            frames.append(frame)

            if (time.time() - start_time) > args.time_limit:
                print("Time limit reached.")
                running = False
        
        # Cap FPS for display (not needed for recording logic, but good for watching)
        env.clock.tick(FPS)

    env.close()

    if args.record and frames:
        print(f"Saving {len(frames)} frames to {OUTPUT_VIDEO}...")
        # Save using imageio (ffmpeg)
        imageio.mimsave(OUTPUT_VIDEO, frames, fps=FPS, quality=8)
        print("Done!")

if __name__ == "__main__":
    main()