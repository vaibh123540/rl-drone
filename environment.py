# environment.py  (UPDATED: reward shaping to stop spin-farming + encourage movement/kills)
import math
import numpy as np

# Optional pygame (only used if render_mode="human")
try:
    import pygame
except ImportError:
    pygame = None

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError("Please install gymnasium: pip install gymnasium")

# --- PHYSICS CONSTANTS ---
FPS = 60
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800
DRONE_RADIUS = 15
ENEMY_RADIUS = 15
FRIENDLY_RADIUS = 15
OBSTACLE_RADIUS_RANGE = (30, 60)

MAX_SPEED = 5.0
DRAG = 0.97
THRUST_POWER = 0.2
ROTATION_SPEED = 0.1
BULLET_COOLDOWN = 20

NUM_RAYS = 32
RAY_LENGTH = 300
MAX_STEPS = 1000

# counts (fixed-size arrays for speed)
N_OBS = 4
N_EN = 5
N_FR = 3

# --- COLORS ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (200, 50, 50)
GREEN = (50, 200, 50)
GRAY  = (100, 100, 100)
BLUE  = (50, 50, 200)
YELLOW = (255, 255, 0)


class DroneEnv(gym.Env):
    """
    Fast RL-friendly env:
    - No pygame init unless render_mode="human"
    - Vectorized LiDAR
    - Fixed-size arrays for entities

    Reward (UPDATED):
    - No "alive bonus" (prevents reward farming by doing nothing)
    - Step cost (time pressure)
    - Big terminal penalties for crashes / friendly hits
    - Main positive reward for enemy hits
    - Miss penalty is small (doesn't make agent afraid to shoot early)
    - Shaping is based on PROGRESS (distance decreasing) + movement toward enemy
    - Tiny steering penalty only when basically stationary (kills spin-in-place exploits)
    - Optional small bonus for shooting when aligned (only paid when a shot actually fires)
    """
    metadata = {"render_modes": ["human", None], "render_fps": FPS}

    def __init__(self, render_mode=None, seed=None):
        super().__init__()
        self.render_mode = render_mode

        self.rng = np.random.default_rng(seed)

        # Precompute ray offsets (relative to drone heading)
        self.ray_offsets = np.linspace(0.0, 2.0 * np.pi, NUM_RAYS, endpoint=False).astype(np.float32)

        # Action space: [thrust(-1..1), steer(-1..1), shoot(0..1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0,  1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        obs_dim = 5 + 2 + 4 * NUM_RAYS  # base(5) + rel_enemy(2) + lidar(4*NUM_RAYS)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Debug/render helpers
        self.lidar_rays = []
        self.last_shot = None
        self.last_reward = 0.0
        self.total_score = 0.0
        self.step_count = 0
        self.position_history = []
        self.distance_history = []

        # Only init pygame if rendering
        self.screen = None
        self.clock = None
        self.font = None
        if self.render_mode == "human":
            if pygame is None:
                raise ImportError("pygame not installed. pip install pygame")
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 18)

        # State placeholders
        self.drone_pos = np.zeros(2, dtype=np.float32)
        self.drone_vel = np.zeros(2, dtype=np.float32)
        self.drone_angle = 0.0
        self.cooldown = 0

        # Entities as arrays
        self.obstacles_pos = np.zeros((N_OBS, 2), dtype=np.float32)
        self.obstacles_r   = np.zeros((N_OBS,), dtype=np.float32)

        self.enemies_pos = np.zeros((N_EN, 2), dtype=np.float32)
        self.enemies_vel = np.zeros((N_EN, 2), dtype=np.float32)
        self.enemies_r   = np.full((N_EN,), ENEMY_RADIUS, dtype=np.float32)

        self.friends_pos = np.zeros((N_FR, 2), dtype=np.float32)
        self.friends_vel = np.zeros((N_FR, 2), dtype=np.float32)
        self.friends_r   = np.full((N_FR,), FRIENDLY_RADIUS, dtype=np.float32)

        # Reward shaping memory
        self.prev_enemy_dist = None

    def close(self):
        if self.render_mode == "human" and pygame is not None:
            pygame.quit()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.drone_pos[:] = (SCREEN_WIDTH / 2.0, SCREEN_HEIGHT / 2.0)
        self.drone_vel[:] = (0.0, 0.0)
        self.drone_angle = 0.0
        self.cooldown = 0

        self.total_score = 0.0
        self.step_count = 0
        self.lidar_rays = []
        self.last_shot = None
        self.last_reward = 0.0
        self.position_history = []
        self.distance_history = []

        # Spawn obstacles
        existing_centers = []
        existing_radii = []
        for i in range(N_OBS):
            pos, r = self._get_random_pos(
                min_dist=100,
                radius_range=OBSTACLE_RADIUS_RANGE,
                existing_centers=existing_centers,
                existing_radii=existing_radii
            )
            self.obstacles_pos[i] = pos
            self.obstacles_r[i] = r
            existing_centers.append(pos)
            existing_radii.append(r)

        # Spawn enemies (avoid obstacles)
        for i in range(N_EN):
            pos, _ = self._get_random_pos(
                min_dist=150,
                radius_range=(ENEMY_RADIUS, ENEMY_RADIUS),
                existing_centers=list(self.obstacles_pos),
                existing_radii=list(self.obstacles_r)
            )
            self.enemies_pos[i] = pos
            self.enemies_vel[i] = self.rng.normal(0, 0.5, size=2).astype(np.float32)

        # Spawn friendlies (avoid obstacles + enemies)
        all_centers = np.concatenate([self.obstacles_pos, self.enemies_pos], axis=0)
        all_radii = np.concatenate([self.obstacles_r, self.enemies_r], axis=0)
        for i in range(N_FR):
            pos, _ = self._get_random_pos(
                min_dist=150,
                radius_range=(FRIENDLY_RADIUS, FRIENDLY_RADIUS),
                existing_centers=list(all_centers),
                existing_radii=list(all_radii)
            )
            self.friends_pos[i] = pos
            self.friends_vel[i] = self.rng.normal(0, 0.3, size=2).astype(np.float32)

        # init reward shaping memory
        d = self.enemies_pos - self.drone_pos[None, :]
        dist = np.sqrt((d * d).sum(axis=1) + 1e-8)
        self.prev_enemy_dist = float(dist.min())

        return self._get_observation(), {}

    def _get_random_pos(self, min_dist, radius_range, existing_centers, existing_radii):
        for _ in range(200):
            if radius_range[0] == radius_range[1]:
                r = float(radius_range[0])
            else:
                r = float(self.rng.integers(radius_range[0], radius_range[1]))

            x = float(self.rng.integers(int(r), int(SCREEN_WIDTH - r)))
            y = float(self.rng.integers(int(r), int(SCREEN_HEIGHT - r)))
            pos = np.array([x, y], dtype=np.float32)

            # 1) not too close to drone
            if np.linalg.norm(pos - self.drone_pos) < min_dist:
                continue

            # 2) no overlap with existing circles
            valid = True
            for c, cr in zip(existing_centers, existing_radii):
                if np.linalg.norm(pos - c) < (r + float(cr) + 10.0):
                    valid = False
                    break

            if valid:
                return pos, r

        # fallback
        return np.array([50.0, 50.0], dtype=np.float32), float(radius_range[0])

    def step(self, action):
        thrust = float(np.clip(action[0], -1.0, 1.0))
        steer  = float(np.clip(action[1], -1.0, 1.0))
        shoot  = bool(action[2] > 0.5)

        # --- PHYSICS ---
        self.drone_angle = (self.drone_angle + steer * ROTATION_SPEED) % (2.0 * np.pi)

        if thrust > 0.0:
            force = thrust * THRUST_POWER
            self.drone_vel[0] += math.cos(self.drone_angle) * force
            self.drone_vel[1] += math.sin(self.drone_angle) * force

        self.drone_vel *= DRAG

        speed = float(np.linalg.norm(self.drone_vel))
        if speed > MAX_SPEED:
            self.drone_vel[:] = (self.drone_vel / speed) * MAX_SPEED

        self.drone_pos += self.drone_vel

        # update enemies/friends motion
        self._update_entities()

        # --- EVENTS ---
        events = {
            "hit_wall": False, "hit_obstacle": False, "hit_friendly": False, "hit_enemy": False,
            "shot_hit": False, "shot_missed": False, "shot_hit_friendly": False,
            "shot_fired": False, "shot_alignment": 0.0
        }
        terminated = False
        truncated = False

        # --- COLLISIONS WITH WALL ---
        if (self.drone_pos[0] < 0 or self.drone_pos[0] > SCREEN_WIDTH or
            self.drone_pos[1] < 0 or self.drone_pos[1] > SCREEN_HEIGHT):
            events["hit_wall"] = True
            terminated = True

        # --- COLLISIONS WITH CIRCLES (vectorized) ---
        if not terminated:
            if self._collides_any(self.obstacles_pos, self.obstacles_r):
                events["hit_obstacle"] = True
                terminated = True
            elif self._collides_any(self.friends_pos, self.friends_r):
                events["hit_friendly"] = True
                terminated = True
            elif self._collides_any(self.enemies_pos, self.enemies_r):
                events["hit_enemy"] = True
                terminated = True

        # --- SHOOTING ---
        can_shoot = (self.cooldown <= 0)
        if self.cooldown > 0:
            self.cooldown -= 1

        if shoot and can_shoot and not terminated:
            # alignment to closest enemy at the moment we shoot
            d = self.enemies_pos - self.drone_pos[None, :]
            dist = np.sqrt((d * d).sum(axis=1) + 1e-8)
            idx = int(np.argmin(dist))
            to_enemy = d[idx] / (dist[idx] + 1e-8)
            heading = np.array([math.cos(self.drone_angle), math.sin(self.drone_angle)], dtype=np.float32)
            events["shot_alignment"] = float(np.dot(heading, to_enemy))  # [-1..1]

            self.cooldown = BULLET_COOLDOWN
            shot_result, impact_point = self._process_shot()
            events["shot_fired"] = True

            if self.render_mode == "human":
                sp = (float(self.drone_pos[0]), float(self.drone_pos[1]))
                ip = (float(impact_point[0]), float(impact_point[1]))
                self.last_shot = (sp, ip)

            if shot_result == "ENEMY":
                events["shot_hit"] = True
            elif shot_result == "FRIEND":
                events["shot_hit_friendly"] = True
            else:
                events["shot_missed"] = True

        # --- REWARD ---
        reward = self._calculate_reward(events, thrust, steer)

        # --- STEP COUNT / TRUNCATION ---
        self.step_count += 1
        if self.step_count >= MAX_STEPS:
            truncated = True

        self.last_reward = reward
        self.total_score += reward

        # --- INFO FOR LOGGING/DEBUGGING (NEW) ---
        # term_code:
        # 0 = none, 1 = wall, 2 = obstacle, 3 = friendly collision, 4 = enemy collision, 5 = timeout(trunc)
        term_code = 0
        if events["hit_wall"]:
            term_code = 1
        elif events["hit_obstacle"]:
            term_code = 2
        elif events["hit_friendly"]:
            term_code = 3
        elif events["hit_enemy"]:
            term_code = 4
        if truncated:
            term_code = 5  # override

        info = {
            "term_code": int(term_code),
            "shot_fired": int(events["shot_fired"]),
            "shot_hit": int(events["shot_hit"]),
            "shot_missed": int(events["shot_missed"]),
            "shot_hit_friendly": int(events["shot_hit_friendly"]),
            "shot_alignment": float(events["shot_alignment"]),
        }

        return self._get_observation(), reward, terminated, truncated, info


    def _collides_any(self, centers, radii):
        d = centers - self.drone_pos[None, :]
        dist2 = (d * d).sum(axis=1)
        rr = (DRONE_RADIUS + radii) ** 2
        return bool(np.any(dist2 < rr))

    def _update_entities(self):
        # enemies
        self.enemies_pos += self.enemies_vel

        mask = (self.enemies_pos[:, 0] < 0) | (self.enemies_pos[:, 0] > SCREEN_WIDTH)
        self.enemies_vel[mask, 0] *= -1
        mask = (self.enemies_pos[:, 1] < 0) | (self.enemies_pos[:, 1] > SCREEN_HEIGHT)
        self.enemies_vel[mask, 1] *= -1

        self.enemies_pos[:, 0] = np.clip(self.enemies_pos[:, 0], 0, SCREEN_WIDTH)
        self.enemies_pos[:, 1] = np.clip(self.enemies_pos[:, 1], 0, SCREEN_HEIGHT)

        # friends
        self.friends_pos += self.friends_vel

        mask = (self.friends_pos[:, 0] < 0) | (self.friends_pos[:, 0] > SCREEN_WIDTH)
        self.friends_vel[mask, 0] *= -1
        mask = (self.friends_pos[:, 1] < 0) | (self.friends_pos[:, 1] > SCREEN_HEIGHT)
        self.friends_vel[mask, 1] *= -1

        self.friends_pos[:, 0] = np.clip(self.friends_pos[:, 0], 0, SCREEN_WIDTH)
        self.friends_pos[:, 1] = np.clip(self.friends_pos[:, 1], 0, SCREEN_HEIGHT)

    def _process_shot(self):
        start = self.drone_pos
        direction = np.array([math.cos(self.drone_angle), math.sin(self.drone_angle)], dtype=np.float32)

        centers = np.concatenate([self.obstacles_pos, self.friends_pos, self.enemies_pos], axis=0)
        radii = np.concatenate([self.obstacles_r, self.friends_r, self.enemies_r], axis=0)
        n_obs = len(self.obstacles_pos)
        n_fr = len(self.friends_pos)

        f = centers - start[None, :]
        proj = f @ direction
        f_norm_sq = (f * f).sum(axis=1)
        dist_sq = f_norm_sq - proj * proj
        r2 = radii * radii

        valid = (proj > 0.0) & (dist_sq < r2)
        offset = np.sqrt(np.maximum(r2 - dist_sq, 0.0))
        impact = proj - offset
        impact = np.where(valid, impact, np.inf)
        impact = np.where(impact < 300.0, impact, np.inf)

        hit_i = int(np.argmin(impact))
        best = float(impact[hit_i])

        if not np.isfinite(best):
            end_point = start + direction * 300.0
            return "MISS", end_point

        end_point = start + direction * best

        if hit_i < n_obs:
            return "OBSTACLE", end_point
        elif hit_i < n_obs + n_fr:
            return "FRIEND", end_point
        else:
            enemy_index = hit_i - (n_obs + n_fr)
            all_centers = np.concatenate([self.obstacles_pos, self.friends_pos, self.enemies_pos], axis=0)
            all_radii = np.concatenate([self.obstacles_r, self.friends_r, self.enemies_r], axis=0)
            pos, _ = self._get_random_pos(
                min_dist=150,
                radius_range=(ENEMY_RADIUS, ENEMY_RADIUS),
                existing_centers=list(all_centers),
                existing_radii=list(all_radii)
            )
            self.enemies_pos[enemy_index] = pos
            self.enemies_vel[enemy_index] = self.rng.normal(0, 0.5, size=2).astype(np.float32)
            return "ENEMY", end_point

    def _calculate_reward(self, events, thrust, steer):
        """
        MINIMAL FIX to prevent "circle and wait" strategy.
        
        Changes from original:
        1. Only reward distance decrease if agent is moving TOWARD enemy
        2. Track agent's position history to detect camping/circling
        3. Add small penalty for revisiting same area
        """
        r = 0.0
        
        # ----- TERMINAL PENALTIES -----
        if events["hit_wall"]:
            r -= 100.0
        if events["hit_obstacle"]:
            r -= 100.0
        if events["hit_enemy"]:
            r -= 100.0
        if events["hit_friendly"]:
            r -= 120.0
        
        # ----- SHOOTING OUTCOMES -----
        if events["shot_hit"]:
            r += 50.0
        if events["shot_hit_friendly"]:
            r -= 200.0
        if events["shot_missed"]:
            r -= 1  # REDUCED from 0.5 to encourage shooting while pursuing
        
        # ----- DISTANCE-BASED TIME PRESSURE -----
        d = self.enemies_pos - self.drone_pos[None, :]
        dist = np.sqrt((d * d).sum(axis=1) + 1e-8)
        min_dist = float(dist.min())
        
        max_possible_dist = np.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)
        norm_dist = min(min_dist / max_possible_dist, 1.0)
        
        distance_penalty = -0.10 * (norm_dist ** 2)
        r += distance_penalty
        
        # ----- MOVEMENT REQUIREMENT -----
        speed = float(np.linalg.norm(self.drone_vel))
        
        if speed < 0.5:
            r -= 0.05 * (1.0 - speed / 0.5)
        
        if speed < 0.5:
            r -= 0.02 * abs(steer)
        
        # ----- CLOSING DISTANCE (MAIN SHAPING) ----- 
        # FIX #1: Only reward if agent is actually moving TOWARD enemy
        # NEW: Scale reward by distance (bigger reward for closing gap from far away)
        if self.prev_enemy_dist is not None:
            delta = self.prev_enemy_dist - min_dist
            
            # Get direction to nearest enemy
            idx = int(np.argmin(dist))
            to_enemy = d[idx] / (dist[idx] + 1e-8)
            
            # Check if agent is moving toward enemy
            is_pursuing = False
            if speed > 0.1:
                vel_dir = self.drone_vel / (speed + 1e-8)
                toward = float(np.dot(vel_dir, to_enemy))
                is_pursuing = (toward > 0.3)  # At least 30% aligned
            
            if delta > 0.0:
                # CHANGED: Only reward if agent is actively pursuing
                if is_pursuing:
                    speed_factor = min(speed / MAX_SPEED, 1.0)
                    
                    # NEW: Distance multiplier - closing gap from far away is MORE valuable
                    # At 500px: multiplier = 2.5x
                    # At 300px: multiplier = 1.5x  
                    # At 150px: multiplier = 1.0x
                    distance_multiplier = 1.0 + (min_dist / 300.0)
                    
                    r += 0.30 * delta * (0.5 + 0.5 * speed_factor) * distance_multiplier
                else:
                    # Enemy drifted closer, but we weren't chasing - no reward
                    pass
            else:
                # Penalty for letting enemy get farther (scales with distance too)
                distance_factor = 1.0 + (min_dist / 400.0)
                r += 0.15 * delta * distance_factor
        
        self.prev_enemy_dist = min_dist
        
        # ----- VELOCITY ALIGNMENT (encourage pursuit) -----
        # NEW: Scale by distance - pursuing far enemies is MORE valuable
        idx = int(np.argmin(dist))
        to_enemy = d[idx] / (dist[idx] + 1e-8)
        
        if speed > 0.1:
            vel_dir = self.drone_vel / (speed + 1e-8)
            toward = float(np.dot(vel_dir, to_enemy))
            
            # Distance-scaled pursuit bonus
            # Far enemies (>300px): up to 2x bonus
            # Close enemies (<150px): 1x bonus
            pursuit_scale = 1.0 + min(min_dist / 300.0, 1.0)
            
            if toward > 0:
                r += 0.15 * toward * (speed / MAX_SPEED) * pursuit_scale
            else:
                r += 0.05 * toward
        
        # NEW: "Hunter" bonus - strong reward for full-speed pursuit of distant enemies
        if min_dist > 250.0 and speed > 3.0:  # Far enemy + moving fast
            if speed > 0.1:
                vel_dir = self.drone_vel / (speed + 1e-8)
                toward = float(np.dot(vel_dir, to_enemy))
                if toward > 0.7:  # Well-aligned
                    hunting_bonus = 0.08 * (min_dist / 300.0)  # Scales with distance
                    r += hunting_bonus
        
        # FIX #2: Detect circular motion / camping behavior
        # Track position history to penalize staying in same area
        if not hasattr(self, 'position_history'):
            self.position_history = []
        
        self.position_history.append(self.drone_pos.copy())
        if len(self.position_history) > 50:  # Keep last 50 positions
            self.position_history.pop(0)
        
        # If we have enough history, check if agent is camping
        if len(self.position_history) >= 30:
            recent_positions = np.array(self.position_history[-30:])
            center = recent_positions.mean(axis=0)
            distances_from_center = np.linalg.norm(recent_positions - center[None, :], axis=1)
            avg_radius = float(distances_from_center.mean())
            
            # If agent has been moving in a small area (radius < 100 pixels)
            if avg_radius < 100.0:
                camping_penalty = -0.03 * (1.0 - avg_radius / 100.0)
                r += camping_penalty
        
        # FIX #3: Stronger penalty for maintaining distance instead of closing it
        # If agent has been at similar distance for a while, penalize
        if not hasattr(self, 'distance_history'):
            self.distance_history = []
        
        self.distance_history.append(min_dist)
        if len(self.distance_history) > 30:
            self.distance_history.pop(0)
        
        if len(self.distance_history) >= 20:
            recent_dists = np.array(self.distance_history[-20:])
            dist_variance = float(np.var(recent_dists))
            
            # INCREASED threshold: If distance staying constant and not close
            if dist_variance < 900.0 and min_dist > 150.0:  # variance < 30^2 (was 20^2)
                stagnation_penalty = -0.06  # Increased from -0.04
                r += stagnation_penalty
        
        # ----- SHOOTING DISCIPLINE -----
        if events.get("shot_fired", False):
            align = float(events.get("shot_alignment", 0.0))
            
            if align < 0.5:
                r -= 0.30 * (1.0 - align)
            
            if align > 0.7:
                r += 0.50 * (align - 0.7) / 0.3
        
        # ----- ENGAGEMENT RANGE BONUS -----
        # REDUCED: Don't want agent to be satisfied just maintaining distance
        optimal_range = 150.0
        range_error = abs(min_dist - optimal_range) / optimal_range
        
        if min_dist < 250.0:  # Only bonus when reasonably close
            r += 0.01 * (1.0 - min(range_error, 1.0))  # Halved from 0.02
        
        return float(r)

    # ---- Vectorized LiDAR ----
    def _ray_wall_dist(self, origin, dirs):
        """
        FIXED: avoids np.where-evaluates-both-branches divide-by-zero.
        Uses masked division + hard sanitization, so render never sees NaN/Inf.
        """
        x0, y0 = float(origin[0]), float(origin[1])
        dx = dirs[:, 0].astype(np.float32)
        dy = dirs[:, 1].astype(np.float32)

        eps = 1e-12
        dist_x = np.full(dx.shape, np.inf, dtype=np.float32)
        dist_y = np.full(dy.shape, np.inf, dtype=np.float32)

        # X walls
        xb = np.where(dx > 0.0, float(SCREEN_WIDTH), 0.0).astype(np.float32)
        mx = np.abs(dx) > eps
        dist_x[mx] = (xb[mx] - x0) / dx[mx]
        dist_x = np.where((dist_x > 0.0) & (dist_x < RAY_LENGTH), dist_x, np.inf)

        # Y walls
        yb = np.where(dy > 0.0, float(SCREEN_HEIGHT), 0.0).astype(np.float32)
        my = np.abs(dy) > eps
        dist_y[my] = (yb[my] - y0) / dy[my]
        dist_y = np.where((dist_y > 0.0) & (dist_y < RAY_LENGTH), dist_y, np.inf)

        dmin = np.minimum(dist_x, dist_y)

        out = np.where(np.isfinite(dmin), dmin / RAY_LENGTH, 1.0).astype(np.float32)
        out = np.nan_to_num(out, nan=1.0, posinf=1.0, neginf=1.0)
        out = np.clip(out, 0.0, 1.0)
        return out

    def _ray_circle_min_dist(self, origin, dirs, centers, radii):
        if centers is None or len(centers) == 0:
            return np.ones((NUM_RAYS,), dtype=np.float32)

        f = centers - origin[None, :]               # (N,2)
        proj = f @ dirs.T                           # (N,R)
        f_norm_sq = (f * f).sum(axis=1)             # (N,)
        dist_sq = f_norm_sq[:, None] - proj * proj  # (N,R)
        r2 = (radii * radii)[:, None]               # (N,1)

        valid = (proj > 0.0) & (dist_sq < r2)
        offset = np.sqrt(np.maximum(r2 - dist_sq, 0.0))
        impact = proj - offset
        impact = np.where(valid, impact, np.inf)
        impact = np.where(impact < RAY_LENGTH, impact, np.inf)

        dmin = impact.min(axis=0)  # (R,)
        out = np.where(np.isfinite(dmin), dmin / RAY_LENGTH, 1.0).astype(np.float32)
        out = np.nan_to_num(out, nan=1.0, posinf=1.0, neginf=1.0)
        out = np.clip(out, 0.0, 1.0)
        return out

    def _get_observation(self):
        obs = np.array([
            self.drone_vel[0] / MAX_SPEED,
            self.drone_vel[1] / MAX_SPEED,
            math.sin(self.drone_angle),
            math.cos(self.drone_angle),
            self.cooldown / float(BULLET_COOLDOWN)
        ], dtype=np.float32)

        angles = self.ray_offsets + self.drone_angle
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)  # (R,2)

        wall_d = self._ray_wall_dist(self.drone_pos, dirs)
        obs_d  = self._ray_circle_min_dist(self.drone_pos, dirs, self.obstacles_pos, self.obstacles_r)
        en_d   = self._ray_circle_min_dist(self.drone_pos, dirs, self.enemies_pos, self.enemies_r)
        fr_d   = self._ray_circle_min_dist(self.drone_pos, dirs, self.friends_pos, self.friends_r)

        dist_mat = np.stack([wall_d, obs_d, en_d, fr_d], axis=0)  # (4,R)
        min_dist = dist_mat.min(axis=0)
        idx = dist_mat.argmin(axis=0)

        # HARD sanitize (prevents NaN/Inf from ever reaching pygame math)
        min_dist = np.nan_to_num(min_dist, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
        min_dist = np.clip(min_dist, 0.0, 1.0)

        hit = min_dist < 1.0

        walls = np.ones(NUM_RAYS, dtype=np.float32)
        obsts = np.ones(NUM_RAYS, dtype=np.float32)
        enems = np.ones(NUM_RAYS, dtype=np.float32)
        frnds = np.ones(NUM_RAYS, dtype=np.float32)

        m0 = hit & (idx == 0); walls[m0] = min_dist[m0]
        m1 = hit & (idx == 1); obsts[m1] = min_dist[m1]
        m2 = hit & (idx == 2); enems[m2] = min_dist[m2]
        m3 = hit & (idx == 3); frnds[m3] = min_dist[m3]

        lidar = np.concatenate([walls, obsts, enems, frnds], axis=0).astype(np.float32)

        # closest enemy relative position (scaled)
        d = self.enemies_pos - self.drone_pos[None, :]
        dist2 = (d * d).sum(axis=1)
        ci = int(np.argmin(dist2))
        rel = d[ci]
        rel_obs = np.array([rel[0] / SCREEN_WIDTH, rel[1] / SCREEN_HEIGHT], dtype=np.float32)

        out = np.concatenate([obs, rel_obs, lidar], axis=0).astype(np.float32)

        # Only build ray segments if rendering
        if self.render_mode == "human":
            self.lidar_rays = []
            for i in range(NUM_RAYS):
                ray_len_px = float(min_dist[i]) * RAY_LENGTH
                start = self.drone_pos
                end = start + dirs[i] * ray_len_px

                # pygame is happiest with plain python tuples
                start_t = (float(start[0]), float(start[1]))
                end_t   = (float(end[0]), float(end[1]))

                color = (50, 50, 50)
                if not hit[i]:
                    color = (50, 50, 50)
                elif idx[i] == 0:
                    color = WHITE
                elif idx[i] == 1:
                    color = GRAY
                elif idx[i] == 2:
                    color = RED
                elif idx[i] == 3:
                    color = GREEN

                self.lidar_rays.append((start_t, end_t, color))

        return out

    def render(self):
        if self.render_mode != "human":
            return

        self.screen.fill(BLACK)

        # rays
        for start, end, color in self.lidar_rays:
            pygame.draw.line(self.screen, color, start, end, 1)

        # obstacles
        for i in range(N_OBS):
            pygame.draw.circle(self.screen, GRAY, self.obstacles_pos[i].astype(int), int(self.obstacles_r[i]))

        # enemies
        for i in range(N_EN):
            pygame.draw.circle(self.screen, RED, self.enemies_pos[i].astype(int), ENEMY_RADIUS)

        # friendlies
        for i in range(N_FR):
            pygame.draw.circle(self.screen, GREEN, self.friends_pos[i].astype(int), FRIENDLY_RADIUS)

        # last shot line
        if self.last_shot is not None:
            pygame.draw.line(self.screen, YELLOW, self.last_shot[0], self.last_shot[1], 3)
            self.last_shot = None

        # drone
        dp = (float(self.drone_pos[0]), float(self.drone_pos[1]))
        pygame.draw.circle(self.screen, BLUE, (int(dp[0]), int(dp[1])), DRONE_RADIUS)

        end_pos = self.drone_pos + np.array([math.cos(self.drone_angle), math.sin(self.drone_angle)], dtype=np.float32) * 20
        ep = (float(end_pos[0]), float(end_pos[1]))
        pygame.draw.line(self.screen, WHITE, dp, ep, 2)

        # HUD
        pygame.draw.rect(self.screen, (20, 20, 20), (5, 5, 220, 110))
        pygame.draw.rect(self.screen, WHITE, (5, 5, 220, 110), 2)
        lines = [
            f"Step: {self.step_count}",
            f"Reward: {self.last_reward:.2f}",
            f"Total: {self.total_score:.2f}",
            f"FPS: {int(self.clock.get_fps()) if self.clock else 0}",
        ]
        for i, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, WHITE), (15, 15 + i * 22))

        pygame.display.flip()
        if self.clock:
            self.clock.tick(FPS)


def manual_control():
    if pygame is None:
        raise ImportError("pygame not installed. pip install pygame")

    env = DroneEnv(render_mode="human")
    obs, _ = env.reset()
    total_reward = 0.0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if keys[pygame.K_UP]:
            action[0] = 1.0
        if keys[pygame.K_LEFT]:
            action[1] = -1.0
        elif keys[pygame.K_RIGHT]:
            action[1] = 1.0
        if keys[pygame.K_SPACE]:
            action[2] = 1.0

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        env.render()

        if terminated or truncated:
            print(f"Finished! Score: {total_reward:.2f}")
            obs, _ = env.reset()
            total_reward = 0.0

    env.close()


if __name__ == "__main__":
    manual_control()