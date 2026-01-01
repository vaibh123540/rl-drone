import pygame
import numpy as np
import math

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
MAX_STEPS = 1000 # [Fix 3] Time Limit

# --- COLORS ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (200, 50, 50)
GREEN = (50, 200, 50)
GRAY  = (100, 100, 100)
BLUE  = (50, 50, 200)
YELLOW = (255, 255, 0)

class DroneEnv:
    def __init__(self, render_mode="human"):
        pygame.init()
        self.render_mode = render_mode
        
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 18)
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.font = pygame.font.SysFont("Arial", 18)
        
        # Debug vars
        self.lidar_rays = [] 
        self.last_shot = None
        self.last_reward = 0.0
        self.total_score = 0.0
        self.step_count = 0

    def reset(self):
        """Resets the environment for a new episode."""
        self.drone_pos = np.array([SCREEN_WIDTH/2, SCREEN_HEIGHT/2], dtype=float)
        self.drone_vel = np.array([0.0, 0.0], dtype=float)
        self.drone_angle = 0.0 
        self.cooldown = 0
        
        self.total_score = 0.0
        self.step_count = 0
        self.lidar_rays = []

        self.enemies = []
        self.friendlies = []
        self.obstacles = []
        
        # [Fix 4] robust spawning checking against ALL entities
        # 1. Spawn Obstacles
        for _ in range(4):
            pos, r = self._get_random_pos(min_dist=100, radius_range=OBSTACLE_RADIUS_RANGE, existing_list=[])
            self.obstacles.append({'pos': pos, 'r': r})

        # 2. Spawn Enemies
        for _ in range(5):
            pos, r = self._get_random_pos(min_dist=150, radius_range=(ENEMY_RADIUS, ENEMY_RADIUS), existing_list=self.obstacles)
            vel = np.random.randn(2) * 0.5
            self.enemies.append({'pos': pos, 'r': r, 'vel': vel})

        # 3. Spawn Friendlies
        for _ in range(3):
            # Check against obstacles AND enemies
            all_circles = self.obstacles + self.enemies
            pos, r = self._get_random_pos(min_dist=150, radius_range=(FRIENDLY_RADIUS, FRIENDLY_RADIUS), existing_list=all_circles)
            vel = np.random.randn(2) * 0.3
            self.friendlies.append({'pos': pos, 'r': r, 'vel': vel})

        return self._get_observation(), {} # Gymnasium returns (obs, info)

    def _get_random_pos(self, min_dist, radius_range, existing_list):
        """
        Finds a valid spawn position that does not overlap with:
        1. The Drone (min_dist)
        2. Any circle in 'existing_list' (obstacles, enemies, etc.)
        """
        for _ in range(100): 
            if radius_range[0] == radius_range[1]:
                r = radius_range[0]
            else:
                r = np.random.randint(radius_range[0], radius_range[1])
            
            x = np.random.randint(r, SCREEN_WIDTH - r)
            y = np.random.randint(r, SCREEN_HEIGHT - r)
            pos = np.array([x, y], dtype=float)
            
            # 1. Check Drone Distance
            if np.linalg.norm(pos - self.drone_pos) < min_dist:
                continue

            # 2. Check Overlap with Existing Entities
            valid = True
            for entity in existing_list:
                dist = np.linalg.norm(pos - entity['pos'])
                if dist < (r + entity['r'] + 10): # +10 buffer
                    valid = False
                    break
            
            if valid:
                return pos, r

        return np.array([0,0]), 10 # Fallback

    def step(self, action):
        # 1. Action Decode
        thrust = np.clip(action[0], -1.0, 1.0)
        steer = np.clip(action[1], -1.0, 1.0)
        shoot = action[2] > 0.5

        # 2. Physics
        self.drone_angle += steer * ROTATION_SPEED
        self.drone_angle %= (2 * np.pi)

        if thrust > 0:
            force = thrust * THRUST_POWER
            self.drone_vel[0] += math.cos(self.drone_angle) * force
            self.drone_vel[1] += math.sin(self.drone_angle) * force
        
        self.drone_vel *= DRAG

        # Clamp Velocity
        speed = np.linalg.norm(self.drone_vel)
        if speed > MAX_SPEED:
            self.drone_vel = (self.drone_vel / speed) * MAX_SPEED

        self.drone_pos += self.drone_vel
        self._update_entities()

        # 3. Events
        events = {
            "hit_wall": False, "hit_obstacle": False, "hit_friendly": False, "hit_enemy": False,
            "shot_hit": False, "shot_missed": False, "shot_hit_friendly": False
        }
        terminated = False
        truncated = False # [Fix 3]

        # 4. Collisions
        if (self.drone_pos[0] < 0 or self.drone_pos[0] > SCREEN_WIDTH or 
            self.drone_pos[1] < 0 or self.drone_pos[1] > SCREEN_HEIGHT):
            events["hit_wall"] = True
            terminated = True

        def check_hit(pos, r):
            return np.linalg.norm(self.drone_pos - pos) < (DRONE_RADIUS + r)

        if not terminated:
            for obs in self.obstacles:
                if check_hit(obs['pos'], obs['r']):
                    events["hit_obstacle"] = True; terminated = True; break
            for friend in self.friendlies:
                if check_hit(friend['pos'], friend['r']):
                    events["hit_friendly"] = True; terminated = True; break
            for enemy in self.enemies:
                if check_hit(enemy['pos'], enemy['r']):
                    events["hit_enemy"] = True; terminated = True; break

        # 5. Shooting
        can_shoot = (self.cooldown <= 0) # Track specifically for shaping
        if self.cooldown > 0: self.cooldown -= 1
        
        if shoot and can_shoot and not terminated:
            self.cooldown = BULLET_COOLDOWN
            shot_result, impact_point = self._process_shot() 
            self.last_shot = (self.drone_pos.copy(), impact_point)

            if shot_result == "ENEMY": events["shot_hit"] = True
            elif shot_result == "FRIEND": events["shot_hit_friendly"] = True
            else: events["shot_missed"] = True

        # 6. Reward & Truncation
        reward = self._calculate_reward(events, can_shoot) # Pass cooldown status
        
        self.step_count += 1
        if self.step_count >= MAX_STEPS:
            truncated = True
        
        self.last_reward = reward
        self.total_score += reward
        
        # [Fix 3] Return 5 values for Gymnasium compatibility
        return self._get_observation(), reward, terminated, truncated, {}

    def _update_entities(self):
        for entity in self.enemies + self.friendlies:
            entity['pos'] += entity['vel']
            for i in [0, 1]:
                if entity['pos'][i] < 0 or entity['pos'][i] > (SCREEN_WIDTH if i==0 else SCREEN_HEIGHT):
                    entity['vel'][i] *= -1

    def _process_shot(self):
        start = self.drone_pos
        direction = np.array([math.cos(self.drone_angle), math.sin(self.drone_angle)])
        end = start + direction * 300
        
        closest_hit_dist = 300.0
        hit_type = "MISS" 
        hit_index = -1

        def check_circles(entities, label):
            nonlocal closest_hit_dist, hit_type, hit_index
            for i, entity in enumerate(entities):
                v_circle = entity['pos'] - start
                v_line = end - start
                line_len = np.linalg.norm(v_line)
                v_line_unit = v_line / line_len
                proj = np.dot(v_circle, v_line_unit)
                if 0 < proj < line_len:
                    closest_point = start + v_line_unit * proj
                    dist_to_center = np.linalg.norm(entity['pos'] - closest_point)
                    if dist_to_center < entity['r']:
                        offset = math.sqrt(entity['r']**2 - dist_to_center**2)
                        impact_dist = proj - offset
                        if impact_dist < closest_hit_dist:
                            closest_hit_dist = impact_dist; hit_type = label; hit_index = i

        check_circles(self.obstacles, "OBSTACLE")
        check_circles(self.friendlies, "FRIEND")
        check_circles(self.enemies, "ENEMY")
        
        actual_end_point = start + direction * closest_hit_dist

        if hit_type == "ENEMY":
            self.enemies.pop(hit_index)
            # Respawn safely
            all_circles = self.obstacles + self.enemies + self.friendlies
            pos, r = self._get_random_pos(min_dist=150, radius_range=(ENEMY_RADIUS, ENEMY_RADIUS), existing_list=all_circles)
            vel = np.random.randn(2) * 0.5
            self.enemies.append({'pos': pos, 'r': r, 'vel': vel})
            return "ENEMY", actual_end_point
        elif hit_type == "FRIEND": return "FRIEND", actual_end_point
        elif hit_type == "OBSTACLE": return "OBSTACLE", actual_end_point
        return "MISS", actual_end_point

    def _calculate_reward(self, events, can_shoot):
        total_reward = 0.0
        
        if events["shot_hit"]: total_reward += 10.0
        total_reward -= 0.01 
        if events["shot_missed"]: total_reward -= 0.1
        if events["shot_hit_friendly"]: total_reward -= 30.0 
        
        # [Fix 2] Ensure ALL penalties are here
        if events["hit_wall"]: total_reward -= 10.0
        if events["hit_obstacle"]: total_reward -= 10.0
        if events["hit_friendly"]: total_reward -= 20.0 
        if events["hit_enemy"]: total_reward -= 10.0 
        
        # [Fix 5] Gated Reward Shaping
        # Only reward aiming if we are actually capable of shooting (cooldown ready)
        if self.enemies and can_shoot:
            dists = [np.linalg.norm(e['pos'] - self.drone_pos) for e in self.enemies]
            closest_idx = np.argmin(dists)
            
            vec_to_enemy = self.enemies[closest_idx]['pos'] - self.drone_pos
            vec_to_enemy /= (dists[closest_idx] + 0.1)
            
            heading = np.array([math.cos(self.drone_angle), math.sin(self.drone_angle)])
            alignment = np.dot(heading, vec_to_enemy)
            
            if alignment > 0:
                total_reward += alignment * 0.05
                
        return total_reward

    def _get_observation(self):
        obs = [
            self.drone_vel[0] / MAX_SPEED, self.drone_vel[1] / MAX_SPEED,
            math.sin(self.drone_angle), math.cos(self.drone_angle),
            self.cooldown / BULLET_COOLDOWN
        ]
        
        angles = np.linspace(0, 2*np.pi, NUM_RAYS, endpoint=False) + self.drone_angle
        self.lidar_rays = [] 
        
        # [Fix 1] 4 Channels now!
        walls = np.ones(NUM_RAYS)
        obstacles = np.ones(NUM_RAYS)
        enemies = np.ones(NUM_RAYS)
        friends = np.ones(NUM_RAYS)
        
        for i, angle in enumerate(angles):
            dist, obj_type = self._cast_ray(angle)
            ray_len_px = dist * RAY_LENGTH
            start = self.drone_pos
            end = start + np.array([math.cos(angle), math.sin(angle)]) * ray_len_px
            
            color = (50, 50, 50) 
            if obj_type == "WALL": walls[i] = dist; color = WHITE
            elif obj_type == "OBSTACLE": obstacles[i] = dist; color = GRAY # Separate Channel
            elif obj_type == "ENEMY": enemies[i] = dist; color = RED
            elif obj_type == "FRIEND": friends[i] = dist; color = GREEN
            self.lidar_rays.append((start, end, color))
            
        lidar_data = np.concatenate((walls, obstacles, enemies, friends))
        
        if self.enemies:
            dists = [np.linalg.norm(e['pos'] - self.drone_pos) for e in self.enemies]
            closest = self.enemies[np.argmin(dists)]
            rel_pos = (closest['pos'] - self.drone_pos)
            # Nice-to-have fix: scale X by width, Y by height
            obs.extend([rel_pos[0]/SCREEN_WIDTH, rel_pos[1]/SCREEN_HEIGHT])
        else:
            obs.extend([0.0, 0.0])

        return np.concatenate((obs, lidar_data), axis=None).astype(np.float32)

    def _cast_ray(self, angle):
        dx, dy = math.cos(angle), math.sin(angle)
        dir_vec = np.array([dx, dy])
        min_dist = RAY_LENGTH
        obj_type = "NONE"

        if dx != 0:
            x_boundary = SCREEN_WIDTH if dx > 0 else 0
            dist_x = (x_boundary - self.drone_pos[0]) / dx
            if 0 < dist_x < min_dist: min_dist = dist_x; obj_type = "WALL"
        if dy != 0:
            y_boundary = SCREEN_HEIGHT if dy > 0 else 0
            dist_y = (y_boundary - self.drone_pos[1]) / dy
            if 0 < dist_y < min_dist: min_dist = dist_y; obj_type = "WALL"

        def check_circles(entities, label):
            nonlocal min_dist, obj_type
            for entity in entities:
                v_to_center = entity['pos'] - self.drone_pos
                proj = np.dot(v_to_center, dir_vec)
                if proj > 0:
                    dist_sq = np.dot(v_to_center, v_to_center) - (proj * proj)
                    if dist_sq < entity['r']**2:
                        offset = math.sqrt(entity['r']**2 - dist_sq)
                        impact_dist = proj - offset
                        if 0 < impact_dist < min_dist: min_dist = impact_dist; obj_type = label

        # [Fix 1] Check obstacles with "OBSTACLE" label now
        check_circles(self.obstacles, "OBSTACLE")
        check_circles(self.enemies, "ENEMY")
        check_circles(self.friendlies, "FRIEND")
        return min_dist / RAY_LENGTH, obj_type

    def render(self):
        if self.render_mode != "human": return
        self.screen.fill(BLACK)
        for start, end, color in self.lidar_rays: pygame.draw.line(self.screen, color, start, end, 1)
        for o in self.obstacles: pygame.draw.circle(self.screen, GRAY, o['pos'].astype(int), o['r'])
        for e in self.enemies: pygame.draw.circle(self.screen, RED, e['pos'].astype(int), e['r'])
        for f in self.friendlies: pygame.draw.circle(self.screen, GREEN, f['pos'].astype(int), f['r'])
        if self.last_shot: pygame.draw.line(self.screen, YELLOW, self.last_shot[0], self.last_shot[1], 3); self.last_shot = None
        pygame.draw.circle(self.screen, BLUE, self.drone_pos.astype(int), DRONE_RADIUS)
        end_pos = self.drone_pos + np.array([math.cos(self.drone_angle), math.sin(self.drone_angle)]) * 20
        pygame.draw.line(self.screen, WHITE, self.drone_pos, end_pos, 2)
        
        pygame.draw.rect(self.screen, (20, 20, 20), (5, 5, 200, 100))
        pygame.draw.rect(self.screen, WHITE, (5, 5, 200, 100), 2)
        lines = [f"Step: {self.step_count}", f"Reward: {self.last_reward:.2f}", f"Total: {self.total_score:.2f}", f"FPS: {int(self.clock.get_fps())}"]
        for i, line in enumerate(lines): self.screen.blit(self.font.render(line, True, WHITE), (15, 15 + i * 20))

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit()

def manual_control():
    env = DroneEnv()
    obs, _ = env.reset() # Gymnasium style reset returns (obs, info)
    total_reward = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        keys = pygame.key.get_pressed()
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if keys[pygame.K_UP]: action[0] = 1.0
        if keys[pygame.K_LEFT]: action[1] = -1.0
        elif keys[pygame.K_RIGHT]: action[1] = 1.0
        if keys[pygame.K_SPACE]: action[2] = 1.0

        obs, reward, terminated, truncated, info = env.step(action) # Gym style unpack
        total_reward += reward
        env.render()

        if terminated or truncated:
            print(f"Finished! Score: {total_reward:.2f}")
            obs, _ = env.reset()
            total_reward = 0
        
        env.clock.tick(FPS)
    pygame.quit()

if __name__ == "__main__":
    manual_control()