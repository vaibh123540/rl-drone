# Drone Combat: Reinforcement Learning with PPO

A 2D continuous control environment where an autonomous drone learns to hunt enemy targets while avoiding obstacles and friendly units using Proximal Policy Optimization (PPO).

### Agent Behavior

![Drone Demo](assets/drone-demo.mov)

The trained agent exhibits:
- **Active pursuit**: Aggressively chases enemies rather than camping
- **Collision avoidance**: Navigates around obstacles and friendlies
- **Predictive shooting**: Leads targets based on relative motion
- **Tactical positioning**: Maintains optimal engagement range (~150 units)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Watch the agent play

```bash
python play.py --weights runs/drone_ppo_weights.pt
```

By default the agent is deterministic

Key arguments:
- `--no-lidar`: Don't display the lidar for a clean UI
- `--stochastic`: View the stochastic action selection performance

### Training a New Agent

```bash
python train.py --total-timesteps 1000000 --num-envs 32
```

Key arguments:
- `--total-timesteps`: Total environment steps (default: 150,000)
- `--num-envs`: Number of parallel environments (default: 8)
- `--steps`: Rollout steps per environment (default: 128)
- `--lr`: Learning rate (default: 3e-4)
- `--hidden`: Hidden layer size (default: 128)
- `--run-name`: Name for saving weights and logs (default: "drone_ppo")
- `--resume`: Path to checkpoint file to resume training

### Evaluating a Trained Agent

```bash
python train.py --total-timesteps the_total_time_steps --resume runs/drone_ppo_ckpt.pt
```

### Play the game yourself

```bash
python environment.py
```

Controls: Arrow keys to move, Space to shoot

---

## Agent Performance

### Training Performance

![Training Return](assets/training_return.png)

The agent learns to achieve positive returns after ~2M timesteps, eventually reaching peak evaluation scores of **+440**. The high variance in training returns (blue) reflects the stochastic nature of exploration, while deterministic evaluation (red dashed line) shows steady improvement.

### Shooting Accuracy

![Hit Rate](assets/hit_rate.png)

Shooting accuracy improves from ~5% to **~45%** at peak performance, demonstrating the agent learned effective target tracking and engagement strategies.

---

## Environment

### Overview

The environment simulates a 2D combat scenario (800×800 pixels) where:
- **1 Blue Drone (Agent)**: Controllable player
- **5 Red Enemies**: Targets that respawn when destroyed
- **3 Green Friendlies**: Non-targets that penalize friendly fire
- **4 Gray Obstacles**: Static barriers

All entities except obstacles exhibit physics-based movement with velocity and momentum.

### Physics

The drone operates under the following dynamics:

**Velocity Update:**
```
v_{t+1} = drag × v_t + thrust × [cos(θ), sin(θ)]
```

**Position Update:**
```
x_{t+1} = x_t + v_{t+1}
```

**Constraints:**
- Max speed: 5.0 units/step
- Drag coefficient: 0.97
- Thrust power: 0.2
- Rotation speed: 0.1 rad/step
- Bullet cooldown: 5 steps (The agent was trained with a value of 20, reduced for demo)

### Observation Space

The observation is a 165-dimensional vector:

```python
[
    vel_x / MAX_SPEED,           # Normalized velocity (2D)
    vel_y / MAX_SPEED,
    sin(angle),                  # Heading direction (2D)
    cos(angle),
    cooldown / COOLDOWN_MAX,     # Shooting availability
    rel_enemy_x / SCREEN_WIDTH,  # Nearest enemy (2D)
    rel_enemy_y / SCREEN_HEIGHT,
    lidar_data                   # 32 rays × 4 channels = 128D
]
```

**LiDAR Sensing:**
- 32 rays uniformly distributed in 360°
- Each ray returns distance to the nearest entity (normalized to [0, 1])
- 4 channels: walls, obstacles, enemies, friendlies
- Ray length: 300 units

### Action Space

Continuous action vector `[thrust, steer, shoot]`:

| Action | Range | Distribution | Transformation |
|--------|-------|--------------|----------------|
| Thrust | [-1, 1] | Gaussian → Sigmoid | `thrust = σ(N(μ_t, σ_t²))` |
| Steer | [-1, 1] | Gaussian → Tanh | `steer = tanh(N(μ_s, σ_s²))` |
| Shoot | {0, 1} | Bernoulli | `shoot = Bernoulli(`&rho`_shoot)` |

The hybrid action space combines continuous control (thrust/steer) with discrete decision-making (shoot).

---

## MDP Formulation

### State Space

**State**: `s_t ∈ ℝ^165` containing:
- Drone kinematic state (velocity, heading, cooldown)
- Relative enemy position
- LiDAR perception (distance to entities in all directions)

**State Transition**: Deterministic physics + stochastic enemy motion

### Action Space

**Actions**: `a_t = (thrust, steer, shoot) ∈ [-1,1] × [-1,1] × {0,1}`

**Policy**: Stochastic, parameterized by neural network `π_θ(a|s)`

### Reward Function

The reward function balances multiple objectives:

```python
R(s, a, s') = R_terminal + R_shooting + R_shaping
```

**Terminal Penalties** (episode-ending events):
```
R_terminal = {
    -150  if hit wall
    -150  if hit obstacle  
    -150  if hit enemy (collision)
    -150  if hit friendly
    0     otherwise
}
```

**Shooting Outcomes**:
```
R_shooting = {
    +75   if enemy destroyed
    -200  if friendly fire
    -1    if shot missed
    0     otherwise
}
```

**Reward Shaping** (continuous, every step):

1. **Distance Penalty** (encourages pursuit):
   ```
   R_distance = -0.10 × (d_norm)²
   ```
   where `d_norm = min_enemy_dist / max_possible_dist`

2. **Approach Reward** (reward for closing distance):
   ```
   R_approach = {
       0.30 × Δd × speed_factor × dist_mult    if pursuing
       0.15 × Δd × dist_factor                  otherwise
   }
   ```
   Only rewards when actively moving toward enemy.

3. **Alignment Bonus** (reward for facing enemy while moving):
   ```
   R_align = 0.15 × (v̂ · ê) × (v/v_max) × pursuit_scale
   ```
   where `v̂` is velocity direction, `ê` is direction to enemy.

4. **Anti-Camping Penalties**:
   - **Stationary Penalty**: `-0.05` if speed < 0.5
   - **Position Variance**: `-0.03` if recent positions span < 100 units over 30 steps
   - **Distance Stagnation**: `-0.06` if distance variance < 900 over 20 steps

5. **Shooting Discipline**:
   ```
   R_shooting_discipline = {
       -0.30 × (1 - alignment)    if alignment < 0.5
       +0.50 × (alignment - 0.7)  if alignment > 0.7
   }
   ```

### Discount Factor

γ = 0.99 (favors long-term strategy over immediate rewards)

### Episode Termination

**Success**: Not explicitly defined (continuous task)
**Failure**: Collision with any entity or boundary
**Truncation**: 1000 timesteps

---

## Algorithm: Proximal Policy Optimization (PPO)

### Architecture

**Actor-Critic Network** (`ActorCritic` class):

```
Input (165D observation)
    ↓
[Linear(165 → 128) → Tanh]  ← Trunk (shared features)
    ↓
[Linear(128 → 128) → Tanh]
    ↓
    ├─→ μ_thrust (1D)         ← Thrust mean
    ├─→ log(σ_thrust)         ← Thrust log-std (learned parameter)
    ├─→ μ_steer (1D)          ← Steer mean  
    ├─→ log(σ_steer)          ← Steer log-std (learned parameter)
    ├─→ shoot_logit (1D)      ← Shooting probability (pre-sigmoid)
    └─→ V(s) (1D)             ← State value estimate
```

**Parameter Count**: ~44,000 trainable parameters

### Policy Distributions

The policy outputs parameters for three distributions:

**1. Thrust (Squashed Gaussian)**:
```
z_thrust ~ N(μ_t, σ_t²)
thrust = σ(z_thrust)  # Sigmoid squashing to [0, 1]
```

Log probability with Jacobian correction:
```
log π(thrust|s) = log N(z; μ_t, σ_t²) - log|dσ/dz|
                = -½[(z-μ_t)/σ_t]² - log(σ_t) - ½log(2π) 
                  - log[thrust(1-thrust)]
```

**2. Steer (Squashed Gaussian)**:
```
z_steer ~ N(μ_s, σ_s²)  
steer = tanh(z_steer)  # Tanh squashing to [-1, 1]
```

Log probability:
```
log π(steer|s) = log N(z; μ_s, σ_s²) - log|d tanh/dz|
               = -½[(z-μ_s)/σ_s]² - log(σ_s) - ½log(2π)
                 - log(1 - steer²)
```

**3. Shoot (Bernoulli)**:
```
p = σ(shoot_logit)
shoot ~ Bernoulli(p)
```

Log probability:
```
log π(shoot|s) = shoot·log(p) + (1-shoot)·log(1-p)
```

Joint policy:
```
log π(a|s) = log π(thrust|s) + log π(steer|s) + log π(shoot|s)
```

### PPO Clipped Objective

Standard policy gradient suffers from high variance and requires tiny learning rates. PPO addresses this with a clipped surrogate objective:

```
L^CLIP(θ) = 𝔼_t [min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)]
```

where:
- `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` is the probability ratio
- `Â_t` is the advantage estimate (GAE)
- `ε = 0.2` is the clipping parameter

**Intuition**: The `clip` operation prevents the policy from changing too drastically in a single update:
- If advantage is positive (good action), allow increase up to `1+ε`
- If advantage is negative (bad action), allow decrease down to `1-ε`
- Take the minimum to be conservative

### Value Function Loss

Clipped value loss to prevent large value function updates:

```
L^VF(θ) = 𝔼_t [max((V_θ(s_t) - V^target)², (V_clip - V^target)²)]
```

where:
```
V_clip = V_old(s_t) + clip(V_θ(s_t) - V_old(s_t), -ε, +ε)
V^target = Â_t + V_θ(s_t)  (TD target)
```

### Entropy Regularization

To encourage exploration, we add an entropy bonus:

```
H[π_θ] = 𝔼_s [𝔼_a~π_θ(-log π_θ(a|s))]
```

For our distributions:
- Gaussian: `H = ½log(2πe σ²)`
- Bernoulli: `H = -p log(p) - (1-p)log(1-p)`

### Complete Loss Function

```
L(θ) = -L^CLIP(θ) + c_v·L^VF(θ) - c_e·H[π_θ]
```

Hyperparameters:
- `c_v = 0.5` (value function coefficient)
- `c_e = 0.01` (entropy coefficient)

### Generalized Advantage Estimation (GAE)

GAE balances bias-variance tradeoff in advantage estimation:

```
Â_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
```

where `δ_t = r_t + γV(s_{t+1}) - V(s_t)` is the TD error.

**Parameters**:
- `γ = 0.99` (discount factor)
- `λ = 0.95` (GAE parameter, higher = less bias, more variance)

**Implementation** (backward recursion):
```python
A_t = δ_t + (γλ)(1 - done_t)·A_{t+1}
```

### Optimization: Adam

**Adam** (Adaptive Moment Estimation) combines the benefits of AdaGrad and RMSProp:

```
m_t = β₁·m_{t-1} + (1-β₁)·∇L_t        # First moment (momentum)
v_t = β₂·v_{t-1} + (1-β₂)·(∇L_t)²    # Second moment (adaptive lr)

m̂_t = m_t / (1 - β₁^t)                # Bias correction
v̂_t = v_t / (1 - β₂^t)

θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

**Why Adam?**
1. **Per-parameter learning rates**: Different parameters update at different speeds based on gradient history
2. **Momentum**: Accelerates learning in consistent directions
3. **Stability**: `√v̂_t` prevents exploding gradients
4. **Bias correction**: Accounts for initial zero moments

**Hyperparameters**:
- `α = 3e-4` (learning rate)
- `β₁ = 0.9` (momentum decay)
- `β₂ = 0.999` (variance decay)  
- `ε = 1e-5` (numerical stability)

### Training Procedure

1. **Collect Rollouts** (128 steps × 32 envs = 4096 transitions)
   ```python
   for step in range(128):
       action, logprob, value = policy.act(obs)
       next_obs, reward, done = env.step(action)
       buffer.add(obs, action, logprob, reward, done, value)
   ```

2. **Compute Returns & Advantages** using GAE

3. **Update Policy** (8 epochs × multiple minibatches)
   ```python
   for epoch in range(8):
       for minibatch in buffer.minibatches(512):
           new_logprob, entropy, new_value = policy.evaluate(minibatch)
           ratio = exp(new_logprob - old_logprob)
           
           # PPO loss
           loss = -min(ratio * adv, clip(ratio) * adv).mean()
           loss += 0.5 * value_loss - 0.01 * entropy.mean()
           
           loss.backward()
           clip_grad_norm(policy.parameters(), 0.5)
           optimizer.step()
   ```

4. **Repeat** until convergence

---

## Results & Outcomes

### Final Performance Metrics

After 10M training steps (32 parallel environments):

| Metric | Value |
|--------|-------|
| **Peak Evaluation Return** | +440 |
| **Average Return (final)** | +80-120 |
| **Shooting Accuracy** | 30-45% |
| **Friendly Fire Rate** | <2% |
| **Average Episode Length** | 600-800 steps |
| **Training Time** | ~3 hours (M1 MacBook) |
| **Sample Efficiency** | 10M steps to convergence |

### Behavioral Analysis

The trained agent demonstrates:

1. **Active Hunting**: Pursues enemies aggressively rather than defensive camping
2. **Spatial Awareness**: Uses LiDAR to navigate complex obstacle layouts
3. **Predictive Aiming**: Leads moving targets, achieving 45% hit rate
4. **Risk Management**: Balances aggression with collision avoidance
5. **Tactical Positioning**: Prefers 150-250 unit engagement range

### Training Dynamics

**Phase 1 (0-2M steps)**: Random exploration, negative returns
- Agent crashes frequently
- Shooting is random
- No coherent strategy

**Phase 2 (2M-5M steps)**: Suboptimal convergence
- Returns improve to ~0
- Agent discovers "turret strategy" (see Challenges)
- Hit rate plateaus at 15-18%

**Phase 3 (5M-8M steps)**: Breakthrough
- Returns jump to +100-200
- Active pursuit emerges
- Hit rate improves to 30-40%

**Phase 4 (8M-10M steps)**: Refinement
- Peak returns reach +440
- Stable policy with low variance
- Consistent high-accuracy shooting

---

## Challenges Faced

### 1. Suboptimal Policy Convergence: The "Turret Problem"

**Issue**: For ~3M training steps (timesteps 2M-5M), the agent converged to a highly suboptimal but locally stable policy:
- Remain stationary near the center
- Rotate continuously to track enemies
- Shoot whenever aligned

This "turret strategy" achieved small positive rewards (~+20 to +50) by:
- Minimizing collision risk (no movement)
- Exploiting the distance-based shaping reward (staying central)
- Occasionally hitting enemies through rotation

**Why it happened**:
1. **Reward Shaping Exploitation**: Initial reward function had:
   ```python
   R_distance = -0.05 × distance_norm  # Too weak
   ```
   The agent found it safer to farm small negative distance penalties than risk crashing for larger pursuit rewards.

2. **Insufficient Movement Penalties**: No explicit cost for staying stationary.

3. **Credit Assignment**: The sparse +75 enemy hit reward was overshadowed by dense shaping rewards, making the agent prioritize survival over aggression.

**Solution**: Multi-faceted reward redesign:

1. **Quadratic Distance Penalty**:
   ```python
   R_distance = -0.10 × (distance_norm)²  # Penalty grows with distance
   ```

2. **Conditional Approach Rewards**:
   ```python
   if pursuing and closing_distance:
       R += 0.30 × delta × speed_factor × distance_multiplier
   ```
   Only reward distance reduction when actively moving toward enemy.

3. **Anti-Camping Penalties**:
   ```python
   # Position clustering
   if avg_position_variance < 100.0:
       R -= 0.03
   
   # Distance stagnation  
   if distance_variance < 900.0 and min_dist > 150:
       R -= 0.06
   
   # Idle penalty
   if speed < 0.5:
       R -= 0.05
   ```

4. **Velocity Alignment Bonus**:
   ```python
   R_align = 0.15 × (velocity_dir · enemy_dir) × speed_norm × scale
   ```

**Outcome**: After these changes, the agent broke out of the turret strategy within 1M steps and began active pursuit.

### 2. NaN Issues During Late Training

**Issue**: Around timestep 8M, training occasionally crashed with NaNs propagating through the network.

**Root Causes**:

1. **Extreme Actions**: When agents became very confident, squashed Gaussian outputs approached the boundaries:
   ```python
   thrust → 1.0  ⇒  logit(thrust) → +∞
   steer → ±1.0  ⇒  atanh(steer) → ±∞
   ```

2. **Log-Probability Explosion**: The Jacobian correction term:
   ```python
   log π = ... - log[thrust(1-thrust)]
   ```
   becomes `-∞` when `thrust ∈ {0, 1}`.

3. **Advantage Outliers**: Rare high-reward episodes created huge advantages after normalization.

**Solutions Implemented**:

1. **Action Clamping in Inverse Functions**:
   ```python
   def logit(x):
       eps = 1e-6
       x = torch.clamp(x, eps, 1 - eps)  # Keep away from boundaries
       return torch.log(x) - torch.log1p(-x)
   
   def atanh(x):
       eps = 1e-6
       x = torch.clamp(x, -1 + eps, 1 - eps)
       return 0.5 * (torch.log1p(x) - torch.log1p(-x))
   ```

2. **Log-Std Bounds**:
   ```python
   logstd = torch.clamp(self.logstd, -5.0, 2.0)
   ```
   Limits standard deviation to `[0.0067, 7.39]`, preventing both over-confidence and extreme exploration.

3. **Gradient Clipping**:
   ```python
   nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
   ```

4. **Numerical Stability in Log Probs**:
   ```python
   logp = logp - torch.log(thrust * (1 - thrust) + 1e-8)  # Small epsilon
   ```

5. **Advantage Normalization**:
   ```python
   advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
   ```

**Outcome**: NaN issues were completely eliminated, allowing stable training to 10M+ steps.

### 3. Hyperparameter Sensitivity

**Challenge**: Small changes in hyperparameters had large impacts on convergence:
- Learning rate too high (5e-4) → policy collapse
- Entropy coefficient too low (0.001) → premature convergence to turret
- Clipping too tight (0.1) → slow learning
- GAE λ too low (0.8) → high variance, instability

**Solution**: Extensive grid search and adaptive tuning:
- Started with conservative values
- Increased entropy coefficient (0.01) to encourage exploration
- Used moderate clipping (0.2) for stable updates
- High GAE λ (0.95) for lower variance

---

## Learning Outcomes

### Technical Insights

1. **Reward Shaping is Critical**: Dense shaping rewards are necessary for sparse reward tasks, but they must be carefully designed to avoid reward hacking. The agent will always find the easiest path to positive reward.

2. **Exploration vs. Exploitation Balance**: Entropy regularization alone is insufficient for complex tasks. Anti-camping penalties and explicit diversity rewards are often necessary.

3. **Numerical Stability is Non-Negotiable**: Even mathematically correct implementations can fail due to floating-point arithmetic. Always clamp, always add epsilons, always clip gradients.

4. **Policy Parameterization Matters**: Squashed Gaussians enable bounded continuous control, but require careful Jacobian corrections. The choice of squashing function affects learning dynamics.

5. **PPO is Robust but Not Magic**: PPO's clipping mechanism provides stability, but poor reward design or architectural choices can still cause failure modes.

### Algorithmic Understanding

1. **GAE Reduces Variance**: Without GAE (λ=0, pure TD), training was extremely noisy. GAE (λ=0.95) smoothed learning significantly.

2. **Value Function Quality Matters**: A well-trained critic reduces policy gradient variance. Value loss clipping prevents value function overfitting.

3. **Minibatch Updates Break Correlation**: Shuffling and splitting rollouts into minibatches decorrelates samples, improving learning efficiency.

4. **Multiple Epochs Improve Sample Efficiency**: Reusing samples (8 epochs) extracts more information from expensive environment interactions.

### Practical Lessons

1. **Vectorized Environments are Essential**: 32 parallel environments improved sample collection speed by ~25× compared to single environment.

2. **Logging is Critical**: Detailed metrics (hit rate, termination types, alignment) revealed behavioral issues that total return alone would miss.

3. **Checkpointing Saves Time**: Several times, training diverged after 5M+ steps. Loading from checkpoints prevented complete restarts.

4. **Deterministic Evaluation**: Stochastic policy evaluation is noisy. Deterministic evaluation (using mean actions) provides clearer progress signals.

5. **Visualization Guides Debugging**: Watching the agent play revealed the turret problem instantly, while metrics alone suggested slow but steady progress.

### Domain Knowledge

1. **Physics-Based RL is Sensitive**: Drag, max speed, and thrust power constants affect learning difficulty. Too much drag makes pursuit impossible; too little makes collision avoidance trivial.

2. **Sensor Design Impacts Learning**: 32 LiDAR rays provided sufficient spatial awareness. Fewer rays (<16) made navigation difficult; more rays (>64) slowed training without clear benefit.

3. **Cooldown Mechanics**: The 5-step shooting cooldown was essential. Without it, agents spam-shot constantly. Too long (>10), and shooting became rare.

4. **Enemy Respawning**: Respawning destroyed enemies maintains task difficulty. Fixed enemies would make the environment progressively easier during an episode.

---

## Future Work

Potential improvements and extensions:

1. **Curriculum Learning**: Start with fewer enemies/obstacles, gradually increase difficulty
2. **Multi-Agent**: Train multiple drones cooperatively or competitively
3. **Recurrent Policies**: LSTM/GRU for temporal reasoning (e.g., tracking occluded enemies)
4. **Transformer Architectures**: Attention mechanisms for multi-entity tracking
5. **Visual Observations**: Replace LiDAR with pixel-based input (CNN encoder)
6. **Hierarchical RL**: High-level strategy network + low-level control network
7. **Model-Based RL**: Learn environment dynamics, plan actions
8. **Automatic Reward Tuning**: Meta-learn reward coefficients

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{drone_combat_rl2026,
  author = {[Your Name]},
  title = {Drone Combat: Continuous Control with PPO},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/[username]/drone-combat-rl}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

- **Gymnasium**: Environment interface
- **PyTorch**: Deep learning framework  
- **Pygame**: Visualization
- **PPO Paper**: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
- **GAE Paper**: Schulman et al. (2015) - "High-Dimensional Continuous Control Using Generalized Advantage Estimation"