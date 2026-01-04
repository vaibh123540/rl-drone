import numpy as np
import torch
import torch.nn as nn

def atanh(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse hyperbolic tangent function with numerical stability clamping.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: The inverse hyperbolic tangent of x.
    """
    eps = 1e-6
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def logit(x: torch.Tensor) -> torch.Tensor:
    """
    Logit function (inverse sigmoid) with numerical stability clamping.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: The logit of x.
    """
    eps = 1e-6
    x = torch.clamp(x, eps, 1 - eps)
    return torch.log(x) - torch.log1p(-x)


class ActorCritic(nn.Module):
    """
    Hybrid Actor-Critic network for continuous control with discrete shooting action.
    """

    def __init__(self, obs_dim: int, hidden: int = 256):
        """
        Initialize the network architecture.

        Args:
            obs_dim (int): Dimension of the observation space.
            hidden (int): Size of the hidden layers.
        """
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Action heads
        self.mu_thrust = nn.Linear(hidden, 1)
        self.logstd_thrust = nn.Parameter(torch.zeros(1))

        self.mu_steer = nn.Linear(hidden, 1)
        self.logstd_steer = nn.Parameter(torch.zeros(1))

        self.shoot_logits = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        """
        Forward pass through the network.

        Args:
            obs (torch.Tensor): Observation tensor.

        Returns:
            tuple: (mu_thrust, logstd_thrust, mu_steer, logstd_steer, shoot_logit, value)
        """
        h = self.trunk(obs)

        mu_t = self.mu_thrust(h)
        mu_s = self.mu_steer(h)

        logstd_t = torch.clamp(self.logstd_thrust, -5.0, 2.0).expand_as(mu_t)
        logstd_s = torch.clamp(self.logstd_steer, -5.0, 2.0).expand_as(mu_s)

        shoot_logit = self.shoot_logits(h)
        value = self.value_head(h).squeeze(-1)
        return mu_t, logstd_t, mu_s, logstd_s, shoot_logit, value

    @torch.no_grad()
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Estimate the value of a state.

        Args:
            obs (torch.Tensor): Observation tensor.

        Returns:
            torch.Tensor: Value estimate.
        """
        _, _, _, _, _, v = self.forward(obs)
        return v

    @torch.no_grad()
    def act(self, obs: torch.Tensor, shoot_mask: torch.Tensor):
        """
        Sample actions from the policy for rollout collection.

        Args:
            obs (torch.Tensor): Batch of observations (B, obs_dim).
            shoot_mask (torch.Tensor): Mask for shooting availability (B, 1).

        Returns:
            tuple:
                actions (torch.Tensor): Sampled actions (B, 3).
                logprob (torch.Tensor): Log probability of actions (B,).
                value (torch.Tensor): Value estimate (B,).
        """
        mu_t, logstd_t, mu_s, logstd_s, shoot_logit, value = self.forward(obs)

        std_t = torch.exp(logstd_t)
        std_s = torch.exp(logstd_s)

        # Reparameterization trick (though not strictly needed for sampling)
        z_t = mu_t + std_t * torch.randn_like(mu_t)
        z_s = mu_s + std_s * torch.randn_like(mu_s)

        thrust = torch.sigmoid(z_t)
        steer = torch.tanh(z_s)

        probs = torch.sigmoid(shoot_logit)
        shoot_sample = torch.bernoulli(probs)
        shoot = shoot_sample * shoot_mask

        # Calculate logprobs including Jacobian corrections for squashing functions
        logp_t = (-0.5 * (((z_t - mu_t) / (std_t + 1e-8)) ** 2) - logstd_t - 0.5 * np.log(2 * np.pi))
        logp_t = logp_t - torch.log(thrust * (1.0 - thrust) + 1e-8)

        logp_s = (-0.5 * (((z_s - mu_s) / (std_s + 1e-8)) ** 2) - logstd_s - 0.5 * np.log(2 * np.pi))
        logp_s = logp_s - torch.log(1.0 - steer * steer + 1e-8)

        logp_shoot = shoot_sample * torch.log(probs + 1e-8) + (1.0 - shoot_sample) * torch.log(1.0 - probs + 1e-8)
        logp_shoot = logp_shoot * shoot_mask

        logprob = (logp_t + logp_s + logp_shoot).squeeze(-1)
        actions = torch.cat([thrust, steer, shoot], dim=-1)

        return actions, logprob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, shoot_mask: torch.Tensor):
        """
        Evaluate actions for PPO updates.

        Args:
            obs (torch.Tensor): Batch of observations.
            actions (torch.Tensor): Batch of actions taken.
            shoot_mask (torch.Tensor): Mask for shooting availability.

        Returns:
            tuple:
                logprob (torch.Tensor): New log probabilities of the actions.
                entropy (torch.Tensor): Entropy of the policy distribution.
                value (torch.Tensor): Value estimates.
        """
        mu_t, logstd_t, mu_s, logstd_s, shoot_logit, value = self.forward(obs)

        std_t = torch.exp(logstd_t)
        std_s = torch.exp(logstd_s)

        thrust = actions[:, 0:1]
        steer = actions[:, 1:2]
        shoot = actions[:, 2:3]

        # Invert squashing functions to get back to Gaussian space
        z_t = logit(thrust)
        z_s = atanh(steer)

        logp_t = (-0.5 * (((z_t - mu_t) / (std_t + 1e-8)) ** 2) - logstd_t - 0.5 * np.log(2 * np.pi))
        logp_t = logp_t - torch.log(thrust * (1.0 - thrust) + 1e-8)

        logp_s = (-0.5 * (((z_s - mu_s) / (std_s + 1e-8)) ** 2) - logstd_s - 0.5 * np.log(2 * np.pi))
        logp_s = logp_s - torch.log(1.0 - steer * steer + 1e-8)

        probs = torch.sigmoid(shoot_logit)
        logp_shoot = shoot * torch.log(probs + 1e-8) + (1.0 - shoot) * torch.log(1.0 - probs + 1e-8)
        logp_shoot = logp_shoot * shoot_mask

        logprob = (logp_t + logp_s + logp_shoot).squeeze(-1)

        ent_t = (logstd_t + 0.5 * np.log(2 * np.pi * np.e)).sum(dim=-1).squeeze(-1)
        ent_s = (logstd_s + 0.5 * np.log(2 * np.pi * np.e)).sum(dim=-1).squeeze(-1)
        ent_b = (-(probs * torch.log(probs + 1e-8) + (1.0 - probs) * torch.log(1.0 - probs + 1e-8))).squeeze(-1)
        ent_b = ent_b * shoot_mask.squeeze(-1)

        entropy = ent_t + ent_s + ent_b
        return logprob, entropy, value