"""
Streaming Flow Policy - Stochastic variant for lasso task.
Follows pusht/sfps.py patterns with adaptations for lasso state space.
"""
from typing import Dict, Optional
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from pydrake.all import PiecewisePolynomial
from torchdyn.core import NeuralODE


class StreamingFlowPolicyStochasticLasso(nn.Module):
    """
    Stochastic Streaming Flow Policy for lasso robotic arm task.

    Uses a dual-path flow with action and latent dimensions to enable
    stochastic predictions - useful when multiple valid trajectories exist.

    Conditional flow:
    - At time t=0, we sample:
        - a₀ ~ N(ξ(0), σ₀)
        - z₀ ~ N(0, 1)

    - Flow trajectory at time t:
        - a(t) = a₀ + (ξ(t) - ξ(0)) + (σᵣt) z₀
        - z(t) = (1 - (1-σ₁)t)z₀ + tξ(t)

    - Conditional velocity field:
        - va(a, z, t) = ξ̇(t) + σᵣz₀
        - vz(a, z, t) = ξ(t) + tξ̇(t) - (1-σ₁)z₀

    State space:
        - Observation (obs_dim=13): target(2) + eef_pos(3) + eef_quat(4) + rope_shape(3) + gripper(1)
        - Action (action_dim=7): eef_pos(3) + eef_quat(4)
    """
    # Indices for extracting EEF state from observation
    EEF_START_IDX = 2
    EEF_END_IDX = 9  # eef_pos(3) + eef_quat(4) = 7 dims

    def __init__(self,
                 velocity_net: nn.Module,
                 action_dim: int = 7,
                 σ0: float = 0.0,
                 σ1: float = 0.0,
                 pred_horizon: int = 16,
                 device: torch.device = 'cuda',
        ):
        """
        Args:
            velocity_net: Neural network that predicts velocity field.
                Should have fc_timesteps=2 for dual-path flow.
            action_dim: Dimension of action space (default 7 for lasso).
            σ0: Standard deviation at t=0.
            σ1: Standard deviation at t=1.
            pred_horizon: Number of future actions to predict.
            device: Torch device.
        """
        super().__init__()
        assert 0 <= σ0 <= σ1, "σ0 must be less than or equal to σ1"
        σr = np.sqrt(np.square(σ1) - np.square(σ0))

        self.velocity_net = velocity_net
        self.action_dim = action_dim
        self.device = device

        # Register as buffers for serialization
        self.register_buffer('pred_horizon', torch.tensor(pred_horizon, dtype=torch.int32))
        self.register_buffer('σ0', torch.tensor(σ0, dtype=torch.float32))
        self.register_buffer('σ1', torch.tensor(σ1, dtype=torch.float32))
        self.register_buffer('σr', torch.tensor(σr, dtype=torch.float32))
        self.pred_horizon: Tensor
        self.σ0: Tensor
        self.σ1: Tensor
        self.σr: Tensor

    def TransformTrainingDatum(self, datum: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Transform a training datum for stochastic flow matching.

        Args:
            datum: Dict with:
                'obs': (OBS_HORIZON, OBS_DIM) observations
                'action': (PRED_HORIZON, ACTION_DIM) action trajectory

        Returns:
            Dict with:
                'obs': (OBS_HORIZON, OBS_DIM) observations
                'a': (1, ACTION_DIM) action at sampled time
                'z': (1, ACTION_DIM) latent at sampled time
                'va': (1, ACTION_DIM) target action velocity
                'vz': (1, ACTION_DIM) target latent velocity
                't': scalar time in [0, 1]
        """
        obs, action = datum['obs'], datum['action']
        OBS_HORIZON, OBS_DIM = obs.shape
        PRED_HORIZON, ACTION_DIM = action.shape
        assert PRED_HORIZON == self.pred_horizon.item()
        assert OBS_HORIZON == 2, "Currently only supports obs_horizon=2"

        # Extract current EEF state from last observation
        current_eef = obs[-1, self.EEF_START_IDX:self.EEF_END_IDX]

        # Ensure first action matches current EEF state
        if not np.allclose(action[0], current_eef, atol=1e-5):
            action = action.copy()
            action[0] = current_eef

        # Create piecewise linear trajectory
        traj_times = np.linspace(0, 1, PRED_HORIZON)
        traj: PiecewisePolynomial = PiecewisePolynomial.FirstOrderHold(
            traj_times, action.T,
        )

        # Sample random time and evaluate trajectory
        time = np.float32(np.random.rand())
        ξt = traj.value(time).T  # (1, ACTION_DIM)
        ξ̇t = traj.EvalDerivative(time).T  # (1, ACTION_DIM)

        σ0 = self.σ0.item()
        σ1 = self.σ1.item()
        σr = self.σr.item()

        # Sample z0 from N(0, 1)
        z0 = np.random.randn(1, ACTION_DIM)

        # Sample action at time t
        ε_a0 = σ0 * np.random.randn(1, ACTION_DIM)
        at = ξt + ε_a0 + σr * time * z0

        # Compute latent at time t
        zt = (1 - (1 - σ1) * time) * z0 + time * ξt

        # Compute conditional flow velocities
        va = ξ̇t + σr * z0
        vz = ξt + time * ξ̇t - (1 - σ1) * z0

        return {
            'obs': obs,
            'a': at.astype(np.float32),
            'z': zt.astype(np.float32),
            'va': va.astype(np.float32),
            'vz': vz.astype(np.float32),
            't': time,
        }

    @torch.enable_grad()
    def Loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Compute stochastic flow matching loss.

        Args:
            batch: Dict with:
                'obs': (B, OBS_HORIZON, OBS_DIM)
                'a': (B, 1, ACTION_DIM) actions
                'z': (B, 1, ACTION_DIM) latents
                'va': (B, 1, ACTION_DIM) target action velocities
                'vz': (B, 1, ACTION_DIM) target latent velocities
                't': (B,) times

        Returns:
            Scalar MSE loss over both action and latent velocities.
        """
        obs = batch['obs'].to(self.device)
        a = batch['a'].to(self.device)
        z = batch['z'].to(self.device)
        va = batch['va'].to(self.device)
        vz = batch['vz'].to(self.device)
        t = batch['t'].to(self.device)

        # Flatten observations for FiLM conditioning
        obs_cond = obs.flatten(start_dim=1)

        # Concatenate action and latent for dual-path input
        x = torch.cat((a, z), dim=-2)  # (B, 2, ACTION_DIM)
        v_target = torch.cat((va, vz), dim=-2)  # (B, 2, ACTION_DIM)

        # Predict velocity
        v_pred = self.velocity_net(
            sample=x, timestep=t, global_cond=obs_cond
        )  # (B, 2, ACTION_DIM)

        # L2 loss
        loss = nn.functional.mse_loss(v_pred, v_target)
        return loss

    @torch.inference_mode()
    def __call__(self,
                 nobs: Tensor,
                 num_actions: Optional[int] = None,
                 integration_steps_per_action: int = 6,
    ) -> Tensor:
        """
        Generate action trajectory via stochastic ODE integration.

        Args:
            nobs: (OBS_HORIZON, OBS_DIM) normalized observations
            num_actions: Number of actions to predict (default: pred_horizon)
            integration_steps_per_action: ODE integration resolution

        Returns:
            (1, NUM_ACTIONS, ACTION_DIM) predicted action trajectory
        """
        obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

        # Setup ODE solver
        ode_solver = NeuralODE(
            vector_field=VectorFieldWrapper(self.velocity_net, obs_cond),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        # Initial action: current EEF position from last observation
        a0 = nobs[-1, self.EEF_START_IDX:self.EEF_END_IDX]  # (ACTION_DIM,)

        # Sample latent variable -- this is the stochastic step!
        z0 = torch.randn_like(a0)  # (ACTION_DIM,)

        # Setup integration time span
        num_actions = num_actions or self.pred_horizon.item()
        assert 1 <= num_actions <= self.pred_horizon.item()
        num_future_actions = num_actions - 1
        t_max = num_future_actions / (self.pred_horizon.item() - 1)
        total_integration_steps = 1 + num_future_actions * integration_steps_per_action
        t_span = torch.linspace(0, t_max, total_integration_steps)

        select_action_indices = np.arange(
            0,
            total_integration_steps,
            integration_steps_per_action,
        )

        # Initial state: stack action and latent
        x0 = torch.stack((a0, z0), dim=0)  # (2, ACTION_DIM)

        # Integrate ODE
        traj = ode_solver.trajectory(x=x0, t_span=t_span)  # (K, 2, ACTION_DIM)

        # Extract action trajectory (first component of dual path)
        naction = traj[select_action_indices, 0, :]  # (NUM_ACTIONS, ACTION_DIM)
        naction = naction.unsqueeze(0)  # (1, NUM_ACTIONS, ACTION_DIM)
        return naction


class VectorFieldWrapper(nn.Module):
    """Wraps velocity network for torchdyn ODE solver compatibility."""

    def __init__(self, model: nn.Module, obs_cond: Tensor):
        super().__init__()
        self.model = model
        self.obs_cond = obs_cond

    def forward(self, t: Tensor, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Args:
            t: Scalar time
            x: (2, ACTION_DIM) position [action, latent]

        Returns:
            (2, ACTION_DIM) velocity [action_vel, latent_vel]
        """
        x = x.unsqueeze(0)  # (1, 2, ACTION_DIM)
        v: Tensor = self.model(
            sample=x,
            timestep=t.repeat(x.shape[0]),
            global_cond=self.obs_cond,
        )  # (1, 2, ACTION_DIM)
        v = v.squeeze(0)  # (2, ACTION_DIM)
        return v
