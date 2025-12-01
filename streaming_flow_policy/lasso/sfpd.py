"""
Streaming Flow Policy - Deterministic variant for lasso task.
Follows pusht/sfpd.py patterns with adaptations for lasso state space.
"""
from typing import Dict, Optional
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from pydrake.all import PiecewisePolynomial
from torchdyn.core import NeuralODE


class StreamingFlowPolicyDeterministicLasso(nn.Module):
    """
    Deterministic Streaming Flow Policy for lasso robotic arm task.

    State space:
        - Observation (obs_dim=13): target(2) + eef_pos(3) + eef_quat(4) + rope_shape(3) + gripper(1)
        - Action (action_dim=7): eef_pos(3) + eef_quat(4)

    The EEF state is extracted from observation indices 2:9 (eef_pos + eef_quat).
    """
    # Indices for extracting EEF state from observation
    EEF_START_IDX = 2
    EEF_END_IDX = 9  # eef_pos(3) + eef_quat(4) = 7 dims

    def __init__(self,
                 velocity_net: nn.Module,
                 action_dim: int = 7,
                 pred_horizon: int = 16,
                 sigma: float = 0.0,
                 device: torch.device = 'cuda',
        ):
        """
        Args:
            velocity_net: Neural network that predicts velocity field.
            action_dim: Dimension of action space (default 7 for lasso).
            pred_horizon: Number of future actions to predict.
            sigma: Standard deviation of Gaussian noise added during training.
            device: Torch device.
        """
        super().__init__()
        self.velocity_net = velocity_net
        self.action_dim = action_dim
        self.device = device

        # Register as buffers for serialization
        self.register_buffer('pred_horizon', torch.tensor(pred_horizon, dtype=torch.int32))
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32))
        self.pred_horizon: Tensor
        self.sigma: Tensor

    def TransformTrainingDatum(self, datum: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Transform a training datum for flow matching.

        Args:
            datum: Dict with:
                'obs': (OBS_HORIZON, OBS_DIM) observations
                'action': (PRED_HORIZON, ACTION_DIM) action trajectory

        Returns:
            Dict with:
                'obs': (OBS_HORIZON, OBS_DIM) observations
                'x': (1, ACTION_DIM) position at sampled time
                'v': (1, ACTION_DIM) velocity at sampled time
                't': scalar time in [0, 1]
        """
        obs, action = datum['obs'], datum['action']
        OBS_HORIZON, OBS_DIM = obs.shape
        PRED_HORIZON, ACTION_DIM = action.shape
        assert PRED_HORIZON == self.pred_horizon.item()
        assert OBS_HORIZON == 2, "Currently only supports obs_horizon=2"

        # Extract current EEF state from last observation
        # obs[-1, 2:9] = eef_pos(3) + eef_quat(4)
        current_eef = obs[-1, self.EEF_START_IDX:self.EEF_END_IDX]

        # Ensure first action matches current EEF state
        # This is the same pattern as pusht - if mismatch, fix it
        if not np.allclose(action[0], current_eef, atol=1e-5):
            action = action.copy()
            action[0] = current_eef

        # Create piecewise linear trajectory from action sequence
        traj_times = np.linspace(0, 1, PRED_HORIZON)
        traj: PiecewisePolynomial = PiecewisePolynomial.FirstOrderHold(
            traj_times, action.T,
        )

        # Sample random time and evaluate trajectory
        time = np.float32(np.random.rand())
        x = traj.value(time).T  # (1, ACTION_DIM)
        v = traj.EvalDerivative(time).T  # (1, ACTION_DIM)

        # Add noise to position
        x = x + self.sigma.item() * np.random.randn(*x.shape)
        x = x.astype(np.float32)

        return {
            'obs': obs,
            'x': x.astype(np.float32),
            'v': v.astype(np.float32),
            't': time,
        }

    @torch.enable_grad()
    def Loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Compute flow matching loss.

        Args:
            batch: Dict with:
                'obs': (B, OBS_HORIZON, OBS_DIM)
                'x': (B, 1, ACTION_DIM) positions
                'v': (B, 1, ACTION_DIM) target velocities
                't': (B,) times

        Returns:
            Scalar MSE loss.
        """
        obs = batch['obs'].to(self.device)
        x = batch['x'].to(self.device)
        v = batch['v'].to(self.device)
        t = batch['t'].to(self.device)

        # Flatten observations for FiLM conditioning
        obs_cond = obs.flatten(start_dim=1)  # (B, OBS_HORIZON * OBS_DIM)

        # Predict velocity
        v_pred = self.velocity_net(
            sample=x, timestep=t, global_cond=obs_cond
        )

        # L2 loss
        loss = nn.functional.mse_loss(v_pred, v)
        return loss

    @torch.inference_mode()
    def __call__(self,
                 nobs: Tensor,
                 num_actions: Optional[int] = None,
                 integration_steps_per_action: int = 6,
    ) -> Tensor:
        """
        Generate action trajectory via ODE integration.

        Args:
            nobs: (OBS_HORIZON, OBS_DIM) normalized observations
            num_actions: Number of actions to predict (default: pred_horizon)
            integration_steps_per_action: ODE integration resolution

        Returns:
            (1, NUM_ACTIONS, ACTION_DIM) predicted action trajectory
        """
        obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)  # (1, OBS_HORIZON * OBS_DIM)

        # Setup ODE solver
        ode_solver = NeuralODE(
            vector_field=VectorFieldWrapper(self.velocity_net, obs_cond),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        # Initial state: current EEF position from last observation
        x0 = nobs[-1, self.EEF_START_IDX:self.EEF_END_IDX]  # (ACTION_DIM,)

        # Setup integration time span
        num_actions = num_actions or self.pred_horizon.item()
        assert 1 <= num_actions <= self.pred_horizon.item()
        num_future_actions = num_actions - 1
        t_max = num_future_actions / (self.pred_horizon.item() - 1)
        total_integration_steps = 1 + num_future_actions * integration_steps_per_action
        t_span = torch.linspace(0, t_max, total_integration_steps)

        # Select which integration steps correspond to actions
        select_action_indices = np.arange(
            0,
            total_integration_steps,
            integration_steps_per_action,
        )

        # Integrate ODE
        traj = ode_solver.trajectory(x=x0, t_span=t_span)  # (K, ACTION_DIM)

        # Extract action trajectory
        naction = traj[select_action_indices]  # (NUM_ACTIONS, ACTION_DIM)
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
            x: (ACTION_DIM,) position

        Returns:
            (ACTION_DIM,) velocity
        """
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, ACTION_DIM)
        v: Tensor = self.model(
            sample=x,
            timestep=t.repeat(x.shape[0]),
            global_cond=self.obs_cond,
        )  # (1, 1, ACTION_DIM)
        v = v.flatten()  # (ACTION_DIM,)
        return v
