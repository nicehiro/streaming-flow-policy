"""
Visualization tools for lasso flow policy.

Provides functions to visualize velocity fields and sampled trajectories
for the 7-dimensional action space (3D position + 4D quaternion orientation).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from torchdyn.core import NeuralODE


# Dimension labels
POSITION_LABELS = ["eef_x", "eef_y", "eef_z"]
ORIENTATION_LABELS = ["quat_w", "quat_x", "quat_y", "quat_z"]
POSITION_DIMS = [0, 1, 2]
ORIENTATION_DIMS = [3, 4, 5, 6]


class _VectorFieldWrapperDeterministic(nn.Module):
    """Wraps velocity network for torchdyn ODE solver (deterministic single-path)."""

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


class _VectorFieldWrapperStochastic(nn.Module):
    """Wraps velocity network for torchdyn ODE solver (stochastic dual-path)."""

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


def sample_trajectories(
    velocity_net: nn.Module,
    obs: Tensor,
    num_trajectories: int,
    action_dim: int = 7,
    num_steps: int = 100,
    eef_start_idx: int = 2,
    eef_end_idx: int = 9,
    stochastic: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample trajectories by integrating through the learned flow.

    Args:
        velocity_net: Trained velocity network.
        obs: Observation tensor of shape (OBS_HORIZON, OBS_DIM).
        num_trajectories: Number of trajectories to sample.
        action_dim: Dimension of action space.
        num_steps: Number of integration steps.
        eef_start_idx: Start index of EEF state in observation.
        eef_end_idx: End index of EEF state in observation.
        stochastic: If True, use dual-path (action + latent) for stochastic model.
                    If False, use single-path for deterministic model.

    Returns:
        trajectories: Array of shape (num_trajectories, num_steps, action_dim)
        times: Array of shape (num_steps,)
    """
    device = next(velocity_net.parameters()).device
    obs = obs.to(device)
    obs_cond = obs.unsqueeze(0).flatten(start_dim=1)  # (1, OBS_HORIZON * OBS_DIM)

    # Initial action from observation
    a0 = obs[-1, eef_start_idx:eef_end_idx]  # (ACTION_DIM,)

    trajectories = []
    times = np.linspace(0, 1, num_steps)
    t_span = torch.tensor(times, dtype=torch.float32, device=device)

    # Choose wrapper based on model type
    if stochastic:
        wrapper_cls = _VectorFieldWrapperStochastic
    else:
        wrapper_cls = _VectorFieldWrapperDeterministic

    for _ in range(num_trajectories):
        # Setup ODE solver
        ode_solver = NeuralODE(
            vector_field=wrapper_cls(velocity_net, obs_cond),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        if stochastic:
            # Sample latent variable for stochastic model
            z0 = torch.randn_like(a0)  # (ACTION_DIM,)
            x0 = torch.stack((a0, z0), dim=0)  # (2, ACTION_DIM)

            # Integrate
            with torch.no_grad():
                traj = ode_solver.trajectory(
                    x=x0, t_span=t_span
                )  # (num_steps, 2, ACTION_DIM)

            # Extract action trajectory (first component)
            action_traj = traj[:, 0, :].cpu().numpy()  # (num_steps, ACTION_DIM)
        else:
            # Deterministic model: single path
            x0 = a0  # (ACTION_DIM,)

            # Integrate
            with torch.no_grad():
                traj = ode_solver.trajectory(
                    x=x0, t_span=t_span
                )  # (num_steps, ACTION_DIM)

            action_traj = traj.cpu().numpy()  # (num_steps, ACTION_DIM)

        trajectories.append(action_traj)

    return np.stack(trajectories, axis=0), times


def evaluate_velocity_field(
    velocity_net: nn.Module,
    obs: Tensor,
    dim_idx: int,
    action_grid: np.ndarray,
    time_grid: np.ndarray,
    reference_action: np.ndarray,
    action_dim: int = 7,
    stochastic: bool = False,
) -> np.ndarray:
    """
    Evaluate velocity field at a grid of (action, time) points for one dimension.

    Args:
        velocity_net: Trained velocity network.
        obs: Observation tensor of shape (OBS_HORIZON, OBS_DIM).
        dim_idx: Which action dimension to vary (0-6).
        action_grid: 1D array of action values for the dimension.
        time_grid: 1D array of time values.
        reference_action: Reference action for other dimensions, shape (ACTION_DIM,).
        action_dim: Dimension of action space.
        stochastic: If True, use dual-path input for stochastic model.

    Returns:
        velocity: Array of shape (len(time_grid), len(action_grid)) with velocity
                  component for the specified dimension.
    """
    device = next(velocity_net.parameters()).device
    obs = obs.to(device)
    obs_cond = obs.unsqueeze(0).flatten(start_dim=1)  # (1, OBS_HORIZON * OBS_DIM)

    # Create meshgrid
    T, A = np.meshgrid(time_grid, action_grid, indexing="ij")  # (T, A)
    T_flat = T.flatten()
    A_flat = A.flatten()

    velocities = []

    with torch.no_grad():
        for t_val, a_val in zip(T_flat, A_flat):
            # Build action vector with varied dimension
            action = reference_action.copy()
            action[dim_idx] = a_val

            t = torch.tensor([t_val], dtype=torch.float32, device=device)

            if stochastic:
                # Stochastic: dual-path input
                latent = np.zeros(action_dim)
                x = np.stack([action, latent], axis=0)  # (2, ACTION_DIM)
                x = torch.tensor(x, dtype=torch.float32, device=device)
                x = x.unsqueeze(0)  # (1, 2, ACTION_DIM)

                v = velocity_net(
                    sample=x, timestep=t, global_cond=obs_cond
                )  # (1, 2, ACTION_DIM)
                v_action = v[0, 0, dim_idx].cpu().numpy()
            else:
                # Deterministic: single-path input
                x = torch.tensor(action, dtype=torch.float32, device=device)
                x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, ACTION_DIM)

                v = velocity_net(
                    sample=x, timestep=t, global_cond=obs_cond
                )  # (1, 1, ACTION_DIM)
                v_action = v[0, 0, dim_idx].cpu().numpy()

            velocities.append(v_action)

    velocities = np.array(velocities).reshape(T.shape)
    return velocities


def plot_lasso_flow_position(
    velocity_net: nn.Module,
    obs: Tensor,
    action_stats: Dict[str, np.ndarray],
    num_trajectories: int = 10,
    num_grid_points: int = 20,
    num_traj_steps: int = 100,
    figsize: Tuple[float, float] = (12, 4),
    dpi: int = 150,
    stochastic: bool = False,
) -> plt.Figure:
    """
    Plot velocity field and trajectories for position dimensions (eef_x, eef_y, eef_z).

    Args:
        velocity_net: Trained velocity network.
        obs: Observation tensor of shape (OBS_HORIZON, OBS_DIM).
        action_stats: Dict with 'min' and 'max' arrays for action normalization bounds.
        num_trajectories: Number of trajectories to sample.
        num_grid_points: Number of grid points for velocity field.
        num_traj_steps: Number of steps for trajectory integration.
        figsize: Figure size.
        dpi: Figure DPI.
        stochastic: If True, use stochastic (dual-path) model. If False, deterministic.

    Returns:
        Matplotlib figure with 3 subplots.
    """
    # Sample trajectories
    trajectories, times = sample_trajectories(
        velocity_net,
        obs,
        num_trajectories,
        num_steps=num_traj_steps,
        stochastic=stochastic,
    )  # (N, T, 7), (T,)

    # Reference action: mean of first trajectory point
    reference_action = trajectories[:, 0, :].mean(axis=0)  # (7,)

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    for i, (dim_idx, label) in enumerate(zip(POSITION_DIMS, POSITION_LABELS)):
        ax = axes[i]

        # Get action bounds for this dimension
        a_min = action_stats["min"][dim_idx]
        a_max = action_stats["max"][dim_idx]

        # Create grids
        action_grid = np.linspace(a_min, a_max, num_grid_points)
        time_grid = np.linspace(0, 1, num_grid_points)

        # Evaluate velocity field
        velocities = evaluate_velocity_field(
            velocity_net,
            obs,
            dim_idx,
            action_grid,
            time_grid,
            reference_action,
            stochastic=stochastic,
        )

        # Create meshgrid for plotting
        T, A = np.meshgrid(time_grid, action_grid, indexing="ij")

        # Plot quiver (velocity field)
        # v_t is always 1 (time progresses uniformly), v_a is the velocity
        ax.quiver(
            A,
            T,
            velocities,
            np.ones_like(velocities),
            color="gray",
            alpha=0.6,
            pivot="mid",
            scale=30,
            width=0.004,
            headwidth=3,
            headlength=4,
        )

        # Plot trajectories
        for traj in trajectories:
            ax.plot(traj[:, dim_idx], times, color="blue", alpha=0.5, linewidth=1)

        ax.set_xlabel(label)
        ax.set_ylabel("Time")
        ax.set_xlim(a_min, a_max)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label} Flow")

    plt.tight_layout()
    return fig


def plot_lasso_flow_orientation(
    velocity_net: nn.Module,
    obs: Tensor,
    action_stats: Dict[str, np.ndarray],
    num_trajectories: int = 10,
    num_grid_points: int = 20,
    num_traj_steps: int = 100,
    figsize: Tuple[float, float] = (16, 4),
    dpi: int = 150,
    stochastic: bool = False,
) -> plt.Figure:
    """
    Plot velocity field and trajectories for orientation dimensions (quaternion).

    Args:
        velocity_net: Trained velocity network.
        obs: Observation tensor of shape (OBS_HORIZON, OBS_DIM).
        action_stats: Dict with 'min' and 'max' arrays for action normalization bounds.
        num_trajectories: Number of trajectories to sample.
        num_grid_points: Number of grid points for velocity field.
        num_traj_steps: Number of steps for trajectory integration.
        figsize: Figure size.
        dpi: Figure DPI.
        stochastic: If True, use stochastic (dual-path) model. If False, deterministic.

    Returns:
        Matplotlib figure with 4 subplots.
    """
    # Sample trajectories
    trajectories, times = sample_trajectories(
        velocity_net,
        obs,
        num_trajectories,
        num_steps=num_traj_steps,
        stochastic=stochastic,
    )  # (N, T, 7), (T,)

    # Reference action: mean of first trajectory point
    reference_action = trajectories[:, 0, :].mean(axis=0)  # (7,)

    fig, axes = plt.subplots(1, 4, figsize=figsize, dpi=dpi)

    for i, (dim_idx, label) in enumerate(zip(ORIENTATION_DIMS, ORIENTATION_LABELS)):
        ax = axes[i]

        # Get action bounds for this dimension
        a_min = action_stats["min"][dim_idx]
        a_max = action_stats["max"][dim_idx]

        # Create grids
        action_grid = np.linspace(a_min, a_max, num_grid_points)
        time_grid = np.linspace(0, 1, num_grid_points)

        # Evaluate velocity field
        velocities = evaluate_velocity_field(
            velocity_net,
            obs,
            dim_idx,
            action_grid,
            time_grid,
            reference_action,
            stochastic=stochastic,
        )

        # Create meshgrid for plotting
        T, A = np.meshgrid(time_grid, action_grid, indexing="ij")

        # Plot quiver (velocity field)
        ax.quiver(
            A,
            T,
            velocities,
            np.ones_like(velocities),
            color="gray",
            alpha=0.6,
            pivot="mid",
            scale=30,
            width=0.004,
            headwidth=3,
            headlength=4,
        )

        # Plot trajectories
        for traj in trajectories:
            ax.plot(traj[:, dim_idx], times, color="blue", alpha=0.5, linewidth=1)

        ax.set_xlabel(label)
        ax.set_ylabel("Time")
        ax.set_xlim(a_min, a_max)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label} Flow")

    plt.tight_layout()
    return fig


def plot_lasso_flow_position_3d(
    velocity_net: nn.Module,
    obs: Tensor,
    action_stats: Dict[str, np.ndarray],
    num_trajectories: int = 10,
    num_velocity_samples: int = 5,
    num_traj_steps: int = 100,
    figsize: Tuple[float, float] = (8, 8),
    dpi: int = 150,
    stochastic: bool = False,
) -> plt.Figure:
    """
    Plot 3D trajectories with velocity vectors for position dimensions.

    Args:
        velocity_net: Trained velocity network.
        obs: Observation tensor of shape (OBS_HORIZON, OBS_DIM).
        action_stats: Dict with 'min' and 'max' arrays for action normalization bounds.
        num_trajectories: Number of trajectories to sample.
        num_velocity_samples: Number of points per trajectory to show velocity vectors.
        num_traj_steps: Number of steps for trajectory integration.
        figsize: Figure size.
        dpi: Figure DPI.
        stochastic: If True, use stochastic (dual-path) model. If False, deterministic.

    Returns:
        Matplotlib figure with 3D plot.
    """
    # Sample trajectories
    trajectories, times = sample_trajectories(
        velocity_net,
        obs,
        num_trajectories,
        num_steps=num_traj_steps,
        stochastic=stochastic,
    )  # (N, T, 7), (T,)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Color normalization for time
    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.cm.viridis

    # Indices for velocity samples along trajectory
    velocity_indices = np.linspace(
        0, num_traj_steps - 1, num_velocity_samples, dtype=int
    )

    device = next(velocity_net.parameters()).device
    obs_tensor = obs.to(device)
    obs_cond = obs_tensor.unsqueeze(0).flatten(start_dim=1)

    for traj in trajectories:
        # Extract position dimensions
        x = traj[:, 0]  # eef_x
        y = traj[:, 1]  # eef_y
        z = traj[:, 2]  # eef_z

        # Plot trajectory with time-colored segments
        for j in range(len(times) - 1):
            color = cmap(norm(times[j]))
            ax.plot(
                x[j : j + 2], y[j : j + 2], z[j : j + 2], color=color, linewidth=1.5
            )

        # Plot velocity vectors at selected points
        for idx in velocity_indices:
            t_val = times[idx]
            action = traj[idx]

            t_input = torch.tensor([t_val], dtype=torch.float32, device=device)

            if stochastic:
                # Stochastic: dual-path input
                latent = np.zeros_like(action)
                x_input = np.stack([action, latent], axis=0)
                x_input = torch.tensor(x_input, dtype=torch.float32, device=device)
                x_input = x_input.unsqueeze(0)

                with torch.no_grad():
                    v = velocity_net(
                        sample=x_input, timestep=t_input, global_cond=obs_cond
                    )
                    v_action = v[0, 0, :3].cpu().numpy()
            else:
                # Deterministic: single-path input
                x_input = torch.tensor(action, dtype=torch.float32, device=device)
                x_input = x_input.unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    v = velocity_net(
                        sample=x_input, timestep=t_input, global_cond=obs_cond
                    )
                    v_action = v[0, 0, :3].cpu().numpy()

            # Scale velocity for visualization
            scale = 0.1
            ax.quiver(
                action[0],
                action[1],
                action[2],
                v_action[0] * scale,
                v_action[1] * scale,
                v_action[2] * scale,
                color="red",
                alpha=0.7,
                arrow_length_ratio=0.3,
            )

    # Set labels
    ax.set_xlabel("eef_x")
    ax.set_ylabel("eef_y")
    ax.set_zlabel("eef_z")
    ax.set_title("3D Position Trajectories with Velocity Vectors")

    # Set axis limits from stats
    ax.set_xlim(action_stats["min"][0], action_stats["max"][0])
    ax.set_ylim(action_stats["min"][1], action_stats["max"][1])
    ax.set_zlim(action_stats["min"][2], action_stats["max"][2])

    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("Time")

    return fig
