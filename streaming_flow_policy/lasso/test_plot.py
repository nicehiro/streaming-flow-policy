"""
Test script for lasso flow visualization.

Usage:
    uv run python -m streaming_flow_policy.lasso.test_plot
"""
import torch
import matplotlib.pyplot as plt

from streaming_flow_policy.lasso.dataset import LassoStateDataset
from streaming_flow_policy.pusht.dp_state_notebook.network import ConditionalUnet1D
from streaming_flow_policy.lasso.sfpd import StreamingFlowPolicyDeterministicLasso
from streaming_flow_policy.lasso.plot import (
    plot_lasso_flow_position,
    plot_lasso_flow_orientation,
    plot_lasso_flow_position_3d,
)


def main():
    # Parameters (must match training)
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    obs_dim = 13
    action_dim = 7

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset for stats and sample observation
    dataset = LassoStateDataset(
        zarr_path='mock_lasso_dataset.zarr',
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
    )
    action_stats = dataset.stats['action']
    print(f"Action stats: min={action_stats['min']}, max={action_stats['max']}")

    # Create velocity network (same architecture as training)
    velocity_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        fc_timesteps=1,  # Single path for deterministic
    )

    # Create policy wrapper
    policy = StreamingFlowPolicyDeterministicLasso(
        velocity_net=velocity_net,
        action_dim=action_dim,
        pred_horizon=pred_horizon,
        sigma=0.1,
        device=device,
    )

    # Load trained weights
    state_dict = torch.load('models/lasso_sfpd.pth', map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    velocity_net = policy.velocity_net
    print("Loaded model weights")

    # Get a sample observation from dataset
    sample = dataset[0]
    obs = torch.tensor(sample['obs'], dtype=torch.float32)
    print(f"Observation shape: {obs.shape}")

    # Generate plots
    print("Generating position flow plot...")
    fig1 = plot_lasso_flow_position(
        velocity_net, obs, action_stats,
        num_trajectories=5,
        num_grid_points=15,
        num_traj_steps=50,
        stochastic=False,
    )
    fig1.savefig('lasso_flow_position.png', dpi=150, bbox_inches='tight')
    print("Saved: lasso_flow_position.png")

    print("Generating orientation flow plot...")
    fig2 = plot_lasso_flow_orientation(
        velocity_net, obs, action_stats,
        num_trajectories=5,
        num_grid_points=15,
        num_traj_steps=50,
        stochastic=False,
    )
    fig2.savefig('lasso_flow_orientation.png', dpi=150, bbox_inches='tight')
    print("Saved: lasso_flow_orientation.png")

    print("Generating 3D position flow plot...")
    fig3 = plot_lasso_flow_position_3d(
        velocity_net, obs, action_stats,
        num_trajectories=5,
        num_velocity_samples=5,
        num_traj_steps=50,
        stochastic=False,
    )
    fig3.savefig('lasso_flow_position_3d.png', dpi=150, bbox_inches='tight')
    print("Saved: lasso_flow_position_3d.png")

    plt.close('all')
    print("Done!")


if __name__ == '__main__':
    main()
