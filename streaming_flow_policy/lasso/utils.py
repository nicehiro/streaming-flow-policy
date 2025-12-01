"""
Utility functions for lasso task.
"""
import numpy as np
import zarr
from typing import Tuple


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion(s) to unit length.

    Args:
        q: Quaternion(s) of shape (..., 4)

    Returns:
        Normalized quaternion(s) of same shape
    """
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / (norm + 1e-8)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (Hamilton product).
    Quaternion format: [w, x, y, z]

    Args:
        q1, q2: Quaternions of shape (..., 4)

    Returns:
        Product quaternion of shape (..., 4)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.stack([w, x, y, z], axis=-1)


def create_mock_dataset(
    zarr_path: str,
    num_episodes: int = 10,
    episode_length: int = 100,
    obs_dim: int = 13,
    action_dim: int = 7,
    seed: int = 42,
) -> None:
    """
    Create a mock zarr dataset for testing.

    The mock data has:
        - Random but smooth trajectories
        - Normalized quaternions
        - Proper episode structure

    Args:
        zarr_path: Path to save zarr dataset
        num_episodes: Number of episodes
        episode_length: Length of each episode
        obs_dim: Observation dimension (default 13)
        action_dim: Action dimension (default 7)
        seed: Random seed
    """
    np.random.seed(seed)

    total_steps = num_episodes * episode_length
    obs_data = np.zeros((total_steps, obs_dim), dtype=np.float32)
    action_data = np.zeros((total_steps, action_dim), dtype=np.float32)
    episode_ends = np.arange(episode_length, total_steps + 1, episode_length)

    for ep in range(num_episodes):
        start_idx = ep * episode_length
        end_idx = start_idx + episode_length

        # Generate smooth random trajectory
        # Target position (fixed per episode)
        target = np.random.uniform(-1, 1, size=(1, 2))
        obs_data[start_idx:end_idx, 0:2] = target

        # EEF position: smooth random walk
        eef_pos = np.cumsum(np.random.randn(episode_length, 3) * 0.01, axis=0)
        eef_pos = eef_pos - eef_pos.mean(axis=0)  # Center
        obs_data[start_idx:end_idx, 2:5] = eef_pos

        # EEF quaternion: start from identity, small rotations
        eef_quat = np.zeros((episode_length, 4))
        eef_quat[:, 0] = 1.0  # w=1 for identity
        eef_quat += np.random.randn(episode_length, 4) * 0.1
        eef_quat = normalize_quaternion(eef_quat)
        obs_data[start_idx:end_idx, 5:9] = eef_quat

        # Rope shape: one-hot (random per episode)
        shape_idx = np.random.randint(0, 3)
        rope_shape = np.zeros((episode_length, 3))
        rope_shape[:, shape_idx] = 1.0
        obs_data[start_idx:end_idx, 9:12] = rope_shape

        # Gripper state: smooth [0, 1]
        gripper = np.abs(np.sin(np.linspace(0, 2*np.pi, episode_length)))
        obs_data[start_idx:end_idx, 12] = gripper

        # Actions: EEF state (matching observations, shifted by 1)
        action_data[start_idx:end_idx, 0:3] = eef_pos  # eef_pos
        action_data[start_idx:end_idx, 3:7] = eef_quat  # eef_quat

    # Save to zarr v3
    root = zarr.open(zarr_path, mode='w')
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    # Create arrays - zarr v3 infers shape from data
    data_group.create_array('obs', data=obs_data)
    data_group.create_array('action', data=action_data)
    meta_group.create_array('episode_ends', data=episode_ends)

    print(f"Created mock dataset at {zarr_path}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Episode length: {episode_length}")
    print(f"  Total steps: {total_steps}")
    print(f"  Obs shape: {obs_data.shape}")
    print(f"  Action shape: {action_data.shape}")


def get_obs_indices() -> dict:
    """Return observation index mapping for convenience."""
    return {
        'target_pos': (0, 2),
        'eef_pos': (2, 5),
        'eef_quat': (5, 9),
        'rope_shape': (9, 12),
        'gripper': (12, 13),
    }


def get_action_indices() -> dict:
    """Return action index mapping for convenience."""
    return {
        'eef_pos': (0, 3),
        'eef_quat': (3, 7),
    }
