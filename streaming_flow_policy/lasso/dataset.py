"""
Dataset for lasso robotic arm task.
Follows the same patterns as pusht/dataset.py and pusht/dp_state_notebook/dataset.py.
"""
from typing import Dict, Optional, Callable
import numpy as np
import torch
import zarr


def create_sample_indices(
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0
    ) -> np.ndarray:
    """
    Create indices for sampling sequences from the dataset.

    Args:
        episode_ends: Marks one-past the last index for each episode.
        sequence_length: Length of the sequences to sample.
        pad_before: Number of timesteps to pad before the sequence.
        pad_after: Number of timesteps to pad after the sequence.

    Returns:
        np.ndarray of shape (N, 4) with columns:
        (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
    """
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(
        train_data: Dict[str, np.ndarray],
        sequence_length: int,
        buffer_start_idx: int,
        buffer_end_idx: int,
        sample_start_idx: int,
        sample_end_idx: int
    ) -> Dict[str, np.ndarray]:
    """
    Sample a sequence from the dataset with proper padding.

    Args:
        train_data: Dict with 'obs' and 'action' arrays of shape (N, K).
        sequence_length: Length of sequence to sample.
        buffer_start_idx, buffer_end_idx: Indices into the buffer.
        sample_start_idx, sample_end_idx: Indices into the output sample.

    Returns:
        Dict with 'obs' and 'action' arrays of shape (sequence_length, K).
    """
    result: Dict[str, np.ndarray] = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)

            # Repeat the first timestep for padding before
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]

            # Repeat the last timestep for padding after
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]

            # Fill in the middle
            data[sample_start_idx:sample_end_idx] = sample

        result[key] = data
    return result


def get_data_stats(data: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute min/max statistics for normalization."""
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats


def normalize_data(data: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Normalize data to [-1, 1] range."""
    # normalize to [0, 1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Unnormalize data from [-1, 1] back to original range."""
    # unnormalize to [0, 1]
    ndata = (ndata + 1) / 2
    # unnormalize to original range
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


class LassoStateDataset(torch.utils.data.Dataset):
    """
    Dataset for lasso robotic arm task.

    Observation (obs_dim=13):
        - target_position (x, y): indices 0:2
        - eef_position (x, y, z): indices 2:5
        - eef_pose (quaternion): indices 5:9
        - rope_shape (one-hot): indices 9:12
        - gripper_state: indices 12:13

    Action (action_dim=7):
        - eef_position (x, y, z): indices 0:3
        - eef_pose (quaternion): indices 3:7

    Sequence structure:
    |o|o|                             observations: 2
    | |a|a|a|a|a|a|a|a|               actions executed: 8
    |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
    """
    def __init__(
            self,
            zarr_path: str,
            pred_horizon: int = 16,
            obs_horizon: int = 2,
            action_horizon: int = 8,
            transform_datum_fn: Optional[Callable] = None,
        ):
        """
        Args:
            zarr_path: Path to zarr dataset file.
            pred_horizon: Number of future actions to predict.
            obs_horizon: Number of past observations to use.
            action_horizon: Number of actions to execute before replanning.
            transform_datum_fn: Optional function to transform each datum.
        """
        self.zarr_path = zarr_path
        self.transform_datum_fn = transform_datum_fn
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # Load dataset
        dataset_root = self._get_dataset_root()
        train_data = self._load_train_data(dataset_root)
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # Create sample indices with padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1
        )

        # Compute statistics and normalize
        stats: Dict[str, Dict[str, np.ndarray]] = dict()
        normalized_train_data: Dict[str, np.ndarray] = dict()

        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data

    def _get_dataset_root(self) -> zarr.Group:
        """Load dataset from local zarr file."""
        return zarr.open(self.zarr_path, mode='r')

    def _load_train_data(self, dataset_root: zarr.Group) -> Dict[str, np.ndarray]:
        """
        Load observations and actions from zarr.

        Expected zarr structure:
            data/obs: (N, 13) - observations
            data/action: (N, 7) - actions (absolute EEF positions)
            meta/episode_ends: (E,) - episode end indices

        Returns:
            Dict with 'obs' (N, 13) and 'action' (N, 7) arrays.
        """
        obs = dataset_root['data']['obs'][:]  # (N, 13)
        action = dataset_root['data']['action'][:]  # (N, 7)
        return {'obs': obs, 'action': action}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        # Get indices for this sample
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # Sample normalized sequence
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # Keep only obs_horizon observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon, :]

        # Apply transform if provided
        if self.transform_datum_fn is not None:
            nsample = self.transform_datum_fn(nsample)

        return nsample
