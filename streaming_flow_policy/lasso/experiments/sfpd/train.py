"""
Training script for Streaming Flow Policy (Deterministic) on lasso task.
Follows pusht/experiments/sfpd_obs/train.py patterns.

Usage:
    python -m streaming_flow_policy.lasso.experiments.sfpd.train

Or with custom paths:
    python -m streaming_flow_policy.lasso.experiments.sfpd.train --zarr_path /path/to/data.zarr
"""
import argparse
from typing import Dict
import numpy as np
import torch
from torch import Tensor
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from streaming_flow_policy.lasso.dataset import LassoStateDataset
from streaming_flow_policy.pusht.dp_state_notebook.network import ConditionalUnet1D
from streaming_flow_policy.lasso.sfpd import StreamingFlowPolicyDeterministicLasso


def parse_args():
    parser = argparse.ArgumentParser(description='Train SFPD on lasso task')
    parser.add_argument('--zarr_path', type=str, required=True,
                        help='Path to zarr dataset')
    parser.add_argument('--save_path', type=str, default='models/lasso_sfpd.pth',
                        help='Path to save trained model')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='Noise sigma for training')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    return parser.parse_args()


def main():
    args = parse_args()

    # =============================================================================
    # Parameters
    # =============================================================================
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    obs_dim = 13  # target(2) + eef_pos(3) + eef_quat(4) + rope_shape(3) + gripper(1)
    action_dim = 7  # eef_pos(3) + eef_quat(4)

    # =============================================================================
    # Model Setup
    # =============================================================================

    # Create velocity network
    velocity_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        fc_timesteps=1,  # Single path for deterministic
    )

    # Device transfer
    device = torch.device(args.device)
    velocity_net = velocity_net.to(device)

    # Create policy
    policy = StreamingFlowPolicyDeterministicLasso(
        velocity_net=velocity_net,
        action_dim=action_dim,
        pred_horizon=pred_horizon,
        sigma=args.sigma,
        device=device,
    )

    # =============================================================================
    # Dataset Setup
    # =============================================================================

    dataset = LassoStateDataset(
        zarr_path=args.zarr_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        transform_datum_fn=policy.TransformTrainingDatum,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    # =============================================================================
    # Training Setup
    # =============================================================================

    # Exponential Moving Average for stable inference
    ema = EMAModel(parameters=velocity_net.parameters(), power=0.75)

    # Optimizer
    optimizer = torch.optim.AdamW(
        params=velocity_net.parameters(),
        lr=args.lr,
        weight_decay=1e-6,
    )

    # Cosine LR schedule with warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * args.num_epochs,
    )

    # =============================================================================
    # Training Loop
    # =============================================================================

    print(f"Training SFPD on lasso task")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Device: {device}")

    with tqdm(range(args.num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []

            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # Compute loss
                    loss = policy.Loss(nbatch)

                    # Optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                    # Update EMA
                    ema.step(velocity_net.parameters())

                    # Logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            tglobal.set_postfix(loss=np.mean(epoch_loss))

    # =============================================================================
    # Save Model
    # =============================================================================

    # Copy EMA weights to model for inference
    ema.copy_to(velocity_net.parameters())

    # Save
    torch.save(policy.state_dict(), args.save_path)
    print(f"Saved model to {args.save_path}")


if __name__ == '__main__':
    main()
