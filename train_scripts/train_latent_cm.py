#!/usr/bin/env python3

import torch
import torchvision
import numpy as np
import argparse
import os
import json
import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).parent))

from src.latent_models import MetricEncoder, LatentDecoder, LatentFieldNetwork
from src.metric_losses import ClusterLoss, build_metric_loss_from_config
from src.latent_efm import LatentEFM
from src.models import ExponentialMovingAverage
from src.utils import Config, random_color


def load_config_from_json(config_path):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    def dict_to_config(d):
        if isinstance(d, dict):
            config = Config()
            for key, value in d.items():
                if isinstance(value, dict):
                    setattr(config, key, dict_to_config(value))
                elif isinstance(value, list):
                    setattr(config, key, tuple(value))
                else:
                    setattr(config, key, value)
            return config
        return d

    return dict_to_config(config_dict)


def save_checkpoint(state, checkpoint_dir, step, encoder, decoder, field_network):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pth")
    torch.save({
        'step': step,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'field_network_state_dict': field_network.state_dict(),
        'optimizer_state_dict': state['optimizer'].state_dict(),
        'ema_state_dict': state['ema'].state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path, encoder, decoder, field_network, optimizer, ema, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    field_network.load_state_dict(checkpoint['field_network_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ema.load_state_dict(checkpoint['ema_state_dict'])

    start_step = checkpoint.get('step', 0)
    print(f"Checkpoint loaded. Resuming from step {start_step}")

    return start_step


def main():
    parser = argparse.ArgumentParser(description='Latent EFM Training for CMNIST')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to JSON configuration file')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Directory to save model checkpoints')
    parser.add_argument('--data-dir', type=str, default='./data/MNIST/',
                       help='Directory for MNIST dataset')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume training from')
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}")
    config = load_config_from_json(args.config)

    if not hasattr(config.model, 'latent_dim'):
        config.model.latent_dim = 128
    if not hasattr(config.model, 'use_projection_head'):
        config.model.use_projection_head = True

    config.checkpoint_dir = args.checkpoint_dir

    if not hasattr(config, 'p') or not hasattr(config.p, 'x_loc'):
        if not hasattr(config, 'p'):
            config.p = Config()
        config.p.x_loc = 0.0

    if not hasattr(config, 'q') or not hasattr(config.q, 'x_loc'):
        if not hasattr(config, 'q'):
            config.q = Config()
        config.q.x_loc = config.L

    if not hasattr(config.training, 'lambda_metric'):
        config.training.lambda_metric = 0.1
    if not hasattr(config.training, 'lambda_recon'):
        config.training.lambda_recon = 0.1

    if hasattr(config, 'wandb') and hasattr(config.wandb, 'api_key'):
        os.environ['WANDB_API_KEY'] = config.wandb.api_key
        try:
            wandb.login(key=config.wandb.api_key)
        except Exception as e:
            print(f"Warning: WandB login failed: {e}")

    wandb_mode = getattr(config.wandb, 'mode', 'offline') if hasattr(config, 'wandb') else 'offline'
    try:
        wandb.init(
            project=getattr(config.wandb, 'project', 'LatentEFM_CMNIST') if hasattr(config, 'wandb') else 'LatentEFM_CMNIST',
            name=getattr(config.wandb, 'name_exp', 'latent_efm_cmnist') if hasattr(config, 'wandb') else 'latent_efm_cmnist',
            mode=wandb_mode
        )
    except Exception as e:
        print(f"Warning: WandB init failed: {e}")
        print("Switching to offline mode...")
        wandb.init(
            project='LatentEFM_CMNIST',
            name='latent_efm_cmnist',
            mode='offline'
        )

    print("Setting up data loaders...")
    TRANSFORM = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config.data.img_resize),
        torchvision.transforms.ToTensor(),
        random_color,
    ])

    train_data = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=TRANSFORM
    )
    eval_data = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=TRANSFORM
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4
    )

    print("Initializing models...")
    
    # Get architecture parameters from config
    encoder_hidden_dims = getattr(config.model, 'encoder_hidden_dims', [64, 128, 256, 512])
    decoder_hidden_dims = getattr(config.model, 'decoder_hidden_dims', [512, 256, 128, 64])
    # Skip connections removed - using simple encoder-decoder
    
    encoder = MetricEncoder(
        image_size=config.data.image_size,
        in_channels=config.data.num_channels,
        latent_dim=config.model.latent_dim,
        hidden_dims=encoder_hidden_dims,
        use_projection_head=config.model.use_projection_head
    ).to(config.device)

    decoder = LatentDecoder(
        latent_dim=config.model.latent_dim,
        out_channels=config.data.num_channels,
        image_size=config.data.image_size,
        hidden_dims=decoder_hidden_dims
    ).to(config.device)

    field_network = LatentFieldNetwork(
        latent_dim=config.model.latent_dim,
        time_embed_dim=256,
        hidden_dims=[512, 512, 512]
    ).to(config.device)

    metric_feat_dim = (config.model.latent_dim // 2
                       if config.model.use_projection_head
                       else config.model.latent_dim)
    num_classes = 10
    if hasattr(config, 'metric_loss') and getattr(config.metric_loss, 'components', None):
        metric_loss = build_metric_loss_from_config(
            config.metric_loss,
            num_classes=num_classes,
            feat_dim=metric_feat_dim,
            device=config.device,
        )
    else:
        metric_loss = ClusterLoss(
            num_classes=num_classes,
            feat_dim=metric_feat_dim,
            temperature=0.07,
            use_triplet=False,
            weight_contrastive=1.0,
            weight_center=0.1,
            device=config.device,
        )

    all_params = (list(encoder.parameters()) +
                  list(decoder.parameters()) +
                  list(field_network.parameters()) +
                  list(metric_loss.parameters()))

    optimizer = torch.optim.AdamW(
        all_params,
        lr=config.optim.lr,
        betas=(config.optim.beta1, 0.999),
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay
    )

    ema = ExponentialMovingAverage(field_network.parameters(), decay=config.model.ema_rate)
    start_step = 0

    if args.resume:
        start_step = load_checkpoint(
            args.resume, encoder, decoder, field_network,
            optimizer, ema, config.device
        )

    state = {
        'optimizer': optimizer,
        'encoder': encoder,
        'decoder': decoder,
        'field_network': field_network,
        'ema': ema,
        'step': start_step
    }

    print("Initializing Latent EFM...")
    latent_efm = LatentEFM(
        config=config,
        encoder=encoder,
        decoder=decoder,
        field_network=field_network,
        metric_loss=metric_loss
    )

    print("Starting training...")
    state = latent_efm.train(
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        state=state,
        use_recon_loss=True,
        recon_weight=config.training.lambda_recon,
        clip_grad=True,
        max_grad_norm=1.0
    )

    print("Saving final checkpoint...")
    save_checkpoint(state, args.checkpoint_dir, state['step'], encoder, decoder, field_network)

    print("Training completed!")
    wandb.finish()


if __name__ == '__main__':
    main()
