#!/usr/bin/env python3

import torch
import torchvision
import numpy as np
import argparse
import os
import json
import sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from src.models import DDPM, ExponentialMovingAverage
from src.efm_field import EFM
from src.utils import Config, random_color
from src.ode import get_rk45_sampler_pfgm, LearnedImageODESolver


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


def load_checkpoint(checkpoint_path, model, ema, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    ema.load_state_dict(checkpoint['ema_state_dict'])
    ema.copy_to(model.parameters())
    
    step = checkpoint.get('step', 0)
    print(f"Checkpoint loaded from step {step}")
    
    return step


def save_image(image_tensor, filepath):
    if image_tensor.min() < 0:
        image_tensor = (image_tensor + 1.0) / 2.0
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]
    
    if image_tensor.shape[0] == 3:
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        image_np = image_tensor.cpu().numpy()
    
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    
    Image.fromarray(image_np).save(filepath)


def generate_rk45(model, config, num_samples, batch_size, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    sample_idx = 0
    
    print(f"Generating {num_samples} samples using RK45 method...")
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="RK45 generation"):
            current_batch_size = min(batch_size, num_samples - sample_idx)
            
            shape = (current_batch_size, config.data.num_channels,
                     config.data.image_size, config.data.image_size)
            
            batch_y = torch.randn(*shape).to(device)
            
            sampling_fn = get_rk45_sampler_pfgm(
                y=batch_y,
                config=config,
                shape=shape,
                eps=config.training.epsilon,
                device=device
            )
            
            sample, n, traj = sampling_fn(model, batch_y)
            
            if sample.min() < 0:
                sample = (sample + 1.0) / 2.0
            sample = torch.clamp(sample, 0, 1)
            
            for j in range(current_batch_size):
                img_path = os.path.join(output_dir, f"sample_{sample_idx:06d}.png")
                save_image(sample[j:j+1], img_path)
                sample_idx += 1
    
    print(f"RK45 samples saved to {output_dir}")


def generate_euler(model, config, num_samples, batch_size, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    sample_idx = 0
    
    print(f"Generating {num_samples} samples using Euler method...")
    
    ode_solver = LearnedImageODESolver(model, config)
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Euler generation"):
            current_batch_size = min(batch_size, num_samples - sample_idx)
            
            shape = (current_batch_size, config.data.num_channels,
                     config.data.image_size, config.data.image_size)
            
            batch_y = torch.randn(*shape).to(device)
            
            x_init = torch.cat([
                (config.L) * torch.ones(batch_y.shape[0], device=batch_y.device)[:, None],
                batch_y.view(-1, config.DIM - 1)
            ], dim=1).to(device)
            
            sample, traj = ode_solver(x_init)
            
            final_images = sample[:, 1:].view(
                current_batch_size, config.data.num_channels,
                config.data.image_size, config.data.image_size
            )
            
            if final_images.min() < 0:
                final_images = (final_images + 1.0) / 2.0
            final_images = torch.clamp(final_images, 0, 1)
            
            for j in range(current_batch_size):
                img_path = os.path.join(output_dir, f"sample_{sample_idx:06d}.png")
                save_image(final_images[j:j+1], img_path)
                sample_idx += 1
    
    print(f"Euler samples saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate images from checkpoint using RK45 and Euler methods')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to JSON configuration file')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to generate (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for generation (default: 32)')
    parser.add_argument('--output-dir', type=str, default='./generated',
                       help='Output directory for generated images (default: ./generated)')
    parser.add_argument('--method', type=str, choices=['both', 'rk45', 'euler'], default='both',
                       help='Generation method: both, rk45, or euler (default: both)')
    args = parser.parse_args()
    
    print(f"Loading configuration from {args.config}")
    config = load_config_from_json(args.config)
    
    if not hasattr(config, 'DIM') or config.DIM is None:
        config.DIM = config.data.num_channels * config.data.image_size * config.data.image_size + 1
    
    if not hasattr(config, 'q') or not hasattr(config.q, 'x_loc') or config.q.x_loc is None:
        if not hasattr(config, 'q'):
            config.q = Config()
        config.q.x_loc = config.L
    
    device = config.device
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = 'cpu'
        else:
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
    
    print("Initializing model...")
    net = DDPM(config).to(device)
    ema = ExponentialMovingAverage(net.parameters(), decay=config.model.ema_rate)
    
    load_checkpoint(args.checkpoint, net, ema, device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.method in ['both', 'rk45']:
        rk45_dir = os.path.join(args.output_dir, 'RK45')
        generate_rk45(net, config, args.num_samples, args.batch_size, device, rk45_dir)
    
    if args.method in ['both', 'euler']:
        euler_dir = os.path.join(args.output_dir, 'Euler')
        generate_euler(net, config, args.num_samples, args.batch_size, device, euler_dir)
    
    print(f"\nGeneration completed!")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
