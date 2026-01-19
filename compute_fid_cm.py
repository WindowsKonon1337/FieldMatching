#!/usr/bin/env python3


import torch
import torchvision
import numpy as np
import argparse
import os
import json
import sys
import glob
from pathlib import Path
from tqdm import tqdm
from scipy import linalg

sys.path.insert(0, str(Path(__file__).parent))

from src.models import DDPM, ExponentialMovingAverage
from src.efm_field import EFM
from src.utils import Config, optimization_manager, random_color
from src.ode import get_rk45_sampler_pfgm
from src.inception import InceptionV3
from src.fid_score import calculate_frechet_distance


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


def generate_samples_batch(model, config, num_samples, batch_size, device):
    model.eval()
    all_samples = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Generating samples"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
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
            all_samples.append(sample.cpu())
    
    return torch.cat(all_samples, dim=0)


def get_activations_from_tensor(images, model, batch_size=50, dims=2048, device='cuda'):
    model.eval()
    
    from torch.nn.functional import interpolate
    
    if images.min() >= 0:
        images = images * 2.0 - 1.0
    
    if images.shape[2] != 299 or images.shape[3] != 299:
        images = interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    
    num_images = images.shape[0]
    pred_arr = np.empty((num_images, dims))
    
    for i in range(0, num_images, batch_size):
        end = min(i + batch_size, num_images)
        batch = images[i:end].to(device)
        
        with torch.no_grad():
            pred = model(batch)[0]
            
            if pred.size(2) != 1 or pred.size(3) != 1:
                from torch.nn.functional import adaptive_avg_pool2d
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            
            pred_arr[i:end] = pred.cpu().data.numpy().reshape(end - i, -1)
    
    return pred_arr


def calculate_activation_statistics_from_tensor(images, model, batch_size=50, dims=2048, device='cuda'):
    act = get_activations_from_tensor(images, model, batch_size, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_real_statistics(data_loader, model, batch_size=50, dims=2048, device='cuda', num_samples=10000):
    print(f"Computing real data statistics for {num_samples} samples...")
    
    all_images = []
    collected = 0
    
    for images, _ in data_loader:
        all_images.append(images)
        collected += images.shape[0]
        if collected >= num_samples:
            break
    
    images = torch.cat(all_images, dim=0)[:num_samples]
    if images.min() < 0:
        images = (images + 1.0) / 2.0
    images = torch.clamp(images, 0, 1)
    
    mu, sigma = calculate_activation_statistics_from_tensor(
        images, model, batch_size, dims, device
    )
    
    return mu, sigma


def compute_fid_for_checkpoint(checkpoint_path, config, real_mu, real_sigma, 
                               num_samples, gen_batch_size, fid_batch_size, device):
    print(f"\n{'='*60}")
    print(f"Processing checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"{'='*60}")
    
    net = DDPM(config).to(device)
    ema = ExponentialMovingAverage(net.parameters(), decay=config.model.ema_rate)
    
    try:
        step = load_checkpoint(checkpoint_path, net, ema, device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, step
    
    print(f"Generating {num_samples} samples...")
    try:
        generated_samples = generate_samples_batch(
            net, config, num_samples, gen_batch_size, device
        )
    except Exception as e:
        print(f"Error generating samples: {e}")
        return None, step
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()
    
    print("Computing generated samples statistics...")
    try:
        gen_mu, gen_sigma = calculate_activation_statistics_from_tensor(
            generated_samples, inception_model, fid_batch_size, dims=2048, device=device
        )
    except Exception as e:
        print(f"Error computing statistics: {e}")
        return None, step
    
    print("Computing FID...")
    try:
        fid_value = calculate_frechet_distance(real_mu, real_sigma, gen_mu, gen_sigma)
    except Exception as e:
        print(f"Error computing FID: {e}")
        return None, step
    
    print(f"FID: {fid_value:.4f}")
    
    return fid_value, step


def main():
    parser = argparse.ArgumentParser(description='Compute FID for all checkpoints on Colored MNIST')
    parser.add_argument('--checkpoints-dir', type=str, required=True,
                       help='Directory containing checkpoint files')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to JSON configuration file')
    parser.add_argument('--data-dir', type=str, default='./data/MNIST/',
                       help='Directory for MNIST dataset')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of samples to generate for FID (default: 10000)')
    parser.add_argument('--gen-batch-size', type=int, default=50,
                       help='Batch size for generation (default: 50)')
    parser.add_argument('--fid-batch-size', type=int, default=50,
                       help='Batch size for FID computation (default: 50)')
    parser.add_argument('--output', type=str, default='fid_results.json',
                       help='Output file for FID results (default: fid_results.json)')
    parser.add_argument('--checkpoint-pattern', type=str, default='checkpoint_step_*.pth',
                       help='Pattern to match checkpoint files (default: checkpoint_step_*.pth)')
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
    
    print("Setting up data loader for real statistics...")
    TRANSFORM = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config.data.img_resize),
        torchvision.transforms.ToTensor(),
        random_color,
        torchvision.transforms.Normalize([0.5], [0.5])
    ])
    
    real_data = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=TRANSFORM
    )
    
    real_loader = torch.utils.data.DataLoader(
        real_data,
        batch_size=args.fid_batch_size,
        shuffle=True
    )
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()
    
    real_mu, real_sigma = compute_real_statistics(
        real_loader, inception_model, 
        batch_size=args.fid_batch_size, 
        dims=2048, 
        device=device,
        num_samples=args.num_samples
    )
    print(f"Real data statistics computed: mu shape={real_mu.shape}, sigma shape={real_sigma.shape}")
    
    checkpoint_pattern = os.path.join(args.checkpoints_dir, args.checkpoint_pattern)
    checkpoint_files = sorted(glob.glob(checkpoint_pattern))
    
    if not checkpoint_files:
        print(f"No checkpoint files found matching pattern: {checkpoint_pattern}")
        return
    
    print(f"\nFound {len(checkpoint_files)} checkpoint files")
    
    results = []
    
    for checkpoint_path in checkpoint_files:
        fid_value, step = compute_fid_for_checkpoint(
            checkpoint_path, config, real_mu, real_sigma,
            args.num_samples, args.gen_batch_size, args.fid_batch_size, device
        )
        
        if fid_value is not None:
            results.append({
                'checkpoint': os.path.basename(checkpoint_path),
                'step': int(step),
                'fid': float(fid_value)
            })
    
    print(f"\n{'='*60}")
    print("Results summary:")
    print(f"{'='*60}")
    
    results.sort(key=lambda x: x['step'])
    
    for result in results:
        print(f"Step {result['step']:8d}: FID = {result['fid']:.4f} ({result['checkpoint']})")
    
    output_data = {
        'num_samples': args.num_samples,
        'gen_batch_size': args.gen_batch_size,
        'fid_batch_size': args.fid_batch_size,
        'results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
