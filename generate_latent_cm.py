#!/usr/bin/env python3

import torch
import torchvision
import numpy as np
import argparse
import os
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from src.latent_models import MetricEncoder, LatentDecoder, LatentFieldNetwork
from src.ode import LatentODESolver
from src.utils import Config


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


def load_models(checkpoint_path, config):
    print(f"Loading models from {checkpoint_path}")

    if not hasattr(config.model, 'use_projection_head'):
        config.model.use_projection_head = False

    encoder = MetricEncoder(
        image_size=config.data.image_size,
        in_channels=config.data.num_channels,
        latent_dim=config.model.latent_dim,
        use_projection_head=config.model.use_projection_head
    ).to(config.device)

    decoder = LatentDecoder(
        latent_dim=config.model.latent_dim,
        out_channels=config.data.num_channels,
        image_size=config.data.image_size
    ).to(config.device)

    field_network = LatentFieldNetwork(
        latent_dim=config.model.latent_dim,
        time_embed_dim=256,
        hidden_dims=[512, 512, 512]
    ).to(config.device)

    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    field_network.load_state_dict(checkpoint['field_network_state_dict'])

    encoder.eval()
    decoder.eval()
    field_network.eval()

    print("Models loaded successfully")
    return encoder, decoder, field_network


def generate_unconditional(field_network, decoder, config, num_samples=16,
                           num_steps=100, method='rk4'):
    print(f"Generating {num_samples} unconditional samples...")

    ode_solver = LatentODESolver(field_network, config)
    z_noise = torch.randn(num_samples, config.model.latent_dim).to(config.device)

    with torch.no_grad():
        z_samples, trajectory = ode_solver.sample(z_noise, num_steps=num_steps, method=method)
        images = decoder(z_samples)

    print(f"Generated {num_samples} samples")
    return images.cpu(), trajectory


def generate_conditional(encoder, field_network, decoder, anchor_image, config,
                         num_samples=16, num_steps=100, method='rk4',
                         interpolation_weight=0.9):
    print(f"Generating {num_samples} samples conditioned on anchor (PixelLatent)...")

    device = config.device

    with torch.no_grad():
        anchor_image = anchor_image.to(device)
        z_anchor = encoder(anchor_image)

        noise_pixel = torch.randn(
            num_samples,
            config.data.num_channels,
            config.data.image_size,
            config.data.image_size,
            device=device,
        )
        z_noise = encoder(noise_pixel)

        z_init = interpolation_weight * z_anchor.repeat(num_samples, 1) + \
                 (1.0 - interpolation_weight) * z_noise

        ode_solver = LatentODESolver(field_network, config)
        z_samples, trajectory = ode_solver.sample(z_init, num_steps=num_steps, method=method)
        images = decoder(z_samples)

    print(f"Generated {num_samples} conditional samples")
    return images.cpu(), trajectory


def cluster_conditional_generation(encoder, field_network, decoder, dataset,
                                   class_id, config, num_samples=16,
                                   num_steps=100, method='rk4'):
    print(f"Generating {num_samples} samples from class {class_id}...")

    class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_id]

    if len(class_indices) == 0:
        raise ValueError(f"No samples found for class {class_id}")

    anchor_idx = np.random.choice(class_indices)
    anchor_image, _ = dataset[anchor_idx]
    anchor_image = anchor_image.unsqueeze(0)

    with torch.no_grad():
        z_anchor = encoder(anchor_image.to(config.device))

    num_centroid_samples = min(100, len(class_indices))
    centroid_indices = np.random.choice(class_indices, num_centroid_samples, replace=False)

    z_class_samples = []
    with torch.no_grad():
        for idx in centroid_indices:
            img, _ = dataset[idx]
            z = encoder(img.unsqueeze(0).to(config.device))
            z_class_samples.append(z)

    z_centroid = torch.mean(torch.cat(z_class_samples, dim=0), dim=0, keepdim=True)
    ode_solver = LatentODESolver(field_network, config)
    z_noise = torch.randn(num_samples, config.model.latent_dim).to(config.device)
    z_init = z_centroid.repeat(num_samples, 1) + 0.3 * z_noise

    with torch.no_grad():
        z_samples, trajectory = ode_solver.sample(z_init, num_steps=num_steps, method=method)
        images = decoder(z_samples)

    print(f"Generated {num_samples} samples from class {class_id}")
    return images.cpu(), anchor_image


def save_images(images, output_path, nrow=4):
    grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=True, scale_each=True)
    torchvision.utils.save_image(grid, output_path)
    print(f"Images saved to {output_path}")


def visualize_images(images, title="Generated Images", nrow=4):
    grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=True, scale_each=True)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate images with Latent EFM on CMNIST')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to JSON configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./generated_cmnist/',
                       help='Directory to save generated images')
    parser.add_argument('--mode', type=str, default='unconditional',
                       choices=['unconditional', 'conditional', 'cluster'],
                       help='Generation mode')
    parser.add_argument('--num-samples', type=int, default=16,
                       help='Number of samples to generate')
    parser.add_argument('--num-steps', type=int, default=100,
                       help='Number of ODE integration steps')
    parser.add_argument('--method', type=str, default='rk4',
                       choices=['euler', 'rk4'],
                       help='ODE integration method')
    parser.add_argument('--class-id', type=int, default=0,
                       help='Digit class ID for cluster-conditional generation (0-9)')
    parser.add_argument('--anchor-image', type=str, default=None,
                       help='Path to anchor image for conditional generation')
    parser.add_argument('--data-dir', type=str, default='./data/MNIST/',
                       help='Directory for MNIST dataset (for cluster mode)')
    parser.add_argument('--anchor-copies', type=int, default=16,
                       help='Number of anchor copies to save in a grid for visualization')
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}")
    config = load_config_from_json(args.config)

    if not hasattr(config.model, 'latent_dim'):
        config.model.latent_dim = 128

    os.makedirs(args.output_dir, exist_ok=True)
    encoder, decoder, field_network = load_models(args.checkpoint, config)

    if args.mode == 'unconditional':
        images, trajectory = generate_unconditional(
            field_network, decoder, config,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            method=args.method
        )
        output_path = os.path.join(args.output_dir, 'unconditional_samples.png')
        save_images(images, output_path)

    elif args.mode == 'conditional':
        if args.anchor_image is None:
            from src.utils import random_color
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(config.data.img_resize),
                torchvision.transforms.ToTensor(),
                random_color,
            ])
            dataset = torchvision.datasets.MNIST(
                root=args.data_dir,
                train=False,
                download=True,
                transform=transform
            )
            idx = np.random.randint(0, len(dataset))
            anchor_tensor, anchor_label = dataset[idx]
            anchor_tensor = anchor_tensor.unsqueeze(0)
            print(f"Using random CMNIST sample as anchor: index={idx}, label={anchor_label}")
        else:
            from PIL import Image
            anchor_img = Image.open(args.anchor_image).convert('RGB')
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((config.data.image_size, config.data.image_size)),
                torchvision.transforms.ToTensor()
            ])
            anchor_tensor = transform(anchor_img).unsqueeze(0)

        images, trajectory = generate_conditional(
            encoder, field_network, decoder, anchor_tensor, config,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            method=args.method
        )
        output_path = os.path.join(args.output_dir, 'conditional_samples.png')
        save_images(images, output_path)
        anchor_path = os.path.join(args.output_dir, 'anchor.png')
        torchvision.utils.save_image(anchor_tensor, anchor_path)
        anchor_copies = anchor_tensor.repeat(args.anchor_copies, 1, 1, 1)
        anchor_copies_path = os.path.join(args.output_dir, 'anchor_copies.png')
        save_images(anchor_copies, anchor_copies_path)

    elif args.mode == 'cluster':
        from src.utils import random_color
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(config.data.img_resize),
            torchvision.transforms.ToTensor(),
            random_color,
        ])
        dataset = torchvision.datasets.MNIST(
            root=args.data_dir,
            train=False,
            download=True,
            transform=transform
        )

        images, anchor = cluster_conditional_generation(
            encoder, field_network, decoder, dataset,
            class_id=args.class_id,
            config=config,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            method=args.method
        )
        output_path = os.path.join(args.output_dir, f'cluster_{args.class_id}_samples.png')
        save_images(images, output_path)
        anchor_path = os.path.join(args.output_dir, f'cluster_{args.class_id}_anchor.png')
        torchvision.utils.save_image(anchor, anchor_path)

    print("Generation completed!")


if __name__ == '__main__':
    main()
