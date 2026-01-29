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

sys.path.insert(0, str(Path(__file__).parent))

from src.latent_models import MetricEncoder, LatentDecoder, LatentFieldNetwork
from src.ode import LatentODESolver
from src.utils import Config, random_color
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


def load_models(checkpoint_path, config, device):
    if not hasattr(config.model, 'use_projection_head'):
        config.model.use_projection_head = False

    encoder = MetricEncoder(
        image_size=config.data.image_size,
        in_channels=config.data.num_channels,
        latent_dim=config.model.latent_dim,
        use_projection_head=config.model.use_projection_head
    ).to(device)

    decoder = LatentDecoder(
        latent_dim=config.model.latent_dim,
        out_channels=config.data.num_channels,
        image_size=config.data.image_size
    ).to(device)

    field_network = LatentFieldNetwork(
        latent_dim=config.model.latent_dim,
        time_embed_dim=256,
        hidden_dims=[512, 512, 512]
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    field_network.load_state_dict(checkpoint['field_network_state_dict'])

    encoder.eval()
    decoder.eval()
    field_network.eval()
    return encoder, decoder, field_network


def get_activations_from_tensor(images, model, batch_size=50, dims=2048, device='cuda'):
    from torch.nn.functional import interpolate, adaptive_avg_pool2d

    model.eval()
    if images.min() < 0:
        images = (images + 1.0) / 2.0
    images = torch.clamp(images, 0, 1)

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
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred_arr[i:end] = pred.cpu().data.numpy().reshape(end - i, -1)

    return pred_arr


def generate_and_compute_stats_batch(
    decoder,
    field_network,
    config,
    num_samples,
    gen_batch_size,
    inception_model,
    fid_batch_size,
    device,
    num_steps=100,
    method='rk4',
):
    decoder.eval()
    field_network.eval()
    inception_model.eval()

    ode_solver = LatentODESolver(field_network, config)
    num_batches = (num_samples + gen_batch_size - 1) // gen_batch_size
    all_activations = []

    with torch.no_grad():
        for i in tqdm(
        range(num_batches),
        desc="Generate & FID stats",
        unit="batch",
        dynamic_ncols=True,
    ):
            current_batch_size = min(gen_batch_size, num_samples - i * gen_batch_size)

            z_noise = torch.randn(current_batch_size, config.model.latent_dim).to(device)
            z_samples, _ = ode_solver.sample(z_noise, num_steps=num_steps, method=method)
            sample = decoder(z_samples)

            if sample.min() < 0:
                sample = (sample + 1.0) / 2.0
            sample = torch.clamp(sample, 0, 1)

            batch_activations = get_activations_from_tensor(
                sample, inception_model, fid_batch_size, dims=2048, device=device
            )
            all_activations.append(batch_activations)

            del sample, z_noise, z_samples
            if device == 'cuda':
                torch.cuda.empty_cache()

    all_activations = np.vstack(all_activations)
    mu = np.mean(all_activations, axis=0)
    sigma = np.cov(all_activations, rowvar=False)
    return mu, sigma


def compute_real_statistics(
    data_loader, model, batch_size=50, dims=2048, device='cuda', num_samples=10000
):
    all_activations = []
    collected = 0
    pbar = tqdm(total=num_samples, desc="Real data stats", unit="samples", dynamic_ncols=True)

    for images, _ in data_loader:
        remaining = num_samples - collected
        if remaining <= 0:
            break
        if images.shape[0] > remaining:
            images = images[:remaining]

        if images.min() < 0:
            images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0, 1)

        batch_activations = get_activations_from_tensor(
            images, model, batch_size, dims, device
        )
        all_activations.append(batch_activations)
        n = images.shape[0]
        collected += n
        pbar.update(n)
        if collected >= num_samples:
            break

    pbar.close()

    all_activations = np.vstack(all_activations)[:num_samples]
    mu = np.mean(all_activations, axis=0)
    sigma = np.cov(all_activations, rowvar=False)
    return mu, sigma


def parse_step_from_checkpoint_path(path):
    basename = os.path.basename(path)
    if basename.startswith("checkpoint_step_") and basename.endswith(".pth"):
        try:
            return int(basename.replace("checkpoint_step_", "").replace(".pth", ""))
        except ValueError:
            pass
    return None


def compute_fid_for_checkpoint(
    checkpoint_path,
    config,
    real_mu,
    real_sigma,
    num_samples,
    gen_batch_size,
    fid_batch_size,
    device,
    inception_model,
    num_steps=100,
    method='rk4',
):
    step = parse_step_from_checkpoint_path(checkpoint_path)

    if device == 'cuda':
        torch.cuda.empty_cache()

    try:
        encoder, decoder, field_network = load_models(checkpoint_path, config, device)
    except Exception as e:
        print(f"Ошибка загрузки чекпоинта: {e}")
        return None, step

    try:
        gen_mu, gen_sigma = generate_and_compute_stats_batch(
            decoder,
            field_network,
            config,
            num_samples,
            gen_batch_size,
            inception_model,
            fid_batch_size,
            device,
            num_steps=num_steps,
            method=method,
        )
    except Exception as e:
        print(f"Ошибка генерации: {e}")
        import traceback
        traceback.print_exc()
        del encoder, decoder, field_network
        if device == 'cuda':
            torch.cuda.empty_cache()
        return None, step

    del encoder, decoder, field_network
    if device == 'cuda':
        torch.cuda.empty_cache()

    try:
        fid_value = calculate_frechet_distance(real_mu, real_sigma, gen_mu, gen_sigma)
    except Exception as e:
        print(f"Ошибка расчёта FID: {e}")
        return None, step

    return fid_value, step


def main():
    parser = argparse.ArgumentParser(
        description='FID для latent_cm по чекпоинтам (батчевая генерация и FID)'
    )
    parser.add_argument(
        '--checkpoints-dir',
        type=str,
        required=True,
        help='Папка с чекпоинтами (.pth)',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config_latent_cmnist.json',
        help='Путь к JSON конфигу (по умолчанию: config_latent_cmnist.json)',
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/MNIST/',
        help='Папка датасета MNIST',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10000,
        help='Число семплов для FID (по умолчанию: 10000)',
    )
    parser.add_argument(
        '--gen-batch-size',
        type=int,
        default=64,
        help='Размер батча генерации (по умолчанию: 64)',
    )
    parser.add_argument(
        '--fid-batch-size',
        type=int,
        default=50,
        help='Размер батча для Inception при подсчёте FID (по умолчанию: 50)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='fid_latent_cm_results.json',
        help='Файл для сохранения результатов JSON (по умолчанию: fid_latent_cm_results.json)',
    )
    parser.add_argument(
        '--checkpoint-pattern',
        type=str,
        default='checkpoint_step_*.pth',
        help='Маска имён чекпоинтов (по умолчанию: checkpoint_step_*.pth)',
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=100,
        help='Число шагов ODE при генерации (по умолчанию: 100)',
    )
    parser.add_argument(
        '--method',
        type=str,
        default='rk4',
        choices=['euler', 'rk4'],
        help='Метод интегрирования ODE (по умолчанию: rk4)',
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(Path(__file__).parent, config_path)
    print(f"Загрузка конфига: {config_path}")
    config = load_config_from_json(config_path)

    if not hasattr(config.model, 'latent_dim'):
        config.model.latent_dim = 128

    device = getattr(config, 'device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA недоступна, используется CPU")
        device = 'cpu'
        config.device = 'cpu'
    else:
        config.device = device
        if device == 'cuda':
            print(f"Устройство: {torch.cuda.get_device_name(0)}")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config.data.img_resize),
        torchvision.transforms.ToTensor(),
        random_color,
        torchvision.transforms.Normalize([0.5], [0.5]),
    ])
    real_data = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    real_loader = torch.utils.data.DataLoader(
        real_data,
        batch_size=args.fid_batch_size,
        shuffle=True,
    )

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_real = InceptionV3([block_idx]).to(device)
    inception_real.eval()

    print(f"Подсчёт статистик реальных данных по {args.num_samples} семплам...")
    real_mu, real_sigma = compute_real_statistics(
        real_loader,
        inception_real,
        batch_size=args.fid_batch_size,
        dims=2048,
        device=device,
        num_samples=args.num_samples,
    )
    print(f"Реальные статистики: mu {real_mu.shape}, sigma {real_sigma.shape}")

    del inception_real
    if device == 'cuda':
        torch.cuda.empty_cache()

    pattern = os.path.join(args.checkpoints_dir, args.checkpoint_pattern)
    checkpoint_files = sorted(glob.glob(pattern))

    def sort_key(p):
        s = parse_step_from_checkpoint_path(p)
        return (s is None, s or 0)

    checkpoint_files = sorted(checkpoint_files, key=sort_key)

    if not checkpoint_files:
        print(f"Чекпоинты не найдены: {pattern}")
        return

    print(f"\nНайдено чекпоинтов: {len(checkpoint_files)}\n")

    results = []
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()

    for checkpoint_path in tqdm(
        checkpoint_files,
        desc="Checkpoints",
        unit="ckpt",
        dynamic_ncols=True,
    ):
        ckpt_name = os.path.basename(checkpoint_path)
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Чекпоинт: {ckpt_name}")
        tqdm.write(f"{'='*60}")

        fid_value, step = compute_fid_for_checkpoint(
            checkpoint_path,
            config,
            real_mu,
            real_sigma,
            args.num_samples,
            args.gen_batch_size,
            args.fid_batch_size,
            device,
            inception_model,
            num_steps=args.num_steps,
            method=args.method,
        )

        if fid_value is not None:
            results.append({
                'checkpoint': ckpt_name,
                'step': int(step) if step is not None else -1,
                'fid': float(fid_value),
            })
            tqdm.write(f"FID = {fid_value:.4f}  (checkpoint: {ckpt_name})")

    print(f"\n{'='*60}")
    print("Итог по чекпоинтам:")
    print(f"{'='*60}")
    results_sorted = sorted(results, key=lambda x: x['step'])
    for r in results_sorted:
        print(f"  step {r['step']:8d}  FID = {r['fid']:.4f}  ({r['checkpoint']})")

    output_data = {
        'checkpoints_dir': args.checkpoints_dir,
        'config': args.config,
        'num_samples': args.num_samples,
        'gen_batch_size': args.gen_batch_size,
        'fid_batch_size': args.fid_batch_size,
        'num_steps': args.num_steps,
        'method': args.method,
        'results': results_sorted,
    }

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(Path(__file__).parent, out_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nРезультаты сохранены: {out_path}")


if __name__ == '__main__':
    main()
