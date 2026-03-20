#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import wandb
from tqdm import tqdm

# Надежно добавляем корень репозитория в sys.path,
# чтобы корректно находился модуль `src.*` независимо от текущей директории запуска.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.efm_field import EFM
from src.models import DDPM, ExponentialMovingAverage
from src.utils import Config, optimization_manager, random_color


def load_config_from_json(config_path: str) -> Config:
    with open(config_path, "r") as f:
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


def save_checkpoint(state, checkpoint_dir, step):
    """Сохраняет чекпоинт модели"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pth")
    torch.save(
        {
            "step": step,
            "model_state_dict": state["model"].state_dict(),
            "optimizer_state_dict": state["optimizer"].state_dict(),
            "ema_state_dict": state["ema"].state_dict(),
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, ema, device):
    """Загружает чекпоинт модели"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    ema.load_state_dict(checkpoint["ema_state_dict"])

    start_step = checkpoint.get("step", 0)
    print(f"Checkpoint loaded. Resuming from step {start_step}")
    return start_step


def parse_digits(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_mnist_mixed_loader(
    *,
    mnist_dataset,
    batch_size: int,
    inlier_digits,
    outlier_digits,
    outlier_prob: float,
    seed: int = 0,
):
    """
    Строит DataLoader, который с вероятностью `outlier_prob` выбирает семплы из `outlier_digits`,
    и с вероятностью (1-outlier_prob) выбирает семплы из `inlier_digits`.
    Реализация через WeightedRandomSampler по индексам внутри отфильтрованного Subset.
    """

    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

    targets = mnist_dataset.targets
    if torch.is_tensor(targets):
        targets = targets.cpu()

    inlier_mask = torch.zeros_like(targets, dtype=torch.bool)
    for d in inlier_digits:
        inlier_mask |= targets == d

    outlier_mask = torch.zeros_like(targets, dtype=torch.bool)
    for d in outlier_digits:
        outlier_mask |= targets == d

    inlier_indices = torch.nonzero(inlier_mask, as_tuple=False).flatten().tolist()
    outlier_indices = torch.nonzero(outlier_mask, as_tuple=False).flatten().tolist()

    if len(inlier_indices) == 0:
        raise ValueError(f"No inlier samples found for digits {inlier_digits}")
    if len(outlier_indices) == 0:
        raise ValueError(f"No outlier samples found for digits {outlier_digits}")

    subset_indices = inlier_indices + outlier_indices
    subset = Subset(mnist_dataset, subset_indices)

    inlier_prob = 1.0 - outlier_prob
    inlier_weight = inlier_prob / len(inlier_indices)
    outlier_weight = outlier_prob / len(outlier_indices)

    weights = np.array([inlier_weight] * len(inlier_indices) + [outlier_weight] * len(outlier_indices))
    weights = torch.as_tensor(weights, dtype=torch.double)

    # Чтобы не получать неполный последний батч (в коде дальше используется small_batch_size)
    effective_num_samples = (len(subset) // batch_size) * batch_size
    if effective_num_samples < batch_size:
        effective_num_samples = len(subset)

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=effective_num_samples,
        replacement=True,
        generator=generator,
    )

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )
    return loader


def build_filtered_mnist_loader(*, mnist_dataset, batch_size: int, digits, shuffle: bool, drop_last: bool):
    from torch.utils.data import DataLoader, Subset

    targets = mnist_dataset.targets
    if torch.is_tensor(targets):
        targets = targets.cpu()

    mask = torch.zeros_like(targets, dtype=torch.bool)
    for d in digits:
        mask |= targets == d

    indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    if len(indices) == 0:
        return None

    subset = Subset(mnist_dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=False,
    )
    return loader


def main():
    parser = argparse.ArgumentParser(description="EFM Training for Colored MNIST (outlier sevens).")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--data-dir", type=str, default="./data/MNIST/", help="Directory for MNIST dataset")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from")
    parser.add_argument("--wandb-dir", type=str, default=None, help="Directory for local (offline) wandb logs")

    parser.add_argument("--inlier-digits", type=str, default="2,3", help="Comma-separated inlier digits")
    parser.add_argument("--outlier-digits", type=str, default="7", help="Comma-separated outlier digits")
    parser.add_argument("--outlier-prob", type=float, default=0.02, help="Fraction of outliers in training sampling")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the weighted sampler")
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}")
    config = load_config_from_json(args.config)

    if not hasattr(config, "DIM") or config.DIM is None:
        config.DIM = config.data.num_channels * config.data.image_size * config.data.image_size + 1

    if not hasattr(config, "q") or not hasattr(config.q, "x_loc") or config.q.x_loc is None:
        if not hasattr(config, "q"):
            config.q = Config()
        config.q.x_loc = config.L
    elif config.q.x_loc != config.L:
        config.q.x_loc = config.L

    config.checkpoint_dir = args.checkpoint_dir

    if hasattr(config, "sampling"):
        if hasattr(config.training, "epsilon"):
            config.sampling.z_min = config.training.epsilon
        if hasattr(config, "L"):
            config.sampling.z_max = config.L

    if config.device == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            config.device = "cpu"
        else:
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            torch.cuda.empty_cache()

    # ---- wandb (локально/оффлайн) ----
    # Требование: локальный wandb, поэтому всегда offline.
    os.environ["WANDB_MODE"] = "offline"
    if args.wandb_dir is None:
        args.wandb_dir = os.path.join(args.checkpoint_dir, "wandb_offline")
    os.environ["WANDB_DIR"] = args.wandb_dir

    if hasattr(config, "wandb") and hasattr(config.wandb, "api_key"):
        # При offline-режиме логин может не требоваться, но пусть будет как в остальных скриптах.
        try:
            os.environ["WANDB_API_KEY"] = config.wandb.api_key
            wandb.login(key=config.wandb.api_key)
        except Exception as e:
            print(f"Warning: WandB login failed: {e}")

    exp_name = getattr(config.wandb, "name_exp", "default") if hasattr(config, "wandb") else "default"
    exp_name = f"{exp_name}_outlier_sevens_{int(round(args.outlier_prob * 100))}pct"
    project = getattr(config.wandb, "project", "EFMGenerationCMNIST") if hasattr(config, "wandb") else "EFMGenerationCMNIST"

    try:
        wandb.init(project=project, name=exp_name, mode="offline", dir=args.wandb_dir)
    except Exception:
        print("Warning: WandB init failed; continuing without logging to wandb.")

    # ---- data ----
    inlier_digits = parse_digits(args.inlier_digits)
    outlier_digits = parse_digits(args.outlier_digits)
    outlier_prob = float(args.outlier_prob)
    if not (0.0 <= outlier_prob <= 1.0):
        raise ValueError("--outlier-prob must be in [0, 1]")

    TRANSFORM = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(config.data.img_resize),
            torchvision.transforms.ToTensor(),
            random_color,
            torchvision.transforms.Normalize([0.5], [0.5]),
        ]
    )

    train_data = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=TRANSFORM,
    )
    eval_data = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=TRANSFORM,
    )

    print("Setting up data loaders with outlier mixing...")
    batch_size = config.training.batch_size

    train_loader = build_mnist_mixed_loader(
        mnist_dataset=train_data,
        batch_size=batch_size,
        inlier_digits=inlier_digits,
        outlier_digits=outlier_digits,
        outlier_prob=outlier_prob,
        seed=args.seed,
    )

    eval_loader_inliers = build_filtered_mnist_loader(
        mnist_dataset=eval_data,
        batch_size=batch_size,
        digits=inlier_digits,
        shuffle=True,
        drop_last=True,
    )
    eval_loader_outliers = build_filtered_mnist_loader(
        mnist_dataset=eval_data,
        batch_size=batch_size,
        digits=outlier_digits,
        shuffle=True,
        drop_last=True,
    )

    if eval_loader_inliers is None:
        raise RuntimeError("No inlier eval loader could be built (unexpected MNIST state).")

    if eval_loader_outliers is None:
        print("Warning: No outlier eval loader could be built; outlier eval will be skipped.")

    # ---- model ----
    print("Initializing model...")
    net = DDPM(config).to(config.device)
    params = net.parameters()
    optimizer = torch.optim.Adam(
        params,
        lr=config.optim.lr,
        betas=(config.optim.beta1, 0.999),
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay,
    )

    ema = ExponentialMovingAverage(net.parameters(), decay=config.model.ema_rate)
    start_step = 0

    if args.resume:
        start_step = load_checkpoint(args.resume, net, optimizer, ema, config.device)

    state = dict(optimizer=optimizer, model=net, ema=ema, step=start_step)
    optimize_fn = optimization_manager(config)

    class EFMWithCheckpoints(EFM):
        def train(
            self,
            train_loader,
            eval_loader_inliers,
            eval_loader_outliers,
            net,
            optimizer,
            optimize_fn,
            state,
            **kwargs,
        ):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            train_iter = iter(train_loader)
            eval_iter_in = iter(eval_loader_inliers)
            eval_iter_out = iter(eval_loader_outliers) if eval_loader_outliers is not None else None

            start_step = state.get("step", 0)
            for step in tqdm(
                range(start_step, self._config.training.n_iters + 1),
                desc="EFM train",
                dynamic_ncols=True,
            ):
                try:
                    batch_x, _ = next(train_iter)
                except StopIteration:
                    print("stop")
                    train_iter = iter(train_loader)
                    batch_x, _ = next(train_iter)

                batch_x = batch_x.to(self._config.device)
                batch_y = torch.randn_like(batch_x).to(self._config.device)

                optimizer = state["optimizer"]
                optimizer.zero_grad()

                perturbed_samples_vec = self.forward_interpolation(
                    batch_x[: self._config.training.small_batch_size],
                    batch_y[: self._config.training.small_batch_size],
                )

                try:
                    assert torch.isnan(perturbed_samples_vec).any().item() is False
                except AssertionError:
                    print("None values in perturbed samples between plates")
                else:
                    perturbed_samples_vec = self.forward_interpolation(
                        batch_x[: self._config.training.small_batch_size],
                        batch_y[: self._config.training.small_batch_size],
                    )

                field = self.GroundTruth(
                    perturbed_samples_vec,
                    torch.cat(
                        [
                            self._config.p.x_loc * torch.ones(len(batch_x))[:, None].to(self._config.device),
                            batch_x.view(-1, self._config.DIM - 1),
                        ],
                        dim=1,
                    ),
                    torch.cat(
                        [
                            self._config.q.x_loc * torch.ones(len(batch_y))[:, None].to(self._config.device),
                            batch_y.view(-1, self._config.DIM - 1),
                        ],
                        dim=1,
                    ),
                )

                try:
                    assert torch.isnan(perturbed_samples_vec).any().item() is False
                except AssertionError:
                    print("None values in Ground Truth field")
                else:
                    field = self.GroundTruth(
                        perturbed_samples_vec,
                        torch.cat(
                            [
                                self._config.p.x_loc * torch.ones(len(batch_x))[:, None].to(self._config.device),
                                batch_x.view(-1, self._config.DIM - 1),
                            ],
                            dim=1,
                        ),
                        torch.cat(
                            [
                                self._config.q.x_loc * torch.ones(len(batch_y))[:, None].to(self._config.device),
                                batch_y.view(-1, self._config.DIM - 1),
                            ],
                            dim=1,
                        ),
                    )

                perturbed_samples_x = perturbed_samples_vec[:, 1:].view(
                    -1,
                    self._config.data.num_channels,
                    self._config.data.image_size,
                    self._config.data.image_size,
                )
                perturbed_samples_z = perturbed_samples_vec[:, 0]
                net_x, net_z = net(perturbed_samples_x, perturbed_samples_z)

                try:
                    assert torch.isnan(net_x).any().item() is False
                    assert torch.isnan(net_z).any().item() is False
                except AssertionError:
                    print("None values in network prediction")
                    continue
                else:
                    net_x, net_z = net(perturbed_samples_x, perturbed_samples_z)

                net_x = net_x.reshape(net_x.shape[0], -1)
                pred = torch.cat([net_z[:, None], net_x], dim=1)

                loss = torch.mean((field - pred) ** 2)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf loss at step {step}, skipping backward")
                    continue

                loss.backward()

                has_nan_grad = False
                for name, param in net.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"NaN gradient in {name} at step {step}")
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    optimizer.zero_grad()
                    continue

                optimize_fn(optimizer, net, step=state["step"], config=self._config)
                state["step"] += 1
                state["ema"].update(net.parameters())
                wandb.log({"loss train": loss.item()}, step=step)

                if torch.cuda.is_available():
                    del loss, pred, field, net_x, net_z, perturbed_samples_x, perturbed_samples_z
                    if step % 100 == 0:
                        torch.cuda.empty_cache()

                def compute_eval_loss(batch_x_eval: torch.Tensor, batch_y_eval: torch.Tensor) -> torch.Tensor:
                    perturbed_samples_vec_eval = self.forward_interpolation(
                        batch_x_eval[: self._config.training.small_batch_size],
                        batch_y_eval[: self._config.training.small_batch_size],
                    )

                    field_eval = self.GroundTruth(
                        perturbed_samples_vec_eval,
                        torch.cat(
                            [
                                self._config.p.x_loc * torch.ones(len(batch_x_eval))[:, None].to(self._config.device),
                                batch_x_eval.view(-1, self._config.DIM - 1),
                            ],
                            dim=1,
                        ),
                        torch.cat(
                            [
                                self._config.q.x_loc * torch.ones(len(batch_y_eval))[:, None].to(self._config.device),
                                batch_y_eval.view(-1, self._config.DIM - 1),
                            ],
                            dim=1,
                        ),
                    )

                    perturbed_samples_x_eval = perturbed_samples_vec_eval[:, 1:].view(
                        -1,
                        self._config.data.num_channels,
                        self._config.data.image_size,
                        self._config.data.image_size,
                    )
                    perturbed_samples_z_eval = perturbed_samples_vec_eval[:, 0]
                    net_x_eval, net_z_eval = net(perturbed_samples_x_eval, perturbed_samples_z_eval)

                    net_x_eval = net_x_eval.view(net_x_eval.shape[0], -1)
                    pred_eval = torch.cat([net_z_eval[:, None], net_x_eval], dim=1)
                    return torch.mean((field_eval - pred_eval) ** 2)

                if step % self._config.training.eval_freq == 0:
                    # Inliers eval
                    try:
                        batch_x_in, _ = next(eval_iter_in)
                    except StopIteration:
                        print("stop")
                        eval_iter_in = iter(eval_loader_inliers)
                        batch_x_in, _ = next(eval_iter_in)

                    batch_x_in = batch_x_in.to(self._config.device)
                    batch_y_in = torch.randn_like(batch_x_in).to(self._config.device)

                    # Outliers eval (optional)
                    batch_x_out = None
                    batch_y_out = None
                    if eval_iter_out is not None:
                        try:
                            batch_x_out, _ = next(eval_iter_out)
                        except StopIteration:
                            print("stop")
                            eval_iter_out = iter(eval_loader_outliers)
                            batch_x_out, _ = next(eval_iter_out)
                        batch_x_out = batch_x_out.to(self._config.device)
                        batch_y_out = torch.randn_like(batch_x_out).to(self._config.device)

                    with torch.no_grad():
                        ema = state["ema"]
                        ema.store(net.parameters())
                        ema.copy_to(net.parameters())

                        eval_loss_in = compute_eval_loss(batch_x_in, batch_y_in)
                        wandb.log({"loss eval": eval_loss_in.item()}, step=step)

                        if batch_x_out is not None:
                            eval_loss_out = compute_eval_loss(batch_x_out, batch_y_out)
                            wandb.log({"loss eval outliers": eval_loss_out.item()}, step=step)

                        ema.restore(net.parameters())

                if step % self._config.training.snapshot_freq == 0:
                    save_checkpoint(state, self._config.checkpoint_dir, step)

                    with torch.no_grad():
                        ema = state["ema"]
                        ema.store(net.parameters())
                        ema.copy_to(net.parameters())

                        shape = (
                            25,
                            self._config.data.num_channels,
                            self._config.data.image_size,
                            self._config.data.image_size,
                        )

                        batch_y = torch.randn(*shape)

                        from src.ode import get_rk45_sampler_pfgm

                        sampling_fn = get_rk45_sampler_pfgm(
                            y=batch_y,
                            config=self._config,
                            shape=shape,
                            eps=self._config.training.epsilon,
                            device=self._config.device,
                        )
                        sample, n, traj = sampling_fn(net, batch_y)

                        sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                        batch_y = np.clip(batch_y.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                        fig_1 = self.plot(sample.reshape(5, 5, 32, 32, 3))
                        fig_2 = self.plot(batch_y.reshape(5, 5, 32, 32, 3))
                        fig_3 = self.plot_trajectory(traj)
                        wandb.log({"Generated Images RK45": fig_1}, step=step)
                        wandb.log({"Init Images": fig_2}, step=step)
                        wandb.log({"Trajectories RK45": fig_3}, step=step)

                        # Euler-сэмплинг отключен в этом эксперименте
                        # (не логируем "Generated Images Euler").

            return net, state

    print("Starting training...")
    efm = EFMWithCheckpoints(config, learnable_masses=getattr(config.training, "learnable_masses", False))
    net, state = efm.train(train_loader, eval_loader_inliers, eval_loader_outliers, net, optimizer, optimize_fn, state)

    print("Saving final checkpoint...")
    save_checkpoint(state, args.checkpoint_dir, state["step"])

    print("Training completed!")
    wandb.finish()


if __name__ == "__main__":
    main()

