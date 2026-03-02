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

from src.models import DDPM, ExponentialMovingAverage
from src.efm_field import EFM
from src.utils import Config, optimization_manager, random_color


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


def save_checkpoint(state, checkpoint_dir, step):
    """Сохраняет чекпоинт модели"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pth")
    torch.save({
        'step': step,
        'model_state_dict': state['model'].state_dict(),
        'optimizer_state_dict': state['optimizer'].state_dict(),
        'ema_state_dict': state['ema'].state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, ema, device):
    """Загружает чекпоинт модели"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    ema.load_state_dict(checkpoint['ema_state_dict'])
    
    start_step = checkpoint.get('step', 0)
    print(f"Checkpoint loaded. Resuming from step {start_step}")
    
    return start_step


def main():
    parser = argparse.ArgumentParser(description='EFM Training for Colored MNIST')
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
    
    if not hasattr(config, 'DIM') or config.DIM is None:
        config.DIM = config.data.num_channels * config.data.image_size * config.data.image_size + 1
    
    if not hasattr(config, 'q') or not hasattr(config.q, 'x_loc') or config.q.x_loc is None:
        if not hasattr(config, 'q'):
            config.q = Config()
        config.q.x_loc = config.L
    elif config.q.x_loc != config.L:
        config.q.x_loc = config.L
    
    config.checkpoint_dir = args.checkpoint_dir
    
    if hasattr(config, 'sampling'):
        if hasattr(config.training, 'epsilon'):
            config.sampling.z_min = config.training.epsilon
        if hasattr(config, 'L'):
            config.sampling.z_max = config.L
    
    if config.device == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            config.device = 'cpu'
        else:
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            torch.cuda.empty_cache()
    
    if hasattr(config, 'wandb') and hasattr(config.wandb, 'api_key'):
        os.environ['WANDB_API_KEY'] = config.wandb.api_key
        try:
            wandb.login(key=config.wandb.api_key)
        except Exception as e:
            print(f"Warning: WandB login failed: {e}")
            print("Continuing with offline mode...")
    
    wandb_mode = getattr(config.wandb, 'mode', 'offline') if hasattr(config, 'wandb') else 'offline'
    try:
        wandb.init(
            project=getattr(config.wandb, 'project', 'EFMGenerationCMNIST') if hasattr(config, 'wandb') else 'EFMGenerationCMNIST',
            name=getattr(config.wandb, 'name_exp', 'default') if hasattr(config, 'wandb') else 'default',
            mode=wandb_mode
        )
    except Exception as e:
        print(f"Warning: WandB init failed: {e}")
        print("Switching to offline mode...")
        wandb.init(
            project=getattr(config.wandb, 'project', 'EFMGenerationCMNIST') if hasattr(config, 'wandb') else 'EFMGenerationCMNIST',
            name=getattr(config.wandb, 'name_exp', 'default') if hasattr(config, 'wandb') else 'default',
            mode='offline'
        )
    
    print("Setting up data loaders...")
    TRANSFORM = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config.data.img_resize),
        torchvision.transforms.ToTensor(),
        random_color,
        torchvision.transforms.Normalize([0.5], [0.5])
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
        shuffle=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_data, 
        batch_size=config.training.batch_size, 
        shuffle=True
    )
    
    print("Initializing model...")
    net = DDPM(config).to(config.device)
    params = net.parameters()
    optimizer = torch.optim.Adam(
        params,
        lr=config.optim.lr, 
        betas=(config.optim.beta1, 0.999), 
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay
    )
    
    ema = ExponentialMovingAverage(net.parameters(), decay=config.model.ema_rate)
    start_step = 0
    
    if args.resume:
        start_step = load_checkpoint(args.resume, net, optimizer, ema, config.device)
    
    state = dict(optimizer=optimizer, model=net, ema=ema, step=start_step)
    optimize_fn = optimization_manager(config)
    
    class EFMWithCheckpoints(EFM):
        def train(self, train_loader, eval_loader, net, optimizer, optimize_fn, state, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            train_iter = iter(train_loader)
            eval_iter = iter(eval_loader)
            
            start_step = state.get('step', 0)
            for step in range(start_step, self._config.training.n_iters + 1):
                try:
                    batch_x, _ = next(train_iter)
                except StopIteration:
                    print('stop')
                    train_iter = iter(train_loader)
                    batch_x, _ = next(train_iter)
                batch_x = batch_x.to(self._config.device)
                batch_y = torch.randn_like(batch_x).to(self._config.device)
                
                optimizer = state['optimizer']
                optimizer.zero_grad()
                
                perturbed_samples_vec = self.forward_interpolation(
                    batch_x[:self._config.training.small_batch_size],
                    batch_y[:self._config.training.small_batch_size]
                )
                
                try:
                    assert torch.isnan(perturbed_samples_vec).any().item() == False
                except AssertionError:
                    print('None values in perturbed samples between plates')
                else:
                    perturbed_samples_vec = self.forward_interpolation(
                        batch_x[:self._config.training.small_batch_size],
                        batch_y[:self._config.training.small_batch_size]
                    )
                
                field = self.GroundTruth(
                    perturbed_samples_vec,
                    torch.cat([
                        self._config.p.x_loc*torch.ones(len(batch_x))[:,None].to(self._config.device),
                        batch_x.view(-1,self._config.DIM-1)
                    ], dim=1),
                    torch.cat([
                        self._config.q.x_loc*torch.ones(len(batch_y))[:,None].to(self._config.device),
                        batch_y.view(-1,self._config.DIM-1)
                    ], dim=1)
                )
                
                try:
                    assert torch.isnan(perturbed_samples_vec).any().item() == False
                except AssertionError:
                    print('None values in Ground Truth field')
                else:
                    field = self.GroundTruth(
                        perturbed_samples_vec,
                        torch.cat([
                            self._config.p.x_loc*torch.ones(len(batch_x))[:,None].to(self._config.device),
                            batch_x.view(-1,self._config.DIM-1)
                        ], dim=1),
                        torch.cat([
                            self._config.q.x_loc*torch.ones(len(batch_y))[:,None].to(self._config.device),
                            batch_y.view(-1,self._config.DIM-1)
                        ], dim=1)
                    )
                
                perturbed_samples_x = perturbed_samples_vec[:, 1:].view(
                    -1, self._config.data.num_channels,
                    self._config.data.image_size,
                    self._config.data.image_size
                )
                perturbed_samples_z = perturbed_samples_vec[:, 0]
                net_x, net_z = net(perturbed_samples_x, perturbed_samples_z)
                
                try:
                    assert torch.isnan(net_x).any().item() == False
                    assert torch.isnan(net_z).any().item() == False
                except AssertionError:
                    print('None values in network prediction')
                    continue
                else:
                    net_x, net_z = net(perturbed_samples_x, perturbed_samples_z)
                
                net_x = net_x.reshape(net_x.shape[0], -1)
                pred = torch.cat([net_z[:, None], net_x], dim=1)
                
                loss = torch.mean((field - pred)**2)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f'NaN/Inf loss at step {step}, skipping backward')
                    continue
                
                loss.backward()
                
                has_nan_grad = False
                for name, param in net.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f'NaN gradient in {name} at step {step}')
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    optimizer.zero_grad()
                    continue
                
                optimize_fn(optimizer, net, step=state['step'], config=self._config)
                state['step'] += 1
                state['ema'].update(net.parameters())
                wandb.log({"loss train": loss.item()}, step=step)
                
                if torch.cuda.is_available():
                    del loss, pred, field, net_x, net_z, perturbed_samples_x, perturbed_samples_z
                    if step % 100 == 0:
                        torch.cuda.empty_cache()
                
                if step % self._config.training.eval_freq == 0:
                    try:
                        batch_x, _ = next(eval_iter)
                    except StopIteration:
                        print('stop')
                        eval_iter = iter(eval_loader)
                        batch_x, _ = next(eval_iter)
                    batch_x = batch_x.to(self._config.device)
                    batch_y = torch.randn_like(batch_x).to(self._config.device)
                    
                    with torch.no_grad():
                        ema = state['ema']
                        ema.store(net.parameters())
                        ema.copy_to(net.parameters())
                        
                        perturbed_samples_vec = self.forward_interpolation(
                            batch_x[:self._config.training.small_batch_size],
                            batch_y[:self._config.training.small_batch_size]
                        )
                        
                        field = self.GroundTruth(
                            perturbed_samples_vec,
                            torch.cat([
                                self._config.p.x_loc*torch.ones(len(batch_x))[:,None].to(self._config.device),
                                batch_x.view(-1,self._config.DIM-1)
                            ], dim=1),
                            torch.cat([
                                self._config.q.x_loc*torch.ones(len(batch_y))[:,None].to(self._config.device),
                                batch_y.view(-1,self._config.DIM-1)
                            ], dim=1)
                        )
                        
                        perturbed_samples_x = perturbed_samples_vec[:, 1:].view(
                            -1, self._config.data.num_channels,
                            self._config.data.image_size,
                            self._config.data.image_size
                        )
                        perturbed_samples_z = perturbed_samples_vec[:, 0]
                        net_x, net_z = net(perturbed_samples_x, perturbed_samples_z)
                        net_x = net_x.view(net_x.shape[0], -1)
                        pred = torch.cat([net_z[:, None], net_x], dim=1)
                        
                        eval_loss = torch.mean((field - pred)**2)
                        
                        ema.restore(net.parameters())
                        wandb.log({"loss eval": eval_loss.item()}, step=step)
                
                if step % self._config.training.snapshot_freq == 0:
                    save_checkpoint(state, self._config.checkpoint_dir, step)
                    
                    with torch.no_grad():
                        ema = state['ema']
                        ema.store(net.parameters())
                        ema.copy_to(net.parameters())
                        
                        shape = (25, self._config.data.num_channels,
                                 self._config.data.image_size, self._config.data.image_size)
                        
                        batch_y = torch.randn(*shape)
                        
                        from src.ode import get_rk45_sampler_pfgm, LearnedImageODESolver
                        
                        sampling_fn = get_rk45_sampler_pfgm(
                            y=batch_y, 
                            config=self._config,
                            shape=shape,
                            eps=self._config.training.epsilon,
                            device=self._config.device
                        )
                        sample, n, traj = sampling_fn(net, batch_y)
                        
                        sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                        batch_y = np.clip(batch_y.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                        fig_1 = self.plot(sample.reshape(5,5,32,32,3))
                        fig_2 = self.plot(batch_y.reshape(5,5,32,32,3))
                        fig_3 = self.plot_trajectory(traj)
                        wandb.log({"Generated Images RK45": fig_1}, step=step)
                        wandb.log({"Init Images": fig_2}, step=step)
                        wandb.log({"Trajectories RK45": fig_3}, step=step)
                        
                        ode_solver = LearnedImageODESolver(net, self._config)
                        batch_y = torch.randn(*shape).to(self._config.device)
                        sample, traj = ode_solver(torch.cat([
                            (self._config.L)*torch.ones(batch_y.shape[0], device=batch_y.device)[:,None],
                            batch_y.view(-1, self._config.DIM-1)
                        ], dim=1).to(self._config.device))
                        
                        fig_1 = self.plota(sample[:,1:].reshape(5,5,3,32,32).detach().cpu())
                        wandb.log({"Generated Images Euler": fig_1}, step=step)
            
            return net, state
    
    print("Starting training...")
    efm = EFMWithCheckpoints(config, learnable_masses=getattr(config.training, 'learnable_masses', False))
    net, state = efm.train(train_loader, eval_loader, net, optimizer, optimize_fn, state)
    
    print("Saving final checkpoint...")
    save_checkpoint(state, args.checkpoint_dir, state['step'])
    
    print("Training completed!")
    wandb.finish()


if __name__ == '__main__':
    main()
