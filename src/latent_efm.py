import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import typing as tp
import wandb


class LatentEFM:
    def __init__(self, config, encoder, decoder, field_network, metric_loss=None):
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.field_network = field_network
        self.metric_loss = metric_loss

        self.latent_dim = config.model.latent_dim
        self.device = config.device

    def forward_interpolation_latent(self, z_real, z_noise):
        batch_size = z_real.shape[0]

        if self.config.training.interpolation == 'Uniform':
            t = torch.distributions.Uniform(
                low=self.config.p.x_loc + self.config.training.epsilon,
                high=self.config.q.x_loc - self.config.training.epsilon
            ).sample(torch.Size([batch_size])).to(self.device)

            alpha = t / self.config.L
            z_interp = (1 - alpha[:, None]) * z_real + alpha[:, None] * z_noise
            perturbed_vec = torch.cat([t[:, None], z_interp], dim=1)

        elif self.config.training.interpolation == 'Gaussian_mixing':
            m = torch.rand((batch_size,), device=self.device) * self.config.training.M
            tau = self.config.training.tau
            z = torch.randn(batch_size, 1).to(self.device) * self.config.training.sigma_end
            z = z.abs()

            if self.config.training.restrict_M:
                idx = (z < 0.005).squeeze()
                num = int(idx.int().sum())
                if num > 0:
                    restrict_m = int(self.config.training.M * 0.7)
                    m[idx] = torch.rand((num,), device=self.device) * restrict_m

            multiplier = (1 + tau) ** m
            perturbed_z = z.squeeze() * multiplier

            gaussian = torch.randn(batch_size, self.latent_dim).to(self.device)
            unit_gaussian = gaussian / (torch.norm(gaussian, p=2, dim=1, keepdim=True) + 1e-8)

            noise = torch.randn_like(z_real) * self.config.training.sigma_end
            norm_m = torch.norm(noise, p=2, dim=1) * multiplier

            perturbation = unit_gaussian * norm_m[:, None]
            z_perturbed = z_real + perturbation

            perturbed_vec = torch.cat([perturbed_z[:, None], z_perturbed], dim=1)

        elif self.config.training.interpolation == 'Uniform_mixing':
            m = torch.rand((batch_size,), device=self.device) * self.config.training.M
            tau = self.config.training.tau
            z = torch.randn(batch_size, 1).to(self.device) * self.config.training.sigma_end
            z = z.abs()

            if self.config.training.restrict_M:
                idx = (z < 0.005).squeeze()
                num = int(idx.int().sum())
                if num > 0:
                    restrict_m = int(self.config.training.M * 0.7)
                    m[idx] = torch.rand((num,), device=self.device) * restrict_m

            multiplier = (1 + tau) ** m
            perturbed_z = z.squeeze() * multiplier

            alpha = perturbed_z / self.config.L
            z_interp = (1 - alpha[:, None]) * z_real + alpha[:, None] * z_noise

            perturbed_vec = torch.cat([perturbed_z[:, None], z_interp], dim=1)

        else:
            raise ValueError(f"Unknown interpolation: {self.config.training.interpolation}")

        return perturbed_vec

    def compute_ground_truth_field(self, perturbed_vec, z_real, z_noise):
        gt_distance_real = torch.norm(
            perturbed_vec.unsqueeze(1) - z_real.unsqueeze(0),
            dim=-1
        )
        gt_distance_noise = torch.norm(
            perturbed_vec.unsqueeze(1) - z_noise.unsqueeze(0),
            dim=-1
        )

        if self.config.training.stability:
            distance_real = torch.min(gt_distance_real, dim=1, keepdim=True)[0] / (gt_distance_real + 1e-7)
            distance_noise = torch.min(gt_distance_noise, dim=1, keepdim=True)[0] / (gt_distance_noise + 1e-7)
        else:
            distance_real = 1.0 / (gt_distance_real + 1e-7)
            distance_noise = 1.0 / (gt_distance_noise + 1e-7)

        data_dim = self.latent_dim + 1
        distance_real = distance_real ** data_dim
        distance_noise = distance_noise ** data_dim

        distance_real = distance_real[:, :, None]
        distance_noise = distance_noise[:, :, None]

        coeff_real = distance_real / (torch.sum(distance_real, dim=1, keepdim=True) + 1e-7)
        coeff_noise = distance_noise / (torch.sum(distance_noise, dim=1, keepdim=True) + 1e-7)

        diff_real = -(perturbed_vec.unsqueeze(1) - z_real.unsqueeze(0))
        diff_noise = -(perturbed_vec.unsqueeze(1) - z_noise.unsqueeze(0))

        field_real = torch.sum(coeff_real * diff_real, dim=1)
        field_noise = torch.sum(coeff_noise * diff_noise, dim=1)

        norm_real = field_real.norm(p=2, dim=-1, keepdim=True)
        norm_noise = field_noise.norm(p=2, dim=-1, keepdim=True)

        field_real = field_real / (norm_real + self.config.training.gamma)
        field_noise = field_noise / (norm_noise + self.config.training.gamma)

        field_real *= np.sqrt(data_dim)
        field_noise *= np.sqrt(data_dim)

        field = -field_real + field_noise

        return field

    def train(self, train_loader, eval_loader, optimizer, state, **kwargs):
        train_iter = iter(train_loader)
        eval_iter = iter(eval_loader)

        for step in tqdm(range(state['step'], self.config.training.n_iters + 1)):
            try:
                batch_x, batch_labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch_x, batch_labels = next(train_iter)

            batch_x = batch_x.to(self.device)
            batch_labels = batch_labels.to(self.device)

            optimizer.zero_grad()

            enc_out = self.encoder(batch_x, return_projection=True)
            if isinstance(enc_out, tuple):
                z_real, z_proj = enc_out
            else:
                z_real, z_proj = enc_out, None

            loss_metric = 0
            metric_log = {}
            if self.metric_loss is not None:
                features_for_metric = z_proj if z_proj is not None else z_real
                loss_metric, metric_log = self.metric_loss(features_for_metric, batch_labels)

            z_noise = torch.randn_like(z_real)

            z_real_aug = torch.cat([
                self.config.p.x_loc * torch.ones(z_real.shape[0], 1).to(self.device),
                z_real
            ], dim=1)

            z_noise_aug = torch.cat([
                self.config.q.x_loc * torch.ones(z_noise.shape[0], 1).to(self.device),
                z_noise
            ], dim=1)

            perturbed_vec = self.forward_interpolation_latent(z_real, z_noise)

            if torch.isnan(perturbed_vec).any():
                print(f'NaN in perturbed_vec at step {step}, skipping')
                continue

            field_gt = self.compute_ground_truth_field(
                perturbed_vec, z_real_aug, z_noise_aug
            )

            if torch.isnan(field_gt).any():
                print(f'NaN in field_gt at step {step}, skipping')
                continue

            z_coord = perturbed_vec[:, 0]
            z_latent = perturbed_vec[:, 1:]

            field_z_coord, field_z_latent = self.field_network(z_coord, z_latent)
            field_pred = torch.cat([field_z_coord[:, None], field_z_latent], dim=1)

            if torch.isnan(field_pred).any():
                print(f'NaN in field_pred at step {step}, skipping')
                continue

            loss_field = torch.mean((field_gt - field_pred) ** 2)

            if torch.isnan(loss_field) or torch.isinf(loss_field):
                print(f'NaN/Inf loss_field at step {step}, skipping')
                continue

            loss_recon = 0
            if kwargs.get('use_recon_loss', False):
                x_recon = self.decoder(z_real)
                loss_recon = F.mse_loss(x_recon, batch_x) * kwargs.get('recon_weight', 0.1)

            total_loss = loss_field + self.config.training.lambda_metric * loss_metric + loss_recon

            total_loss.backward()

            has_nan_grad = False
            for name, param in [('encoder', self.encoder),
                                ('decoder', self.decoder),
                                ('field', self.field_network)]:
                for pname, p in param.named_parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        print(f'NaN gradient in {name}.{pname} at step {step}')
                        has_nan_grad = True
                        break
                if has_nan_grad:
                    break

            if has_nan_grad:
                optimizer.zero_grad()
                continue

            if kwargs.get('clip_grad', False):
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) +
                    list(self.decoder.parameters()) +
                    list(self.field_network.parameters()),
                    kwargs.get('max_grad_norm', 1.0)
                )

            optimizer.step()
            state['step'] += 1
            state['ema'].update(self.field_network.parameters())

            log_dict = {
                'loss_field': loss_field.item(),
                'loss_metric': loss_metric if isinstance(loss_metric, float) else loss_metric.item(),
                'loss_total': total_loss.item()
            }
            if loss_recon != 0:
                log_dict['loss_recon'] = loss_recon.item()
            log_dict.update(metric_log)

            wandb.log(log_dict, step=step)

            if step % self.config.training.eval_freq == 0:
                self.evaluate(eval_loader, state, step)

            if step % self.config.training.snapshot_freq == 0:
                self.save_checkpoint(state, step)

                from src.ode import LatentODESolver

                with torch.no_grad():
                    ema = state['ema']
                    ema.store(self.field_network.parameters())
                    ema.copy_to(self.field_network.parameters())

                    ode_solver = LatentODESolver(self.field_network, self.config)

                    num_samples = 25
                    z_init = torch.randn(num_samples, self.latent_dim).to(self.device)

                    x_init = self.decoder(z_init)
                    x_init_np = (x_init.detach().cpu().numpy() * 255).astype(np.uint8)
                    fig_init = self.plot_images(x_init_np.transpose(0, 2, 3, 1))

                    z_rk, traj_rk = ode_solver.sample(z_init, method='rk4')
                    x_rk = self.decoder(z_rk)
                    x_rk_np = (x_rk.detach().cpu().numpy() * 255).astype(np.uint8)
                    fig_rk = self.plot_images(x_rk_np.transpose(0, 2, 3, 1))

                    fig_traj_rk = self.plot_latent_trajectories(traj_rk)

                    wandb.log({
                        "Generated Images RK45 Latent": fig_rk,
                        "Init Images Latent": fig_init,
                        "Trajectories RK45 Latent": fig_traj_rk
                    }, step=step)

                    plt.close(fig_init)
                    plt.close(fig_rk)
                    if fig_traj_rk is not None:
                        plt.close(fig_traj_rk)

                    z_euler, traj_euler = ode_solver.sample(z_init, method='euler')
                    x_euler = self.decoder(z_euler)
                    x_euler_np = (x_euler.detach().cpu().numpy() * 255).astype(np.uint8)
                    fig_euler = self.plot_images(x_euler_np.transpose(0, 2, 3, 1))

                    wandb.log({
                        "Generated Images Euler Latent": fig_euler
                    }, step=step)

                    plt.close(fig_euler)

                    shape = (
                        25,
                        self.config.data.num_channels,
                        self.config.data.image_size,
                        self.config.data.image_size,
                    )
                    y_pixel = torch.randn(*shape).to(self.device)

                    y_np = (y_pixel.detach().cpu().numpy() * 255).astype(np.uint8)
                    fig_init_px = self.plot_images(y_np.transpose(0, 2, 3, 1))

                    z0 = self.encoder(y_pixel)

                    z_rk_px, traj_rk_px = ode_solver.sample(z0, method="rk4")
                    x_rk_px = self.decoder(z_rk_px)
                    x_rk_px_np = (x_rk_px.detach().cpu().numpy() * 255).astype(np.uint8)
                    fig_rk_px = self.plot_images(x_rk_px_np.transpose(0, 2, 3, 1))

                    fig_traj_px = self.plot_latent_trajectories(traj_rk_px)

                    wandb.log(
                        {
                            "Generated Images RK45 PixelLatent": fig_rk_px,
                            "Init Images PixelLatent": fig_init_px,
                            "Trajectories RK45 PixelLatent": fig_traj_px,
                        },
                        step=step,
                    )

                    plt.close(fig_init_px)
                    plt.close(fig_rk_px)
                    if fig_traj_px is not None:
                        plt.close(fig_traj_px)

                    ema.restore(self.field_network.parameters())

        return state

    def evaluate(self, eval_loader, state, step):
        try:
            batch_x, batch_labels = next(iter(eval_loader))
        except StopIteration:
            return

        batch_x = batch_x.to(self.device)
        batch_labels = batch_labels.to(self.device)

        with torch.no_grad():
            ema = state['ema']
            ema.store(self.field_network.parameters())
            ema.copy_to(self.field_network.parameters())

            z_real = self.encoder(batch_x)
            z_noise = torch.randn_like(z_real)

            z_real_aug = torch.cat([
                self.config.p.x_loc * torch.ones(z_real.shape[0], 1).to(self.device),
                z_real
            ], dim=1)
            z_noise_aug = torch.cat([
                self.config.q.x_loc * torch.ones(z_noise.shape[0], 1).to(self.device),
                z_noise
            ], dim=1)

            perturbed_vec = self.forward_interpolation_latent(z_real, z_noise)

            field_gt = self.compute_ground_truth_field(
                perturbed_vec, z_real_aug, z_noise_aug
            )

            z_coord = perturbed_vec[:, 0]
            z_latent = perturbed_vec[:, 1:]
            field_z_coord, field_z_latent = self.field_network(z_coord, z_latent)
            field_pred = torch.cat([field_z_coord[:, None], field_z_latent], dim=1)

            eval_loss = torch.mean((field_gt - field_pred) ** 2)

            ema.restore(self.field_network.parameters())

            wandb.log({'eval_loss': eval_loss.item()}, step=step)

    def save_checkpoint(self, state, step):
        import os
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_step_{step}.pth")

        torch.save({
            'step': step,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'field_network_state_dict': self.field_network.state_dict(),
            'optimizer_state_dict': state['optimizer'].state_dict(),
            'ema_state_dict': state['ema'].state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def plot_images(self, images, nrows=5, ncols=5):
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
        for i in range(nrows):
            for j in range(ncols):
                idx = i * ncols + j
                if idx < len(images):
                    axes[i, j].imshow(images[idx])
                axes[i, j].axis('off')
        plt.tight_layout(pad=0.1)
        return fig

    def plot_latent_trajectories(self, trajectory, num_imgs: int = 5, num_steps: int = 6):
        if trajectory is None or len(trajectory) == 0:
            return None

        T = len(trajectory)
        if T == 0:
            return None

        num_steps = min(num_steps, T)
        step_indices = np.linspace(0, T - 1, num_steps, dtype=int)

        batch_size = trajectory[0].shape[0]
        num_imgs = min(num_imgs, batch_size)

        fig, axes = plt.subplots(num_imgs, num_steps, figsize=(num_steps, num_imgs))
        if num_imgs == 1:
            axes = np.expand_dims(axes, 0)
        if num_steps == 1:
            axes = np.expand_dims(axes, 1)

        for col, t in enumerate(step_indices):
            z_t = trajectory[t][:num_imgs].to(self.device)
            with torch.no_grad():
                x_t = self.decoder(z_t).detach().cpu()

            for row in range(num_imgs):
                img = x_t[row].permute(1, 2, 0).numpy()
                axes[row, col].imshow(img)
                axes[row, col].axis("off")

        plt.tight_layout(pad=0.1)
        return fig
