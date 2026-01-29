import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MetricEncoder(nn.Module):
    def __init__(self, image_size=32, in_channels=3, latent_dim=128,
                 hidden_dims=[64, 128, 256], use_projection_head=True):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.use_projection_head = use_projection_head

        layers = [
            nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        ]

        current_dim = hidden_dims[0]
        for dim in hidden_dims:
            layers.append(ResidualBlock(current_dim, dim, stride=2))
            current_dim = dim

        self.encoder = nn.Sequential(*layers)

        spatial_size = image_size // (2 ** len(hidden_dims))
        flatten_dim = hidden_dims[-1] * spatial_size * spatial_size

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        if use_projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(inplace=True),
                nn.Linear(latent_dim, latent_dim // 2)
            )

    def forward(self, x, return_projection=False):
        features = self.encoder(x)
        z = self.fc(features)

        if return_projection and self.use_projection_head:
            z_proj = self.projection_head(z)
            return z, z_proj
        return z


class LatentDecoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, image_size=32,
                 hidden_dims=[256, 128, 64]):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.image_size = image_size

        num_upsamples = len(hidden_dims)
        initial_size = image_size // (2 ** num_upsamples)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, hidden_dims[0] * initial_size * initial_size),
            nn.ReLU(inplace=True)
        )

        self.initial_size = initial_size
        self.initial_channels = hidden_dims[0]

        layers = []
        current_dim = hidden_dims[0]
        for dim in hidden_dims[1:]:
            layers.extend([
                nn.Upsample(scale_factor=2, mode='nearest'),
                ResidualBlock(current_dim, dim),
            ])
            current_dim = dim

        layers.extend([
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(current_dim, current_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(current_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_dim, out_channels, 3, padding=1),
            nn.Sigmoid()
        ])

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.initial_channels, self.initial_size, self.initial_size)
        x_recon = self.decoder(h)
        return x_recon


class LatentFieldNetwork(nn.Module):
    def __init__(self, latent_dim=128, time_embed_dim=256,
                 hidden_dims=[512, 512, 512]):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed_dim = time_embed_dim

        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        input_dim = latent_dim + time_embed_dim
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        self.main_net = nn.Sequential(*layers)

        self.z_coord_head = nn.Linear(prev_dim, 1)
        self.z_latent_head = nn.Linear(prev_dim, latent_dim)

        nn.init.zeros_(self.z_coord_head.weight)
        nn.init.zeros_(self.z_coord_head.bias)
        nn.init.zeros_(self.z_latent_head.weight)
        nn.init.zeros_(self.z_latent_head.bias)

    def get_timestep_embedding(self, timesteps, embedding_dim, max_positions=10000):
        assert len(timesteps.shape) == 1
        half_dim = embedding_dim // 2
        emb = np.log(max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode='constant')
        return emb

    def forward(self, z_coord, z_latent):
        t_emb = self.get_timestep_embedding(z_coord, self.time_embed_dim)
        t_emb = self.time_embed(t_emb)

        h = torch.cat([z_latent, t_emb], dim=1)
        h = self.main_net(h)

        field_z_coord = self.z_coord_head(h).squeeze(-1)
        field_z_latent = self.z_latent_head(h)

        return field_z_coord, field_z_latent


class AutoEncoder(nn.Module):
    def __init__(self, image_size=32, in_channels=3, latent_dim=128):
        super().__init__()
        self.encoder = MetricEncoder(image_size, in_channels, latent_dim)
        self.decoder = LatentDecoder(latent_dim, in_channels, image_size)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
