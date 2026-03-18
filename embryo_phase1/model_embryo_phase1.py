"""
Phase1: diffusion over 16 embryo stages conditioned on time series only.
- Time encoder maps (B, 1, T) time -> (B, input_dim, T).
- Noise and denoise only on valid (labeled) timesteps; starting/ending untouched.
"""
from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import diffusion utilities and decoder from main model
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import (
    DecoderModel,
    get_timestep_embedding,
    swish,
    extract,
    cosine_beta_schedule,
    normalize,
    denormalize,
)


class TimeEncoder(nn.Module):
    """Maps (B, 1, T) time series to (B, output_dim, T) for decoder cross-attention."""

    def __init__(self, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class EmbryoPhase1Diffusion(nn.Module):
    """Diffusion over 16 stages conditioned on time. Valid mask excludes starting/ending from noise and loss."""

    def __init__(self, time_encoder_output_dim: int, decoder_params: dict, diffusion_params: dict, num_classes: int, device: torch.device):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.num_timesteps = int(diffusion_params["timesteps"])
        self.sampling_timesteps = diffusion_params["sampling_timesteps"]
        self.ddim_sampling_eta = diffusion_params["ddim_sampling_eta"]
        self.scale = diffusion_params["snr_scale"]

        betas = cosine_beta_schedule(self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

        self.time_encoder = TimeEncoder(time_encoder_output_dim)
        decoder_params = dict(decoder_params)
        decoder_params["input_dim"] = time_encoder_output_dim
        decoder_params["num_classes"] = num_classes
        self.decoder = DecoderModel(**decoder_params)

    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0)
            / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)
        sqrt_ac = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_omc = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_ac * x_start + sqrt_omc * noise

    def prepare_targets(self, event_gt: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Noise only on valid timesteps. event_gt (B, C, T), valid_mask (B, 1, T)."""
        assert event_gt.dim() == 3 and valid_mask.dim() == 3
        t = torch.randint(0, self.num_timesteps, (event_gt.shape[0],), device=self.device).long()
        noise = torch.randn_like(event_gt, device=self.device)
        noise = noise * valid_mask
        x_start = event_gt * valid_mask
        x_start = (x_start * 2.0 - 1.0) * self.scale
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x = torch.clamp(x, min=-self.scale, max=self.scale)
        event_diffused = (x / self.scale + 1.0) / 2.0
        return event_diffused, noise, t

    def forward(self, time_feats: torch.Tensor, t: torch.Tensor, event_diffused: torch.Tensor) -> torch.Tensor:
        return self.decoder(time_feats, t, event_diffused.float())

    def model_predictions(self, time_feats: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_m = torch.clamp(x, min=-self.scale, max=self.scale)
        x_m = denormalize(x_m, self.scale)
        x_start = self.decoder(time_feats, t, x_m.float())
        x_start = F.softmax(x_start, 1)
        x_start = normalize(x_start, self.scale)
        x_start = torch.clamp(x_start, min=-self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def get_training_loss(
        self,
        time_series: torch.Tensor,
        event_gt: torch.Tensor,
        valid_mask: torch.Tensor,
        decoder_ce_criterion: nn.Module,
        decoder_mse_criterion: nn.Module,
    ) -> dict[str, torch.Tensor]:
        time_feats = self.time_encoder(time_series)
        event_diffused, noise, t = self.prepare_targets(event_gt, valid_mask)
        event_out = self.forward(time_feats, t, event_diffused)

        ce_per_frame = decoder_ce_criterion(
            event_out.transpose(2, 1).contiguous().view(-1, self.num_classes),
            torch.argmax(event_gt, dim=1).view(-1).long(),
        )
        ce_per_frame = ce_per_frame.view(event_gt.shape[0], -1)
        n_valid = valid_mask.squeeze(1).sum(dim=1, keepdim=True).clamp(min=1)
        decoder_ce_loss = (ce_per_frame * valid_mask.squeeze(1)).sum(dim=1) / n_valid.squeeze(1)
        decoder_ce_loss = decoder_ce_loss.mean()

        log_out = F.log_softmax(event_out, dim=1)
        mse_per = F.mse_loss(log_out[:, :, 1:], log_out.detach()[:, :, :-1], reduction="none").mean(dim=1)
        decoder_mse_loss = (mse_per * valid_mask.squeeze(1)[:, :-1]).sum(dim=1) / (valid_mask[:, :, 1:].sum(dim=(1, 2)).clamp(min=1))
        decoder_mse_loss = decoder_mse_loss.mean()

        return {"decoder_ce_loss": decoder_ce_loss, "decoder_mse_loss": decoder_mse_loss}

    @torch.no_grad()
    def ddim_sample(self, time_series: torch.Tensor, valid_mask: torch.Tensor | None = None, seed: int | None = None) -> torch.Tensor:
        time_feats = self.time_encoder(time_series)
        B, _, T = time_series.shape
        shape = (B, self.num_classes, T)
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1, device=self.device)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        x_time = torch.randn(shape, device=self.device)
        if valid_mask is not None:
            x_time = x_time * valid_mask
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((B,), time, device=self.device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(time_feats, x_time, time_cond)
            x_return = x_start.clone()
            if time_next < 0:
                x_time = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha_next) / (1 - alpha)).sqrt() * (1 - alpha / alpha_next).sqrt()
            pred = (x_time - (1 - alpha).sqrt() * pred_noise) / alpha.sqrt()
            x_time = alpha_next.sqrt() * pred + (1 - alpha_next - sigma ** 2).sqrt() * pred_noise
            if sigma > 0:
                x_time = x_time + sigma * torch.randn_like(x_time, device=self.device)
            x_time = torch.clamp(x_time, min=-self.scale, max=self.scale)
        x_return = denormalize(x_return, self.scale)
        if valid_mask is not None:
            x_return = x_return * valid_mask
        return x_return
