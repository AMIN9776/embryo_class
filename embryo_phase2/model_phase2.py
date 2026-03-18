"""
Embryo Phase2 diffusion model: time + visual features (FEMI or custom encoder) as conditioning.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTMAEForPreTraining, AutoImageProcessor

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import DecoderModel, extract, cosine_beta_schedule, normalize, denormalize
from embryo_phase1.model_embryo_phase1 import TimeEncoder


class VisualEncoderFEMI(nn.Module):
    """
    Wrap FEMI ViT-MAE to produce per-frame features of shape (B, D_v, T).
    """

    def __init__(self, model_name: str, proj_dim: int, freeze: bool = True, device: torch.device | None = None):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.femi = ViTMAEForPreTraining.from_pretrained(model_name)
        if freeze:
            for p in self.femi.parameters():
                p.requires_grad_(False)
        self.device = device
        hidden_dim = self.femi.config.hidden_size
        self.proj = nn.Conv1d(hidden_dim, proj_dim, 1)

    def forward(self, images: list[list[str]], target_T: int, **kwargs: Any) -> torch.Tensor:
        """
        images: list of length B; each element is a list of image paths (len may equal target_T).
        target_T: desired temporal length (number of timesteps from padded CSV).
        **kwargs: ignored (e.g. time_series for custom encoder).
        Returns: (B, proj_dim, target_T)
        """
        # Normalize shapes coming from DataLoader:
        # - If we get a flat list[str], wrap as [list[str]] for B=1.
        # - If we get time-major [[p0], [p1], ...] with batch_size=1, collapse to [list[str]] too.
        if images:
            # Case 1: flat list[str]
            if isinstance(images[0], str):
                images = [images]
            # Case 2: time-major list of length T, each [path] (batch_size=1)
            elif isinstance(images[0], list) and len(images[0]) == 1 and all(
                isinstance(x, list) and len(x) == 1 for x in images
            ):
                images = [[x[0] for x in images]]

        B = len(images)
        if B == 0:
            raise ValueError("No images provided to VisualEncoderFEMI")
        # Flatten all (b, t) positions with non-empty path
        flat_imgs = []
        flat_indices: list[tuple[int, int]] = []
        from PIL import Image

        for b, paths in enumerate(images):
            for t, p in enumerate(paths):
                if not p:
                    continue
                try:
                    img = Image.open(p)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    flat_imgs.append(img)
                    flat_indices.append((b, t))
                except Exception:
                    continue
        if not flat_imgs:
            # No images available; return zeros with desired T
            return torch.zeros(B, self.proj.out_channels, target_T, device=self.device)
        inputs = self.processor(images=flat_imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.femi(**inputs, output_hidden_states=True)
            # ViTMAEForPreTrainingOutput exposes hidden_states as a tuple; take the last one.
            if outputs.hidden_states is not None:
                h = outputs.hidden_states[-1]  # (N, L, D)
            else:
                # Fallback: use logits as representation if hidden_states are not returned
                h = outputs.logits
        feats = h.mean(dim=1)  # (N, D)
        D = feats.shape[-1]
        # Scatter back to (B, D, target_T)
        out = torch.zeros(B, D, target_T, device=self.device)
        for (b, t), v in zip(flat_indices, feats):
            if 0 <= t < target_T:
                out[b, :, t] = v
        out = self.proj(out)  # (B, proj_dim, target_T)
        return out


class VisualEncoderCustom(nn.Module):
    """
    Wrap the pretrained visual encoder (DINOv2 + FiLM + proj) from embryo_visual_pretrain
    to produce per-frame features (B, output_dim, T) from list[list[str]] image paths.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        output_dim: int,
        num_classes: int,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        from embryo_visual_pretrain.train_visual_encoder import VisualEncoder as PretrainVisualEncoder

        self.encoder = PretrainVisualEncoder(num_classes=num_classes, device=device, dropout_p=0.0)
        ckpt = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        # Pretrain may use 15 classes (tHB excluded); we only need backbone, time_film, proj for embeddings
        ckpt_filtered = {k: v for k, v in ckpt.items() if not k.startswith("classifier.") and not k.startswith("reconstruct.")}
        self.encoder.load_state_dict(ckpt_filtered, strict=False)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        # Pretrain outputs 128-dim; optional projection to match Phase2 visual_feature_dim
        if output_dim != 128:
            self.proj = nn.Conv1d(128, output_dim, 1)
        else:
            self.proj = None

    def forward(
        self,
        images: list[list[str]],
        target_T: int,
        time_series: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        images: list of length B; each element is a list of image paths (len may equal target_T).
        target_T: desired temporal length.
        time_series: (B, 1, T) or (B, T) — actual time in hours per frame for FiLM. If None, use zeros
            (pretrained FiLM was trained with hours; 0–1 frame index would break conditioning).
        Returns: (B, output_dim, target_T)
        """
        if not images:
            raise ValueError("No images provided to VisualEncoderCustom")
        if isinstance(images[0], str):
            images = [images]
        B = len(images)
        from PIL import Image
        from torchvision import transforms

        resize = transforms.Resize((224, 224))
        normalize_tf = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        flat_imgs = []
        flat_indices: list[tuple[int, int]] = []
        for b, paths in enumerate(images):
            for t, p in enumerate(paths):
                if not p:
                    continue
                try:
                    img = Image.open(p)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img = transforms.functional.to_tensor(img)
                    img = resize(img)
                    flat_imgs.append(img)
                    flat_indices.append((b, t))
                except Exception:
                    continue
        if not flat_imgs:
            return torch.zeros(B, self.output_dim, target_T, device=self.device)
        x = torch.stack(flat_imgs, dim=0).to(self.device)
        x = normalize_tf(x)
        # Use actual hours when provided; else zeros (avoid 0–1 frame index which breaks FiLM)
        if time_series is not None:
            ts = time_series.to(self.device)
            if ts.dim() == 3:
                ts = ts.squeeze(1)
            # ts (B, T); for each (b, t) in flat_indices get ts[b, t]
            time_h = torch.tensor(
                [float(ts[b, t].item()) if not math.isnan(ts[b, t].item()) else 0.0 for (b, t) in flat_indices],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            time_h = torch.zeros(len(flat_indices), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            emb, _, _ = self.encoder(x, time_h)
        # emb: (N, 128)
        D = 128
        out = torch.zeros(B, D, target_T, device=self.device)
        for (b, t), v in zip(flat_indices, emb):
            if 0 <= t < target_T:
                out[b, :, t] = v
        if self.proj is not None:
            out = self.proj(out)
        return out


class EmbryoPhase2Diffusion(nn.Module):
    """
    Diffusion over 16 stages conditioned on time + visual features (FEMI or custom encoder).
    Valid mask excludes starting/ending from noise and loss.
    """

    def __init__(
        self,
        time_encoder_output_dim: int,
        decoder_params: dict,
        diffusion_params: dict,
        num_classes: int,
        visual_feature_dim: int,
        device: torch.device,
        *,
        visual_encoder_type: str = "femi",
        femi_model_name: str = "ihlab/FEMI",
        femi_freeze: bool = True,
        custom_encoder_checkpoint: str | Path | None = None,
        fusion_dim: int | None = None,
    ):
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

        # Encoders and fusion (fusion_dim from config; fallback to decoder_params or time_encoder_output_dim)
        if fusion_dim is None:
            fusion_dim = int(decoder_params.get("input_dim", time_encoder_output_dim))
        self.fusion = nn.Conv1d(time_encoder_output_dim + visual_feature_dim, fusion_dim, kernel_size=1)

        self.time_encoder = TimeEncoder(time_encoder_output_dim)
        enc_type = (visual_encoder_type or "femi").lower()
        if enc_type == "custom" and custom_encoder_checkpoint:
            self.visual_encoder = VisualEncoderCustom(
                checkpoint_path=custom_encoder_checkpoint,
                output_dim=visual_feature_dim,
                num_classes=num_classes,
                device=device,
            )
        else:
            self.visual_encoder = VisualEncoderFEMI(
                model_name=femi_model_name,
                proj_dim=visual_feature_dim,
                freeze=femi_freeze,
                device=device,
            )

        decoder_params = dict(decoder_params)
        decoder_params["input_dim"] = fusion_dim
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
        x_start_scaled = (x_start * 2.0 - 1.0) * self.scale
        x_start_scaled = x_start_scaled * valid_mask  # re-zero invalid frames after scaling
        x = self.q_sample(x_start=x_start_scaled, t=t, noise=noise)
        x = torch.clamp(x, min=-self.scale, max=self.scale)
        event_diffused = (x / self.scale + 1.0) / 2.0
        return event_diffused, noise, t

    def _encode_condition(
        self,
        time_series: torch.Tensor,
        image_paths_or_vis: list[list[str]] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode time and visual features and fuse along channels.
        image_paths_or_vis: either list[list[str]] (on-the-fly FEMI) or precomputed (B, D_v, T).
        """
        # Ensure time_series is (B, 1, T) regardless of how DataLoader stacked it
        if time_series.dim() != 3:
            print(f"[Phase2] time_series shape unexpected: {time_series.shape}")
            raise ValueError(f"time_series must be 3D (B, 1, T), got shape {time_series.shape}")
        B, d1, d2 = time_series.shape
        if d1 == 1:
            T = d2
        elif d2 == 1:
            time_series = time_series.transpose(1, 2)
            T = d1
        else:
            print(f"[Phase2] time_series shape ambiguous (neither dim1 nor dim2 == 1): {time_series.shape}")
            raise ValueError(f"Unexpected time_series shape {time_series.shape}; one of the non-batch dims must be 1")

        time_feats = self.time_encoder(time_series)  # (B, D_t, T)
        if isinstance(image_paths_or_vis, torch.Tensor):
            vis_feats = image_paths_or_vis.to(self.device)
            if vis_feats.dim() == 2:
                vis_feats = vis_feats.unsqueeze(0)
            if vis_feats.shape[-1] != T:
                vis_feats = F.interpolate(vis_feats, size=T, mode="linear", align_corners=False)
        else:
            vis_feats = self.visual_encoder(
                images=image_paths_or_vis, target_T=T, time_series=time_series
            )  # (B, D_v, T)

        # Debug: print shapes once to understand mismatch
        if not hasattr(self, "_debug_shapes_printed"):
            print(
                "[Phase2] _encode_condition shapes:",
                f"time_series={tuple(time_series.shape)},",
                f"time_feats={tuple(time_feats.shape)},",
                f"vis_feats={tuple(vis_feats.shape)}",
            )
            self._debug_shapes_printed = True

        fused = torch.cat([time_feats, vis_feats], dim=1)
        fused = self.fusion(fused)
        return fused

    def forward(self, cond_feats: torch.Tensor, t: torch.Tensor, event_diffused: torch.Tensor) -> torch.Tensor:
        return self.decoder(cond_feats, t, event_diffused.float())

    def model_predictions(self, cond_feats: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_m = torch.clamp(x, min=-self.scale, max=self.scale)
        assert x_m.min().item() >= -self.scale - 1e-5 and x_m.max().item() <= self.scale + 1e-5, "x_m out of expected range"
        x_m = denormalize(x_m, self.scale)
        x_start = self.decoder(cond_feats, t, x_m.float())
        x_start = F.softmax(x_start, 1)
        x_start = normalize(x_start, self.scale)
        x_start = torch.clamp(x_start, min=-self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def get_training_loss(
        self,
        time_series: torch.Tensor,
        image_paths: list[list[str]],
        event_gt: torch.Tensor,
        valid_mask: torch.Tensor,
        decoder_ce_criterion: nn.Module,
        decoder_mse_criterion: nn.Module,
        ordinal_loss_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        cond_feats = self._encode_condition(time_series, image_paths)
        event_diffused, noise, t = self.prepare_targets(event_gt, valid_mask)
        event_out = self.forward(cond_feats, t, event_diffused)

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
        decoder_mse_loss = (mse_per * valid_mask.squeeze(1)[:, :-1]).sum(dim=1) / (
            valid_mask[:, :, 1:].sum(dim=(1, 2)).clamp(min=1)
        )
        decoder_mse_loss = decoder_mse_loss.mean()

        out_dict: dict[str, torch.Tensor] = {
            "decoder_ce_loss": decoder_ce_loss,
            "decoder_mse_loss": decoder_mse_loss,
        }
        # Optional ordinal loss on stage index (encodes ordering)
        if ordinal_loss_weight and ordinal_loss_weight > 0.0:
            with torch.no_grad():
                targets_idx = torch.argmax(event_gt, dim=1)  # (B, T)
            probs = F.softmax(event_out, dim=1)  # (B, C, T)
            class_idx = torch.arange(self.num_classes, device=event_out.device, dtype=probs.dtype)
            expected = (probs * class_idx[None, :, None]).sum(dim=1)  # (B, T)
            target_f = targets_idx.float().clamp(min=0)
            ord_loss = F.huber_loss(expected, target_f, reduction="none", delta=2.0)  # (B, T)
            vm = valid_mask.squeeze(1).float()  # (B, T)
            ord_loss = (ord_loss * vm).sum() / vm.sum().clamp(min=1.0)
            out_dict["decoder_ordinal_loss"] = ord_loss

        return out_dict

    @torch.no_grad()
    def ddim_sample(self, time_series: torch.Tensor, image_paths: list[list[str]], valid_mask: torch.Tensor | None = None, seed: int | None = None) -> torch.Tensor:
        cond_feats = self._encode_condition(time_series, image_paths)
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
            import random as pyrandom

            pyrandom.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((B,), time, device=self.device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(cond_feats, x_time, time_cond)
            x_return = x_start.clone()
            if time_next < 0:
                x_time = x_start
                if valid_mask is not None:
                    x_time = x_time * valid_mask
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha_next) / (1 - alpha)).sqrt() * (1 - alpha / alpha_next).sqrt()
            pred = (x_time - (1 - alpha).sqrt() * pred_noise) / alpha.sqrt()
            x_time = alpha_next.sqrt() * pred + (1 - alpha_next - sigma ** 2).sqrt() * pred_noise
            if sigma > 0:
                x_time = x_time + sigma * torch.randn_like(x_time, device=self.device)
            x_time = torch.clamp(x_time, min=-self.scale, max=self.scale)
            if valid_mask is not None:
                x_time = x_time * valid_mask  # re-zero invalid frames each step
        x_return = denormalize(x_return, self.scale)
        if valid_mask is not None:
            x_return = x_return * valid_mask
        return x_return


