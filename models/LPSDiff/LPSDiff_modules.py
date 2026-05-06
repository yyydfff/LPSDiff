
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    b, *_ = t.shape
    t = t.long().clamp(min=0, max=a.shape[-1] - 1)
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_alpha_schedule(timesteps: int) -> torch.Tensor:
    
    start = 1e-4
    end = 0.02
    return torch.linspace(start, end, timesteps)


def _as_1ch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5:
        x = x.squeeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.dim() == 4 and x.size(1) != 1:
        x = x[:, :1]
    return x

@dataclass
class LatentPhaseState:

    hard_index: torch.Tensor
    phase_index: torch.Tensor
    alpha: torch.Tensor
    phase_curve: torch.Tensor
    assignment: torch.Tensor
    mismatch: torch.Tensor
    phase_loss: torch.Tensor
    losses: Dict[str, torch.Tensor]
    diagnostics: Dict[str, torch.Tensor]



class CondEncoder(nn.Module):


    def __init__(self, in_ch: int = 2, out_ch: int = 1, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObservationStateEncoder(nn.Module):
    """Encode the observed degraded image into a restoration-state token."""

    def __init__(self, in_ch: int = 1, hidden: int = 32, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_ch, hidden, stride=2),
            ConvBlock(hidden, hidden * 2, stride=2),
            ConvBlock(hidden * 2, hidden * 4, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 4, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_1ch(x)
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f"Expected [B,1,H,W], got {tuple(x.shape)}")
        return self.proj(self.encoder(x))


class LatentPhaseStateReparameterizer(nn.Module):


    def __init__(
        self,
        timesteps: int,
        latent_dim: int = 64,
        hidden_dim: int = 64,
        lambda_identity: float = 1.0,
        lambda_phase: float = 1.0,
        lambda_energy: float = 0.5,
        ref_power: float = 1.0,
        soft_temp: float = 0.25,
        align_weight: float = 1.0,
        smooth_weight: float = 0.1,
        identity_weight: float = 0.05,
        mismatch_sparse_weight: float = 1e-3,
        confidence_weight: float = 1e-3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.timesteps = int(timesteps)
        if self.timesteps < 2:
            raise ValueError("timesteps must be >= 2")

        self.lambda_identity = float(lambda_identity)
        self.lambda_phase = float(lambda_phase)
        self.lambda_energy = float(lambda_energy)
        self.ref_power = float(ref_power)
        self.soft_temp = float(soft_temp)

        self.align_weight = float(align_weight)
        self.smooth_weight = float(smooth_weight)
        self.identity_weight = float(identity_weight)
        self.mismatch_sparse_weight = float(mismatch_sparse_weight)
        self.confidence_weight = float(confidence_weight)
        self.eps = float(eps)

        self.obs_encoder = ObservationStateEncoder(in_ch=1, hidden=32, latent_dim=latent_dim)

        self.phase_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.timesteps - 1),
        )
        self.energy_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.timesteps - 1),
        )
        self.mismatch_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.timesteps),
        )

    def _grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.linspace(0.0, 1.0, self.timesteps, device=device, dtype=dtype)

    def _reference_curve(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        u = self._grid(device, dtype)
        return (1.0 - u).pow(self.ref_power)

    def _monotone_curve_from_increments(self, logits: torch.Tensor) -> torch.Tensor:
        delta = F.softplus(logits) + self.eps
        cum = torch.cumsum(delta, dim=1)
        total = cum[:, -1:].clamp_min(self.eps)
        progress = cum / total
        return torch.cat([torch.zeros_like(progress[:, :1]), progress], dim=1)

    def _phase_prior(self, z: torch.Tensor) -> torch.Tensor:
        return self._monotone_curve_from_increments(self.phase_head(z))

    def _latent_energy_curve(self, z: torch.Tensor) -> torch.Tensor:
        progress = self._monotone_curve_from_increments(self.energy_head(z))
        return 1.0 - progress

    def _build_state_space(
        self,
        x_obs: torch.Tensor,
        alpha_schedule: torch.Tensor,
    ):
        x_obs = _as_1ch(x_obs)
        z = self.obs_encoder(x_obs)
        dtype = x_obs.dtype
        device = x_obs.device

        grid = self._grid(device, dtype)                         # [T]
        ref = self._reference_curve(device, dtype)                # [T]
        nominal = grid.view(1, self.timesteps, 1)                 # [1,T,1]
        candidate = grid.view(1, 1, self.timesteps)               # [1,1,T]

        phase_prior = self._phase_prior(z)                        # [B,T]
        latent_energy = self._latent_energy_curve(z)              # [B,T]
        mismatch = torch.sigmoid(self.mismatch_head(z))           # [B,T]

        identity_cost = (candidate - nominal).pow(2)
        phase_cost = (candidate - phase_prior.unsqueeze(-1)).pow(2)
        energy_cost = (latent_energy.unsqueeze(1) - ref.view(1, -1, 1)).pow(2)

        mismatch_expanded = mismatch.unsqueeze(-1)
        cost = (
            self.lambda_identity * (1.0 - mismatch_expanded) * identity_cost
            + self.lambda_phase * phase_cost
            + self.lambda_energy * (1.0 + mismatch_expanded) * energy_cost
        )

        assignment = torch.softmax(-cost / max(self.soft_temp, self.eps), dim=-1)   # [B,T,T]
        phase_grid = torch.arange(self.timesteps, device=device, dtype=dtype)
        soft_phase_index_all = (assignment * phase_grid.view(1, 1, -1)).sum(dim=-1)  # [B,T]
        soft_phase_all = (assignment * grid.view(1, 1, -1)).sum(dim=-1)              # [B,T]
        soft_energy = (assignment * latent_energy.unsqueeze(1)).sum(dim=-1)          # [B,T]

        alpha = alpha_schedule.to(device=device, dtype=dtype)
        soft_alpha_all = (assignment * alpha.view(1, 1, -1)).sum(dim=-1)             # [B,T]

        hard_curve = torch.argmin(cost, dim=-1).long()                              # [B,T]
        hard_curve = torch.cummax(hard_curve, dim=1).values
        hard_curve[:, 0] = 0
        hard_curve[:, -1] = self.timesteps - 1
        hard_curve = hard_curve.clamp_(0, self.timesteps - 1)
        phase_curve = hard_curve.float() / float(self.timesteps - 1)

        align_loss = ((1.0 + mismatch) * (soft_energy - ref.view(1, -1)).pow(2)).mean()
        target_step = 1.0 / float(self.timesteps - 1)
        smooth_loss = (soft_phase_all[:, 1:] - soft_phase_all[:, :-1] - target_step).pow(2).mean()
        monotone_loss = F.relu(soft_phase_all[:, :-1] - soft_phase_all[:, 1:]).pow(2).mean()
        smooth_loss = smooth_loss + 0.5 * monotone_loss
        identity_anchor_loss = ((1.0 - mismatch) * (soft_phase_all - grid.view(1, -1)).pow(2)).mean()
        mismatch_sparse_loss = mismatch.mean()
        confidence_loss = (assignment * (1.0 - assignment)).mean()

        phase_loss = (
            self.align_weight * align_loss
            + self.smooth_weight * smooth_loss
            + self.identity_weight * identity_anchor_loss
            + self.mismatch_sparse_weight * mismatch_sparse_loss
            + self.confidence_weight * confidence_loss
        )

        losses = {
            "align_loss": align_loss,
            "smooth_loss": smooth_loss,
            "identity_anchor_loss": identity_anchor_loss,
            "mismatch_sparse_loss": mismatch_sparse_loss,
            "confidence_loss": confidence_loss,
        }
        diagnostics = {
            "latent_token": z.detach(),
            "phase_prior": phase_prior.detach(),
            "latent_energy": latent_energy.detach(),
            "mismatch": mismatch.detach(),
            "hard_curve": hard_curve.detach(),
            "phase_curve": phase_curve.detach(),
            "soft_phase_curve": soft_phase_all.detach(),
            "soft_phase_index_curve": soft_phase_index_all.detach(),
            "soft_alpha_curve": soft_alpha_all.detach(),
            "assignment_entropy": (-(assignment * (assignment + self.eps).log()).sum(dim=-1)).detach(),
        }
        return hard_curve, assignment, soft_phase_index_all, soft_alpha_all, phase_curve, mismatch, phase_loss, losses, diagnostics

    def estimate(self, nominal_index: torch.Tensor, x_obs: torch.Tensor, alpha_schedule: torch.Tensor) -> LatentPhaseState:
        if nominal_index.dim() != 1:
            nominal_index = nominal_index.view(nominal_index.shape[0])
        hard_curve, assignment, soft_phase_index_all, soft_alpha_all, phase_curve, mismatch, phase_loss, losses, diagnostics = \
            self._build_state_space(x_obs, alpha_schedule)

        t_safe = nominal_index.long().clamp(0, self.timesteps - 1).view(-1, 1)
        hard_index = torch.gather(hard_curve, 1, t_safe).squeeze(1).long()
        phase_index = torch.gather(soft_phase_index_all, 1, t_safe).squeeze(1)
        alpha = torch.gather(soft_alpha_all, 1, t_safe).squeeze(1)
        mismatch_at_t = torch.gather(mismatch, 1, t_safe).squeeze(1)

        diagnostics.update({
            "nominal_index": nominal_index.detach(),
            "hard_index_at_t": hard_index.detach(),
            "phase_index_at_t": phase_index.detach(),
            "alpha_at_t": alpha.detach(),
            "mismatch_at_t": mismatch_at_t.detach(),
        })
        return LatentPhaseState(
            hard_index=hard_index,
            phase_index=phase_index,
            alpha=alpha,
            phase_curve=phase_curve,
            assignment=assignment,
            mismatch=mismatch,
            phase_loss=phase_loss,
            losses=losses,
            diagnostics=diagnostics,
        )



class LatentPhaseStateDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: Optional[nn.Module] = None,
        image_size: int = 512,
        channels: int = 1,
        timesteps: int = 10,
        context: bool = True,
        use_latent_phase_state: bool = True,
        self_condition_prob: float = 0.5,
        phase_lambda_identity: float = 1.0,
        phase_lambda_phase: float = 1.0,
        phase_lambda_energy: float = 0.5,
        phase_ref_power: float = 1.0,
        phase_soft_temp: float = 0.25,
        phase_loss_weight: float = 0.1,
        phase_align_weight: float = 1.0,
        phase_smooth_weight: float = 0.1,
        phase_identity_weight: float = 0.05,
        phase_mismatch_sparse_weight: float = 1e-3,
        phase_confidence_weight: float = 1e-3,
        phase_transition_weight: float = 0.05,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.num_timesteps = int(timesteps)
        self.context = bool(context)
        self.self_condition_prob = float(self_condition_prob)
        self.use_latent_phase_state = bool(use_latent_phase_state)
        self.phase_loss_weight = float(phase_loss_weight)
        self.phase_transition_weight = float(phase_transition_weight)

        alphas_cumprod = linear_alpha_schedule(timesteps)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("one_minus_alphas_cumprod", 1.0 - alphas_cumprod)

        self.cond_encoder = CondEncoder(in_ch=2, out_ch=1, hidden=16)

        if self.use_latent_phase_state:
            self.phase_reparameterizer = LatentPhaseStateReparameterizer(
                timesteps=timesteps,
                lambda_identity=phase_lambda_identity,
                lambda_phase=phase_lambda_phase,
                lambda_energy=phase_lambda_energy,
                ref_power=phase_ref_power,
                soft_temp=phase_soft_temp,
                align_weight=phase_align_weight,
                smooth_weight=phase_smooth_weight,
                identity_weight=phase_identity_weight,
                mismatch_sparse_weight=phase_mismatch_sparse_weight,
                confidence_weight=phase_confidence_weight,
            )
        else:
            self.phase_reparameterizer = None

        self.last_phase_state: Optional[LatentPhaseState] = None
        self.last_prev_phase_state: Optional[LatentPhaseState] = None
        self.last_phase_loss: Optional[torch.Tensor] = None
        self.last_phase_stats: Dict[str, float] = {}
        self.last_transition_loss: Optional[torch.Tensor] = None

    def q_sample(self, x_start: torch.Tensor, x_end: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.one_minus_alphas_cumprod, t, x_start.shape) * x_end
        )

    def q_sample_by_alpha(self, x_start: torch.Tensor, x_end: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if alpha.dim() == 1:
            alpha = alpha.view(-1, *((1,) * (x_start.dim() - 1)))
        return alpha * x_start + (1.0 - alpha) * x_end

    def get_x2_bar_from_xt_by_alpha(self, x1_bar: torch.Tensor, xt: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if alpha.dim() == 1:
            alpha = alpha.view(-1, *((1,) * (x1_bar.dim() - 1)))
        return (xt - alpha * x1_bar) / (1.0 - alpha).clamp_min(1e-8)

    def get_x2_bar_from_xt(self, x1_bar: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (
            (xt - extract(self.alphas_cumprod, t, x1_bar.shape) * x1_bar)
            / extract(self.one_minus_alphas_cumprod, t, x1_bar.shape).clamp_min(1e-8)
        )

    def _build_cond(self, y: torch.Tensor, sc: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = _as_1ch(y)
        if sc is None:
            sc = torch.zeros_like(y)
        else:
            sc = _as_1ch(sc)
        return self.cond_encoder(torch.cat([y, sc], dim=1))

    def _estimate_phase_state(
        self,
        nominal_index: torch.Tensor,
        x_obs: torch.Tensor,
        update_last: bool = True,
    ) -> Optional[LatentPhaseState]:
        if not self.use_latent_phase_state or self.phase_reparameterizer is None:
            if update_last:
                self.last_phase_state = None
                self.last_phase_loss = None
                self.last_phase_stats = {}
            return None

        state = self.phase_reparameterizer.estimate(nominal_index, x_obs, self.alphas_cumprod)
        if update_last:
            self.last_phase_state = state
            self.last_phase_loss = self.phase_loss_weight * state.phase_loss
            self.last_phase_stats = {
                "phase_loss": float(state.phase_loss.detach()),
                "hard_index_mean": float(state.hard_index.float().mean().detach()),
                "phase_index_mean": float(state.phase_index.float().mean().detach()),
                "alpha_mean": float(state.alpha.float().mean().detach()),
                "mismatch_mean": float(state.mismatch.float().mean().detach()),
                "mismatch_at_t_mean": float(state.diagnostics["mismatch_at_t"].float().mean().detach()),
            }
            for key, val in state.losses.items():
                self.last_phase_stats[key] = float(val.detach())
        return state

    def _phase_transition_consistency(
        self,
        current_state: Optional[LatentPhaseState],
        prev_state: Optional[LatentPhaseState],
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if current_state is None or prev_state is None:
            if self.alphas_cumprod.device is not None:
                return torch.zeros((), device=self.alphas_cumprod.device)
        gap = current_state.phase_index - prev_state.phase_index
        target_gap = torch.ones_like(gap)
        loss = (gap - target_gap).pow(2)
        if valid_mask is not None:
            valid_mask = valid_mask.float().view_as(loss)
            loss = loss * valid_mask
            denom = valid_mask.sum().clamp_min(1.0)
            return loss.sum() / denom
        return loss.mean()

    # ------------------------ inference ------------------------
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 4,
        img: Optional[torch.Tensor] = None,
        t: Optional[int] = None,
        sampling_routine: str = "ddim",
        n_iter: int = 1,
        start_adjust_iter: int = 1,
        loss_distribution=None,
    ):
        self.denoise_fn.eval()
        self.last_phase_loss = None
        self.last_phase_stats = {}
        self.last_phase_state = None
        self.last_prev_phase_state = None
        self.last_transition_loss = None

        if img is None:
            raise ValueError("sample requires img")
        if t is None:
            t = self.num_timesteps
        if sampling_routine != "ddim":
            raise NotImplementedError(f"sampling_routine={sampling_routine} is not implemented")

        if self.context:
            up_img = img[:, 0].unsqueeze(1)
            down_img = img[:, 2].unsqueeze(1)
            img_mid = img[:, 1].unsqueeze(1)
            img = img_mid
        else:
            img_mid = _as_1ch(img)
            img = img_mid
            up_img, down_img = None, None

        direct_recons = []
        imstep_imgs = []

        sc = None
        if self.self_condition_prob > 0:
            try:
                t0 = torch.zeros((batch_size,), dtype=torch.float32, device=img.device)
                y0 = self._build_cond(img_mid, sc=None)
                x0 = torch.cat((up_img, img_mid, down_img), dim=1) if self.context else img_mid
                sc = self.denoise_fn(x0, t0, y0, img_mid, adjust=False).detach()
            except Exception:
                sc = None

        while t:
            nominal = torch.full((batch_size,), t - 1, device=img.device, dtype=torch.long)
            current_state = self._estimate_phase_state(nominal, img_mid, update_last=True)

            if current_state is not None:
                phase_cond = current_state.phase_index
                current_alpha = current_state.alpha
            else:
                phase_cond = nominal.float()
                current_alpha = extract(self.alphas_cumprod, nominal, img.shape).view(batch_size)

            if self.context:
                full_img = torch.cat((up_img, img, down_img), dim=1)
                x_end = img
            else:
                full_img = img
                x_end = img

            y_enc = self._build_cond(img_mid, sc=sc)
            adjust = False if (t == self.num_timesteps or n_iter < start_adjust_iter) else True


            x1_bar = self.denoise_fn(full_img, phase_cond.float(), y_enc, x_end, adjust=adjust)
            x2_bar = self.get_x2_bar_from_xt_by_alpha(x1_bar, img, current_alpha)
            xt_phi = self.q_sample_by_alpha(x1_bar, x2_bar, current_alpha)

            prev_nominal = torch.clamp(nominal - 1, min=0)
            prev_state = self._estimate_phase_state(prev_nominal, img_mid, update_last=False)
            self.last_prev_phase_state = prev_state
            if t - 1 != 0:
                prev_alpha = prev_state.alpha if prev_state is not None else extract(self.alphas_cumprod, prev_nominal, img.shape).view(batch_size)
                xt_prev_phi = self.q_sample_by_alpha(x1_bar, x2_bar, prev_alpha)
            else:
                xt_prev_phi = x1_bar

            img = img - xt_phi + xt_prev_phi
            direct_recons.append(x1_bar)
            imstep_imgs.append(img)
            t -= 1

        return img.clamp(0.0, 1.0), torch.stack(direct_recons), torch.stack(imstep_imgs)


    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_iter: int,
        only_adjust_two_step: bool = False,
        start_adjust_iter: int = 1,
    ):
        self.last_phase_loss = None
        self.last_phase_stats = {}
        self.last_phase_state = None
        self.last_prev_phase_state = None
        self.last_transition_loss = None

        y = _as_1ch(y)
        b, c, h, w = y.shape
        device = y.device
        assert h == self.image_size and w == self.image_size, f"height and width must be {self.image_size}"

        t_single = torch.randint(0, self.num_timesteps, (1,), device=device).long()
        nominal_t = t_single.repeat((b,))

        if self.context:
            x_end = x[:, 1].unsqueeze(1)
        else:
            x_end = _as_1ch(x)

        current_state = self._estimate_phase_state(nominal_t, x_end, update_last=True)

        if current_state is not None and self.training:
            # Forward restoration state indexed by latent phi, not nominal t.
            x_mix_mid_or_full = self.q_sample_by_alpha(y, x_end, current_state.alpha)
            phase_cond = current_state.phase_index
        else:
            hard_t = current_state.hard_index if current_state is not None else nominal_t
            x_mix_mid_or_full = self.q_sample(y, x_end, hard_t)
            phase_cond = hard_t.float()

        if self.context:
            x_mix = torch.cat((x[:, 0].unsqueeze(1), x_mix_mid_or_full, x[:, 2].unsqueeze(1)), dim=1)
            y_mid = x_end
        else:
            x_mix = x_mix_mid_or_full
            y_mid = x_end

        sc = None
        if torch.rand(1, device=device).item() < self.self_condition_prob:
            with torch.no_grad():
                t0 = torch.zeros((b,), dtype=torch.float32, device=device)
                cond0 = self._build_cond(y_mid, sc=None)
                sc = self.denoise_fn(x_mix, t0, cond0, x_end, adjust=False).detach()

        y_enc = self._build_cond(y_mid, sc=sc)
        if only_adjust_two_step or n_iter < start_adjust_iter:
            x_recon = self.denoise_fn(x_mix, phase_cond.float(), y_enc, x_end, adjust=False)
        else:
            adjust = False if t_single.item() == self.num_timesteps - 1 else True
            x_recon = self.denoise_fn(x_mix, phase_cond.float(), y_enc, x_end, adjust=adjust)


        if n_iter >= start_adjust_iter and t_single.item() >= 1:
            prev_nominal_t = (nominal_t - 1).clamp(min=0)
            prev_state = self._estimate_phase_state(prev_nominal_t, x_end, update_last=False)
            self.last_prev_phase_state = prev_state

            if prev_state is not None and self.training:
                x_mix_sub1_mid_or_full = self.q_sample_by_alpha(x_recon, x_end, prev_state.alpha)
                prev_phase_cond = prev_state.phase_index
                transition_loss = self._phase_transition_consistency(
                    current_state,
                    prev_state,
                    valid_mask=(nominal_t > 0),
                )
                self.last_transition_loss = transition_loss
                if isinstance(self.last_phase_loss, torch.Tensor):
                    self.last_phase_loss = self.last_phase_loss + self.phase_loss_weight * self.phase_transition_weight * transition_loss
                else:
                    self.last_phase_loss = self.phase_loss_weight * self.phase_transition_weight * transition_loss
                self.last_phase_stats["phase_transition_loss"] = float(transition_loss.detach())
            else:
                hard_t = current_state.hard_index if current_state is not None else nominal_t
                t_sub1 = (hard_t - 1).clamp(min=0)
                x_mix_sub1_mid_or_full = self.q_sample(x_recon, x_end, t_sub1)
                prev_phase_cond = t_sub1.float()

            if self.context:
                x_mix_sub1 = torch.cat((x[:, 0].unsqueeze(1), x_mix_sub1_mid_or_full, x[:, 2].unsqueeze(1)), dim=1)
            else:
                x_mix_sub1 = x_mix_sub1_mid_or_full
            x_recon_sub1 = self.denoise_fn(x_mix_sub1, prev_phase_cond.float(), x_recon, x_end, adjust=True)
        else:
            x_recon_sub1, x_mix_sub1 = x_recon, x_mix

        return x_recon, x_mix, x_recon_sub1, x_mix_sub1

    def export_phase_diagnostics(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if self.last_phase_state is not None:
            for k, v in self.last_phase_state.diagnostics.items():
                if isinstance(v, torch.Tensor):
                    out[f"state_{k}"] = v.detach()
        if self.last_prev_phase_state is not None:
            for k, v in self.last_prev_phase_state.diagnostics.items():
                if isinstance(v, torch.Tensor):
                    out[f"prev_state_{k}"] = v.detach()
        if hasattr(self.denoise_fn, "export_phase_diagnostics"):
            local = self.denoise_fn.export_phase_diagnostics()
            for k, v in local.items():
                if isinstance(v, torch.Tensor):
                    out[f"local_{k}"] = v.detach()
        return out



PhaseAsynchronousDiffusion = LatentPhaseStateDiffusion
Diffusion = LatentPhaseStateDiffusion
