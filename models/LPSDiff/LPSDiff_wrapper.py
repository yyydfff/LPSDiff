
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            (diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2),
        )
        return x2 + x1


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class adjust_net(nn.Module):
    def __init__(self, out_channels=64, middle_channels=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, middle_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(middle_channels, middle_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(middle_channels * 2, middle_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(middle_channels * 4, out_channels * 2, 1, padding=0),
        )

    def forward(self, x):
        out = self.model(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out1 = out[:, :out.shape[1] // 2]
        out2 = out[:, out.shape[1] // 2:]
        return out1, out2


class ResidualBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class LocalPhaseStateEvolution(nn.Module):

    def __init__(self, hidden=32, phase_dim=32, unroll_steps=2, step_scale=0.25):
        super().__init__()
        self.unroll_steps = int(unroll_steps)
        self.step_scale = float(step_scale)

        self.phase_mlp = nn.Sequential(
            SinusoidalPosEmb(phase_dim),
            nn.Linear(phase_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1),
            nn.GELU(),
            ResidualBlock2D(hidden),
            ResidualBlock2D(hidden),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            ResidualBlock2D(hidden),
        )
        self.velocity_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.update_gate_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.local_phase_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.last_diagnostics: Dict[str, torch.Tensor] = {}

    def _normalize_phase(self, phase_index, h, w, dtype):
        if phase_index.dim() != 1:
            phase_index = phase_index.view(phase_index.shape[0])
        denom = float(torch.clamp(phase_index.detach().float().max(), min=1.0).item())
        tau = (phase_index.float() / denom).to(dtype=dtype).view(-1, 1, 1, 1)
        return tau.expand(-1, 1, h, w)

    def _to_single_channel(self, x):
        if x is None:
            return None
        if x.dim() == 5:
            x = x.squeeze(1)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() == 4 and x.size(1) != 1:
            x = x[:, :1]
        return x

    def forward(self, condition, x_obs, phase_index):
        condition = self._to_single_channel(condition)
        x_obs = self._to_single_channel(x_obs)
        if x_obs is None:
            x_obs = condition

        b, _, h, w = condition.shape
        phase_map = self._normalize_phase(phase_index, h, w, condition.dtype)
        residual = condition - x_obs
        base = torch.cat([condition, x_obs, residual, phase_map], dim=1)

        feat = self.stem(base)
        pemb = self.phase_mlp(phase_index.float()).view(b, -1, 1, 1)
        feat = self.fuse(feat + pemb)

        velocity = torch.tanh(self.velocity_head(feat))
        update_gate = torch.sigmoid(self.update_gate_head(feat))
        local_phase = torch.sigmoid(self.local_phase_head(feat))

        evolved = condition
        for _ in range(self.unroll_steps):
            evolved = evolved + self.step_scale * update_gate * velocity

        local_state = (1.0 - local_phase) * condition + local_phase * evolved
        local_state = local_state.clamp(0.0, 1.0)

        self.last_diagnostics = {
            "local_phase_map": local_phase.detach(),
            "velocity_map": velocity.detach(),
            "update_gate_map": update_gate.detach(),
            "local_state": local_state.detach(),
            "local_phase_mean": local_phase.detach().mean(),
            "velocity_mean": velocity.detach().mean(),
            "update_gate_mean": update_gate.detach().mean(),
        }
        return local_state, local_phase


class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.inc = nn.Sequential(single_conv(in_channels, 64), single_conv(64, 64))
        self.down1 = nn.AvgPool2d(2)
        self.mlp1 = nn.Sequential(nn.GELU(), nn.Linear(dim, 64))
        self.adjust1 = adjust_net(64)
        self.conv1 = nn.Sequential(single_conv(64, 128), single_conv(128, 128), single_conv(128, 128))

        self.down2 = nn.AvgPool2d(2)
        self.mlp2 = nn.Sequential(nn.GELU(), nn.Linear(dim, 128))
        self.adjust2 = adjust_net(128)
        self.conv2 = nn.Sequential(
            single_conv(128, 256), single_conv(256, 256), single_conv(256, 256),
            single_conv(256, 256), single_conv(256, 256), single_conv(256, 256),
        )

        self.up1 = up(256)
        self.mlp3 = nn.Sequential(nn.GELU(), nn.Linear(dim, 128))
        self.adjust3 = adjust_net(128)
        self.conv3 = nn.Sequential(single_conv(128, 128), single_conv(128, 128), single_conv(128, 128))

        self.up2 = up(128)
        self.mlp4 = nn.Sequential(nn.GELU(), nn.Linear(dim, 64))
        self.adjust4 = adjust_net(64)
        self.conv4 = nn.Sequential(single_conv(64, 64), single_conv(64, 64))
        self.outc = outconv(64, out_channels)

    def forward(self, x, phase_index, phase_adjust, adjust):
        inx = self.inc(x)
        phase_emb = self.time_mlp(phase_index.float())

        down1 = self.down1(inx)
        condition1 = rearrange(self.mlp1(phase_emb), 'b c -> b c 1 1')
        if adjust:
            gamma1, beta1 = self.adjust1(phase_adjust)
            down1 = down1 + gamma1 * condition1 + beta1
        else:
            down1 = down1 + condition1
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        condition2 = rearrange(self.mlp2(phase_emb), 'b c -> b c 1 1')
        if adjust:
            gamma2, beta2 = self.adjust2(phase_adjust)
            down2 = down2 + gamma2 * condition2 + beta2
        else:
            down2 = down2 + condition2
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        condition3 = rearrange(self.mlp3(phase_emb), 'b c -> b c 1 1')
        if adjust:
            gamma3, beta3 = self.adjust3(phase_adjust)
            up1 = up1 + gamma3 * condition3 + beta3
        else:
            up1 = up1 + condition3
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        condition4 = rearrange(self.mlp4(phase_emb), 'b c -> b c 1 1')
        if adjust:
            gamma4, beta4 = self.adjust4(phase_adjust)
            up2 = up2 + gamma4 * condition4 + beta4
        else:
            up2 = up2 + condition4
        conv4 = self.conv4(up2)
        return self.outc(conv4)


class Network(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, context=True):
        super().__init__()
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels)
        self.context = context
        self.local_phase_evolution = LocalPhaseStateEvolution(
            hidden=32,
            phase_dim=32,
            unroll_steps=2,
            step_scale=0.25,
        )

    def _to_single_channel(self, x):
        if x is None:
            return None
        if x.dim() == 5:
            x = x.squeeze(1)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() == 4 and x.size(1) != 1:
            x = x[:, :1]
        return x

    def forward(self, x, phase_index, condition, x_end, adjust=True):
        if self.context:
            x_middle = x[:, 1].unsqueeze(1)
        else:
            x_middle = self._to_single_channel(x)

        condition = self._to_single_channel(condition)
        x_end = self._to_single_channel(x_end)
        if x_end is None:
            x_end = x_middle
        if condition is None:
            condition = x_middle

        local_state, local_phase = self.local_phase_evolution(condition, x_end, phase_index)
        phase_adjust = torch.cat((local_state, local_phase), dim=1)
        out = self.unet(x, phase_index, phase_adjust, adjust=adjust) + x_middle
        return out

    def export_phase_diagnostics(self):
        return self.local_phase_evolution.last_diagnostics


class WeightNet(nn.Module):
    def __init__(self, weight_num=10):
        super().__init__()
        init = torch.ones([1, weight_num, 1, 1]) / weight_num
        self.weights = nn.Parameter(init)

    def forward(self, x):
        weights = F.softmax(self.weights, 1)
        out = (weights * x).sum(dim=1, keepdim=True)
        return out, weights
