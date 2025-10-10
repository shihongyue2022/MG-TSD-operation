# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
import torch.nn.functional as F


# ---------- Safe wrapper ----------
def _safe_conv1d(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    """
    Guard against PyTorch circular padding error:
    RuntimeError: Padding value causes wrapping around more than once.

    If padding_mode == 'circular' and pad >= T, fall back to replicate pad
    and do a single conv1d with padding=0 to preserve output length.
    """
    # handle int / tuple
    pad = conv.padding if isinstance(conv.padding, int) else conv.padding[0]
    if conv.padding_mode == "circular" and pad >= x.size(-1):
        x = F.pad(x, (pad, pad), mode="replicate")
        return F.conv1d(
            x,
            conv.weight,
            conv.bias,
            stride=conv.stride,
            padding=0,
            dilation=conv.dilation,
            groups=conv.groups,
        )
    return conv(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim: int, proj_dim: int, max_steps: int = 500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x  # [B, proj_dim]

    def _build_embedding(self, dim: int, max_steps: int) -> torch.Tensor:
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)         # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)    # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # [T,2*dim]
        return table


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, residual_channels: int, dilation: int):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            in_channels=1,
            out_channels=2 * residual_channels,
            kernel_size=1,
            padding=2,
            padding_mode="circular",
        )
        self.output_projection = nn.Conv1d(
            residual_channels, 2 * residual_channels, kernel_size=1
        )

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(
        self,
        x: torch.Tensor,
        conditioner: torch.Tensor,
        diffusion_step: torch.Tensor,
    ):
        # diffusion step -> [B, residual_channels, 1]
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)

        # safe conv on conditioner & dilated conv (both can be circular)
        conditioner = _safe_conv1d(self.conditioner_projection, conditioner)
        y = x + diffusion_step
        y = _safe_conv1d(self.dilated_conv, y) + conditioner

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)

        # output projection (not circular by default, but wrapping is harmless)
        y = _safe_conv1d(self.output_projection, y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class CondUpsampler(nn.Module):
    def __init__(self, cond_length: int, target_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(cond_length, target_dim // 2)
        self.linear2 = nn.Linear(target_dim // 2, target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


class EpsilonTheta(nn.Module):
    def __init__(
        self,
        target_dim: int,
        cond_length: int,
        time_emb_dim: int = 16,
        residual_layers: int = 8,
        residual_channels: int = 8,
        dilation_cycle_length: int = 2,
        residual_hidden: int = 64,
    ):
        """
        Denoising Network

        Args:
            target_dim (int): Target dimension, e.g. 1
            cond_length (int): Condition length, e.g. 100
            time_emb_dim (int): Time embedding dim (default 16)
            residual_layers (int): # residual layers (default 8)
            residual_channels (int): residual channels (default 8)
            dilation_cycle_length (int): dilation cycle len (default 2)
            residual_hidden (int): hidden dim for diffusion embedding
        """
        super().__init__()
        # circular 1x1 conv with padding=2 can still trip pad>=T when T is tiny
        self.input_projection = nn.Conv1d(
            in_channels=1,
            out_channels=residual_channels,
            kernel_size=1,
            padding=2,
            padding_mode="circular",
        )
        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden
        )
        self.cond_upsampler = CondUpsampler(
            target_dim=target_dim, cond_length=cond_length
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                )
                for i in range(residual_layers)
            ]
        )
        # keep length by design only if padding set accordingly; these are non-circular
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, kernel_size=3)
        self.output_projection = nn.Conv1d(residual_channels, 1, kernel_size=3)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(
        self,
        inputs: torch.Tensor,      # [B,1,T]
        time: torch.Tensor,        # [B]
        cond: torch.Tensor,        # [B,1,cond_length] before upsample
    ) -> torch.Tensor:
        # entrance conv guarded
        x = _safe_conv1d(self.input_projection, inputs)  # [B,C,T]
        x = F.leaky_relu(x, 0.4)

        diffusion_step = self.diffusion_embedding(time)  # [B,hidden]
        cond_up = self.cond_upsampler(cond)              # [B,1,T]

        skips = []
        for layer in self.residual_layers:
            x, skip = layer(x, cond_up, diffusion_step)
            skips.append(skip)

        x = torch.sum(torch.stack(skips, dim=0), dim=0) / math.sqrt(len(self.residual_layers))  # [B,C,T]

        # these are not circular; wrapping with safe call is no-op unless you later change padding_mode
        x = _safe_conv1d(self.skip_projection, x)
        x = F.leaky_relu(x, 0.4)
        x = _safe_conv1d(self.output_projection, x)
        return x  # [B,1,T]
