"""UNet model tailored for ML-based tomography reconstruction.

This module implements a configurable UNet that ingests 1-channel
512x512 images (raw reconstructions or sinograms projected into
image space) and predicts the ML-EM refined reconstruction. The
network keeps spatial resolution through padding so it can be
plugged into the existing pipeline without additional resizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MLReconstructionUNet", "build_unet", "UNetConfig"]


def _make_norm(norm_type: str, num_channels: int) -> nn.Module:
	if norm_type == "batch":
		return nn.BatchNorm2d(num_channels)
	if norm_type == "instance":
		return nn.InstanceNorm2d(num_channels, affine=True)
	# Default to group norm for small batch sizes
	num_groups = min(8, num_channels)
	num_groups = max(1, num_groups)
	return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


def _make_activation(act_type: str) -> nn.Module:
	if act_type == "leaky_relu":
		return nn.LeakyReLU(negative_slope=0.1, inplace=True)
	if act_type == "elu":
		return nn.ELU(inplace=True)
	return nn.ReLU(inplace=True)


class ConvBlock(nn.Module):
	"""Two stacked 3x3 convolutions with normalization and activation."""

	def __init__(self, in_channels: int, out_channels: int, norm: str, activation: str) -> None:
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
		self.norm1 = _make_norm(norm, out_channels)
		self.act1 = _make_activation(activation)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
		self.norm2 = _make_norm(norm, out_channels)
		self.act2 = _make_activation(activation)

	def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
		x = self.act1(self.norm1(self.conv1(x)))
		x = self.act2(self.norm2(self.conv2(x)))
		return x


class DownBlock(nn.Module):
	"""ConvBlock followed by 2x2 max pooling."""

	def __init__(self, in_channels: int, out_channels: int, norm: str, activation: str) -> None:
		super().__init__()
		self.block = ConvBlock(in_channels, out_channels, norm, activation)
		self.pool = nn.MaxPool2d(kernel_size=2)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		x = self.block(x)
		pooled = self.pool(x)
		return x, pooled


class UpBlock(nn.Module):
	"""Upsampling via transpose convolution and ConvBlock refinement."""

	def __init__(self, in_channels: int, out_channels: int, norm: str, activation: str) -> None:
		super().__init__()
		self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
		self.block = ConvBlock(in_channels, out_channels, norm, activation)

	def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
		x = self.up(x)
		if x.shape[-2:] != skip.shape[-2:]:
			diff_y = skip.shape[-2] - x.shape[-2]
			diff_x = skip.shape[-1] - x.shape[-1]
			x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
		x = torch.cat([skip, x], dim=1)
		return self.block(x)


@dataclass
class UNetConfig:
	in_channels: int = 1
	out_channels: int = 1
	base_channels: int = 32
	depth: int = 4
	norm: str = "group"
	activation: str = "relu"
	output_activation: Optional[str] = None


class MLReconstructionUNet(nn.Module):
	"""UNet variant for mapping coarse reconstructions to ML-EM quality."""

	def __init__(self, config: Optional[UNetConfig] = None) -> None:
		super().__init__()
		self.config = config or UNetConfig()
		cfg = self.config

		channels = [cfg.base_channels * (2 ** i) for i in range(cfg.depth)]

		self.down_blocks = nn.ModuleList()
		in_ch = cfg.in_channels
		for ch in channels:
			self.down_blocks.append(DownBlock(in_ch, ch, cfg.norm, cfg.activation))
			in_ch = ch

		self.bottleneck = ConvBlock(in_ch, in_ch * 2, cfg.norm, cfg.activation)
		in_ch *= 2

		self.up_blocks = nn.ModuleList()
		for ch in reversed(channels):
			self.up_blocks.append(UpBlock(in_ch, ch, cfg.norm, cfg.activation))
			in_ch = ch

		self.head = nn.Conv2d(in_ch, cfg.out_channels, kernel_size=1)
		self.output_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
		if cfg.output_activation == "sigmoid":
			self.output_activation = torch.sigmoid
		elif cfg.output_activation == "tanh":
			self.output_activation = torch.tanh

		self.apply(self._init_weights)

	@staticmethod
	def _init_weights(module: nn.Module) -> None:
		if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
			nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
			nn.init.ones_(module.weight)
			nn.init.zeros_(module.bias)

	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		if x.shape[1] != self.config.in_channels:
			raise ValueError(f"Expected {self.config.in_channels} input channels, got {x.shape[1]}")

		skips: list[torch.Tensor] = []
		out = x
		for block in self.down_blocks:
			skip, out = block(out)
			skips.append(skip)

		out = self.bottleneck(out)

		for block, skip in zip(self.up_blocks, reversed(skips)):
			out = block(out, skip)

		out = self.head(out)
		if self.output_activation is not None:
			out = self.output_activation(out)
		if mask is not None:
			out = out * mask
		return out


def build_unet(**kwargs: object) -> MLReconstructionUNet:
	"""Factory helper mirroring legacy configs for convenience."""

	config = UNetConfig(**kwargs)
	return MLReconstructionUNet(config=config)