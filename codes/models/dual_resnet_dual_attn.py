import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.backbones import ResNet
import torchvision.models as tv_models


class CrossModalDualAttentionModule(nn.Module):
	def __init__(self, in_channels, reduction=16):
		super().__init__()
		self.in_channels = in_channels

		# --- 1. Channel Attention Components ---
		self.global_pool = nn.AdaptiveAvgPool2d(1)
		self.channel_gate = nn.Sequential(
			nn.Linear(in_channels, in_channels // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(in_channels // reduction, in_channels, bias=False),
			nn.Sigmoid()
		)

		# --- 2. Tight Spatial Attention Components ---
		self.rgb_proj = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
		self.thermal_proj = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

		# Tight 3x3 kernel explicitly prevents tiny target spatial blurring
		self.spatial_gate = nn.Sequential(
			nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
			nn.Sigmoid()
		)

	def forward(self, rgb_feat, thermal_feat):
		b, c, h, w = rgb_feat.size()

		# Phase A: Compute Global Channel Cross-Attention
		rgb_pool = self.global_pool(rgb_feat).view(b, c)
		thermal_pool = self.global_pool(thermal_feat).view(b, c)

		rgb_ch_attn = self.channel_gate(rgb_pool).view(b, c, 1, 1)
		thermal_ch_attn = self.channel_gate(thermal_pool).view(b, c, 1, 1)

		ch_modulated_rgb = rgb_feat * thermal_ch_attn
		ch_modulated_thermal = thermal_feat * rgb_ch_attn

		# Phase B: Compute Tight Spatial Cross-Attention
		rgb_spatial = self.rgb_proj(ch_modulated_rgb)
		thermal_spatial = self.thermal_proj(ch_modulated_thermal)

		combined_spatial = torch.cat([rgb_spatial, thermal_spatial], dim=1)
		spatial_mask = self.spatial_gate(combined_spatial)

		spatial_modulated_rgb = ch_modulated_rgb * spatial_mask
		spatial_modulated_thermal = ch_modulated_thermal * spatial_mask

		# Phase C: Final Modality Splicing/Concatenation
		return torch.cat([spatial_modulated_rgb, spatial_modulated_thermal], dim=1)


@MODELS.register_module()
class DualResNetDualAttn(nn.Module):
	def __init__(self, depth=50, out_indices=(0, 1, 2, 3), frozen_stages=1, style='pytorch',
	             norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True, **kwargs):
		super().__init__()

		self.rgb_backbone = ResNet(
			depth=depth, in_channels=3, num_stages=4, out_indices=out_indices,
			frozen_stages=frozen_stages, style=style, norm_cfg=norm_cfg, norm_eval=norm_eval
		)

		self.thermal_backbone = ResNet(
			depth=depth, in_channels=3, num_stages=4, out_indices=out_indices,
			frozen_stages=frozen_stages, style=style, norm_cfg=norm_cfg, norm_eval=norm_eval
		)

		base_channels = [256, 512, 1024, 2048]
		self.dual_attention_gates = nn.ModuleList([
			CrossModalDualAttentionModule(in_channels=ch) for ch in base_channels
		])

		self.out_channels = [ch * 2 for ch in base_channels]

	def init_weights(self):
		self.rgb_backbone.init_weights()
		self.thermal_backbone.init_weights()

		try:
			ref_model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
		except AttributeError:
			ref_model = tv_models.resnet50(pretrained=True)

		state_dict = ref_model.state_dict()
		backbone_state = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}

		self.rgb_backbone.load_state_dict(backbone_state, strict=False)
		self.thermal_backbone.load_state_dict(backbone_state, strict=False)

	def forward(self, x):
		rgb_img = x[:, :3, :, :]
		thermal_img = x[:, 3:6, :, :]

		rgb_feats = self.rgb_backbone(rgb_img)
		thermal_feats = self.thermal_backbone(thermal_img)

		fused_outputs = []
		for idx, dual_attn_mod in enumerate(self.dual_attention_gates):
			fused_feat = dual_attn_mod(rgb_feats[idx], thermal_feats[idx])
			fused_outputs.append(fused_feat)

		return tuple(fused_outputs)

