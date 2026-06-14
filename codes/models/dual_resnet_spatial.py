import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.backbones import ResNet
import torchvision.models as tv_models


class SpatialCrossAttentionModule(nn.Module):
	def __init__(self, in_channels):
		super().__init__()

		# Project channels down to a single plane to compute a spatial map
		self.rgb_proj = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
		self.thermal_proj = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

		# Smooth combined attention maps using a local spatial context kernel
		self.spatial_gate = nn.Sequential(
			nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
			nn.Sigmoid()
		)

	def forward(self, rgb_feat, thermal_feat):
		# 1. Compress channel depths into 2D spatial distribution maps
		rgb_spatial = self.rgb_proj(rgb_feat)  # Shape: [B, 1, H, W]
		thermal_spatial = self.thermal_proj(thermal_feat)  # Shape: [B, 1, H, W]

		# 2. Concat spatial maps to compute a mutual cross-modal attention mask
		combined_spatial = torch.cat([rgb_spatial, thermal_spatial], dim=1)  # Shape: [B, 2, H, W]
		attention_mask = self.spatial_gate(combined_spatial)  # Shape: [B, 1, H, W]

		# 3. Multiply mask locally—preserving crisp spatial coordinate points
		modulated_rgb = rgb_feat * attention_mask
		modulated_thermal = thermal_feat * attention_mask

		# 4. Fuse the cross-attended maps cleanly
		return torch.cat([modulated_rgb, modulated_thermal], dim=1)


@MODELS.register_module()
class DualResNetSpatial(nn.Module):
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

		# Instantiate a Spatial Cross-Attention Module for each FPN stage layer
		base_channels = [256, 512, 1024, 2048]
		self.spatial_gates = nn.ModuleList([
			SpatialCrossAttentionModule(in_channels=ch) for ch in base_channels
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
		for idx, spatial_mod in enumerate(self.spatial_gates):
			fused_feat = spatial_mod(rgb_feats[idx], thermal_feats[idx])
			fused_outputs.append(fused_feat)

		return tuple(fused_outputs)