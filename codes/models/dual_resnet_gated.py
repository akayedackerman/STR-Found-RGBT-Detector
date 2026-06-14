import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.backbones import ResNet
import torchvision.models as tv_models


class CrossModalAttentionGate(nn.Module):
	def __init__(self, in_channels, reduction=16):
		super().__init__()
		self.in_channels = in_channels

		# Channel-wise global pooling to compress spatial dimensions into global context vectors
		self.global_pool = nn.AdaptiveAvgPool2d(1)

		# Shared bottleneck gating structure to learn cross-modality scaling properties
		self.gate = nn.Sequential(
			nn.Linear(in_channels, in_channels // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(in_channels // reduction, in_channels, bias=False),
			nn.Sigmoid()
		)

	def forward(self, rgb_feat, thermal_feat):
		b, c, h, w = rgb_feat.size()

		# 1. Compute global channel descriptors
		rgb_pool = self.global_pool(rgb_feat).view(b, c)
		thermal_pool = self.global_pool(thermal_feat).view(b, c)

		# 2. Generate mutual cross-modal attention weights
		# RGB context weights the Thermal feature maps, and vice-versa
		rgb_attn = self.gate(rgb_pool).view(b, c, 1, 1)
		thermal_attn = self.gate(thermal_pool).view(b, c, 1, 1)

		# 3. Apply the modulation gates dynamically
		modulated_rgb = rgb_feat * thermal_attn
		modulated_thermal = thermal_feat * rgb_attn

		# 4. Fuse the modulated feature spaces safely
		return torch.cat([modulated_rgb, modulated_thermal], dim=1)


@MODELS.register_module()
class DualResNet(nn.Module):
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

		# Instantiate an Attention Gate for each feature map resolution stage
		base_channels = [256, 512, 1024, 2048]
		self.attention_gates = nn.ModuleList([
			CrossModalAttentionGate(in_channels=ch) for ch in base_channels
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

		# Map feature stages sequentially through their respective attention gating channels
		fused_outputs = []
		for idx, getattr_mod in enumerate(self.attention_gates):
			fused_feat = getattr_mod(rgb_feats[idx], thermal_feats[idx])
			fused_outputs.append(fused_feat)

		return tuple(fused_outputs)