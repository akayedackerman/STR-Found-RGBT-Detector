import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.backbones import ResNet
import torchvision.models as tv_models
import math

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


class STITTransformerLayer(nn.Module):
	"""Spatio-Temporal Interaction Transformer (STIT) Layer for fine/medium scales.

	Stabilized with LayerNorm and a zero-initialized residual scaling gate
	to completely prevent numerical gradient explosions during early warmup stages.
	"""

	def __init__(self, embed_dim, patch_stride=2, num_heads=8, num_layers=2, max_tokens=6000):
		super().__init__()
		self.embed_dim = embed_dim
		self.patch_stride = patch_stride

		if patch_stride > 1:
			self.patch_embed = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_stride, stride=patch_stride)
			self.patch_recovery = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=patch_stride,
			                                         stride=patch_stride)
			# Stabilizes learnable upsample channels to clip backprop numerical spikes
			self.norm_recovery = nn.LayerNorm(embed_dim)
		else:
			self.patch_embed = nn.Identity()
			self.patch_recovery = nn.Identity()
			self.norm_recovery = nn.Identity()

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=embed_dim,
			nhead=num_heads,
			dim_feedforward=embed_dim * 4,
			dropout=0.1,
			batch_first=True
		)
		self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		# Scaled variance random distributions for early sequence mapping stability
		self.temporal_embed = nn.Parameter(torch.randn(1, 2, 1, embed_dim) * 0.02)

		# ZERO-INITIALIZED GATING PARAMETER:
		# Forces the block to output a stable identity frame flow initially,
		# completely shielding your loss curve from uninitialized attention shocks.
		self.gamma = nn.Parameter(torch.zeros(1))

	def _get_sinusoidal_pos_embed(self, num_tokens, embed_dim, device):
		"""Generates dynamic 1D sinusoidal positional encodings to support any batch size/grid."""
		position = torch.arange(num_tokens, dtype=torch.float32, device=device).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, embed_dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / embed_dim))

		pos_embed = torch.zeros(1, num_tokens, embed_dim, device=device)
		pos_embed[0, :, 0::2] = torch.sin(position * div_term)
		pos_embed[0, :, 1::2] = torch.cos(position * div_term)
		return pos_embed

	def forward(self, feat_prev, feat_curr):
		b, c, h, w = feat_curr.size()

		# Extract downsampled structural patches
		p_prev = self.patch_embed(feat_prev)
		p_curr = self.patch_embed(feat_curr)

		# Get internal patch dimensions dynamically depending on stride execution
		if self.patch_stride > 1:
			ph, pw = p_curr.size(2), p_curr.size(3)
		else:
			ph, pw = h, w
		num_spatial_tokens = ph * pw

		# Flatten spatial maps and add Temporal Segment identification states
		t_prev = p_prev.flatten(2).transpose(1, 2) + self.temporal_embed[:, 0, :, :]
		t_curr = p_curr.flatten(2).transpose(1, 2) + self.temporal_embed[:, 1, :, :]

		# Combine temporal sequence dimensions along the token axis
		temporal_tokens = torch.cat([t_prev, t_curr], dim=1)

		# Generate position encodings that match tensor sizes perfectly
		pos_enc = self._get_sinusoidal_pos_embed(temporal_tokens.size(1), c, temporal_tokens.device)
		temporal_tokens = temporal_tokens + pos_enc

		# Process tokens through the Spatio-Temporal Transformer Encoder
		transformed = self.temporal_transformer(temporal_tokens)

		# Extract and isolate the current time step (t) tokens (the second half of the token block)
		out_curr_tokens = transformed[:, num_spatial_tokens:, :]

		# Restore original spatial patch structure layout
		out_curr_patches = out_curr_tokens.transpose(1, 2).view(b, c, ph, pw)

		# Learnable Transposed Convolution upsampling to original dimensions
		recovered = self.patch_recovery(out_curr_patches)

		# Normalize channels if using strided downsampling
		if self.patch_stride > 1:
			recovered = recovered.flatten(2).transpose(1, 2)
			recovered = self.norm_recovery(recovered)
			recovered = recovered.transpose(1, 2).view(b, c, h, w)

		# Residual skip-connection controlled by learnable gamma gate
		return feat_curr + self.gamma * recovered


class CrossModalDualAttentionModule(nn.Module):
	"""Core Single-Frame CDAF Multi-Modal Fusion Layer.

	Computes global mutual channel weights and maps tight local spatial boundaries
	using a 3x3 kernel, followed by a channel-preserving additive feature fusion.
	"""

	def __init__(self, in_channels, reduction=16):
		super().__init__()
		self.global_pool = nn.AdaptiveAvgPool2d(1)
		self.channel_gate = nn.Sequential(
			nn.Linear(in_channels, in_channels // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(in_channels // reduction, in_channels, bias=False),
			nn.Sigmoid()
		)
		self.rgb_proj = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
		self.thermal_proj = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
		self.spatial_gate = nn.Sequential(
			nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
			nn.Sigmoid()
		)

	def forward(self, rgb_feat, thermal_feat):
		b, c, h, w = rgb_feat.size()

		# Compute global mutual channel cross-attention
		rgb_ch = self.channel_gate(self.global_pool(rgb_feat).view(b, c)).view(b, c, 1, 1)
		thermal_ch = self.channel_gate(self.global_pool(thermal_feat).view(b, c)).view(b, c, 1, 1)

		m_rgb = rgb_feat * thermal_ch
		m_thermal = thermal_feat * rgb_ch

		# Compute tight, local spatial boundaries using a 3x3 kernel coordinate map
		mask = self.spatial_gate(torch.cat([self.rgb_proj(m_rgb), self.thermal_proj(m_thermal)], dim=1))
		return (m_rgb * mask) + (m_thermal * mask)


@MODELS.register_module()
class DualResNetSTRFound(nn.Module):
	def __init__(self, depth=50, out_indices=(0, 1, 2, 3), frozen_stages=1, style='pytorch', **kwargs):
		super().__init__()
		self.rgb_backbone = ResNet(depth=depth, in_channels=3, num_stages=4, out_indices=out_indices,
		                           frozen_stages=frozen_stages, style='pytorch')
		self.thermal_backbone = ResNet(depth=depth, in_channels=3, num_stages=4, out_indices=out_indices,
		                               frozen_stages=frozen_stages, style='pytorch')

		base_channels = [256, 512, 1024, 2048]
		self.cdaf_layers = nn.ModuleList([CrossModalDualAttentionModule(in_channels=ch) for ch in base_channels])

		# Scale-Adaptive Strides
		self.stit_layers = nn.ModuleList([
			STITTransformerLayer(embed_dim=256, patch_stride=4),
			STITTransformerLayer(embed_dim=512, patch_stride=4),
			STITTransformerLayer(embed_dim=1024, patch_stride=2)
		])

		# UPGRADE 2: Lightweight Temporal Contrastive Learning (TCL) Projection Heads
		self.tcl_projections = nn.ModuleList([
			nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				nn.Flatten(),
				nn.Linear(ch, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 128)
			) for ch in base_channels[:3]
		])

		self.out_channels = base_channels

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
		x_curr = x[:, :6, :, :]
		x_past = x[:, 6:, :, :]

		f_curr = self._extract_cdaf(x_curr)
		f_past = self._extract_cdaf(x_past)

		fused_outputs = []
		tcl_curr_vectors = []
		tcl_past_vectors = []

		for idx in range(3):
			# Extract the raw fused cross-modal features before cross-frame interaction
			tcl_past_vectors.append(self.tcl_projections[idx](f_past[idx]))
			tcl_curr_vectors.append(self.tcl_projections[idx](f_curr[idx]))

			# Process through your verified STIT layers
			fused_layer = self.stit_layers[idx](f_past[idx], f_curr[idx])
			fused_outputs.append(fused_layer)

		fused_outputs.append(f_curr[3])  # C5 Bypass

		# If training, hand over the contrastive vector representations alongside feature maps
		if self.training:
			return tuple(fused_outputs), tcl_curr_vectors, tcl_past_vectors
		return tuple(fused_outputs)

	def _extract_cdaf(self, x_frame):
		rgb_feats = self.rgb_backbone(x_frame[:, :3, :, :])
		thermal_feats = self.thermal_backbone(x_frame[:, 3:6, :, :])
		return [self.cdaf_layers[i](rgb_feats[i], thermal_feats[i]) for i in range(len(rgb_feats))]


# ==============================================================================
# RUNTIME INTERCEPT PATCH HOOK MECHANISM
# ==============================================================================
@HOOKS.register_module(name='TCLFeatureInterceptHook')
class TCLFeatureInterceptHook(Hook):
	"""Runtime execution patch intercept hook for STR-Found TCL Loss extraction.

	Intercepts the backbone output tuple pool mid-flight, separates the FPN
	features from the tracking contrast vectors, computes the custom InfoNCE objective,
	and cleanly adds 'loss_tcl' back to MMDet's master loss dictionary.
	"""

	def __init__(self):
		super().__init__()
		from codes.loss.loss import TemporalContrastiveLoss
		self.tcl_loss_calculator = TemporalContrastiveLoss(temperature=0.07, loss_weight=0.1)

	def before_train_iter(self, runner, batch_idx, data_batch=None):
		"""Patches the detector model forward extraction layer right before execution."""
		detector = runner.model
		if hasattr(detector, 'module'):
			detector = detector.module

		if hasattr(detector, '_original_extract_feat'):
			return

		detector._original_extract_feat = detector.extract_feat

		def patched_extract_feat(batch_inputs):
			backbone_outputs = detector.backbone(batch_inputs)

			if isinstance(backbone_outputs, tuple) and len(backbone_outputs) == 3:
				feats, tcl_curr, tcl_past = backbone_outputs
				loss_tcl = self.tcl_loss_calculator(tcl_curr, tcl_past)
				detector._cached_tcl_loss = loss_tcl
				return detector.neck(feats)
			else:
				return detector.neck(backbone_outputs)

		detector.extract_feat = patched_extract_feat
		detector._original_loss = detector.loss

		def patched_loss(batch_inputs, data_samples):
			detector._cached_tcl_loss = None
			loss_dict = detector._original_loss(batch_inputs, data_samples)

			if getattr(detector, '_cached_tcl_loss', None) is not None:
				loss_dict['loss_tcl'] = detector._cached_tcl_loss

			return loss_dict

		detector.loss = patched_loss