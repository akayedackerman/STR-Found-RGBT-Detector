import torch
from mmdet.registry import MODELS
from mmdet.models.backbones import ResNet
import torchvision.models as tv_models


@MODELS.register_module()
class ResNet6Channel(ResNet):
	def __init__(self, in_channels=6, **kwargs):
		# Initialize natively with 6 channels so the parameters are tracked accurately from the start
		super().__init__(in_channels=in_channels, **kwargs)

	def init_weights(self):
		# 1. Let MMEngine initialize the network stages.
		# It will skip conv1.weight due to the shape mismatch but successfully load everything else.
		super().init_weights()

		# 2. Manually load the pretrained weights from the local cache
		try:
			# PyTorch 1.13+ syntax
			ref_model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
		except AttributeError:
			# Older PyTorch fallback syntax
			ref_model = tv_models.resnet50(pretrained=True)

		pretrained_state = ref_model.state_dict()
		pretrained_rgb_weight = pretrained_state['conv1.weight']

		# 3. Perform IN-PLACE modifications to the existing tensor slices.
		# This completely avoids creating new nn.Parameter tracked objects.
		device = self.conv1.weight.device
		dtype = self.conv1.weight.dtype

		# Map pretrained RGB to channels 0, 1, 2
		self.conv1.weight.data[:, :3, :, :] = pretrained_rgb_weight.to(device=device, dtype=dtype)

		# Map the RGB average prior to Thermal channels 3, 4, 5
		rgb_mean_prior = pretrained_rgb_weight.mean(dim=1, keepdim=True)
		self.conv1.weight.data[:, 3:6, :, :] = rgb_mean_prior.to(device=device, dtype=dtype)