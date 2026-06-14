
import torch

from mmdet.registry import MODELS

from mmdet.models.backbones import ResNet

import torchvision.models as tv_models



@MODELS.register_module()

class ResNet6Channel(ResNet):

    def __init__(self, in_channels=6, **kwargs):

        # 1. Initialize with 6 channels natively so parameters are registered correctly

        super().__init__(in_channels=in_channels, **kwargs)



    def init_weights(self):

        # 2. Let MMEngine load everything else from the checkpoint. 

        # It will safely skip conv1.weight due to the 3-vs-6 channel shape mismatch.

        super().init_weights()

        

        # 3. Load the true reference ImageNet tensor from torchvision

        try:

            ref_model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)

        except AttributeError:

            ref_model = tv_models.resnet50(pretrained=True)

            

        pretrained_rgb_weight = ref_model.conv1.weight.data

        

        device = self.conv1.weight.device

        dtype = self.conv1.weight.dtype

        

        # 4. Modify the existing tensor in-place.

        # Slicing directly into .data preserves the parameter object reference completely.

        self.conv1.weight.data[:, :3, :, :] = pretrained_rgb_weight.to(device=device, dtype=dtype)

        

        # 5. Extract the channel mean to use as a prior for Thermal slots 3, 4, 5

        rgb_mean_prior = pretrained_rgb_weight.mean(dim=1, keepdim=True)

        self.conv1.weight.data[:, 3:6, :, :] = rgb_mean_prior.to(device=device, dtype=dtype)

