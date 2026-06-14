
import torch

import torch.nn as nn

from mmdet.registry import MODELS

from mmdet.models.backbones import ResNet

import torchvision.models as tv_models



@MODELS.register_module()

class DualResNet(nn.Module):

    def __init__(self, depth=50, out_indices=(0, 1, 2, 3), frozen_stages=1, style='pytorch', 

                 norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True, **kwargs):

        super().__init__()

        

        # 1. Initialize Independent Visible ResNet Backbone

        self.rgb_backbone = ResNet(

            depth=depth,

            in_channels=3,

            num_stages=4,

            out_indices=out_indices,

            frozen_stages=frozen_stages,

            style=style,

            norm_cfg=norm_cfg,

            norm_eval=norm_eval

        )

        

        # 2. Initialize Independent Thermal ResNet Backbone

        self.thermal_backbone = ResNet(

            depth=depth,

            in_channels=3,

            num_stages=4,

            out_indices=out_indices,

            frozen_stages=frozen_stages,

            style=style,

            norm_cfg=norm_cfg,

            norm_eval=norm_eval

        )

        

        # Output channels double at each FPN stage layer after channel concatenation

        self.out_channels = [256 * 2, 512 * 2, 1024 * 2, 2048 * 2]



    def init_weights(self):

        # Let MMDet run standard random/default initialization first

        self.rgb_backbone.init_weights()

        self.thermal_backbone.init_weights()

        

        # Explicitly load true ImageNet pretrained weights into BOTH backbones

        try:

            ref_model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)

        except AttributeError:

            ref_model = tv_models.resnet50(pretrained=True)

            

        state_dict = ref_model.state_dict()

        backbone_state = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}

        

        # In-place parameter loading to satisfy MMEngine tracking mechanics

        self.rgb_backbone.load_state_dict(backbone_state, strict=False)

        self.thermal_backbone.load_state_dict(backbone_state, strict=False)



    def forward(self, x):

        # Slice channels 0,1,2 for RGB and 3,4,5 for Thermal from the 6-channel pipeline tensor

        rgb_img = x[:, :3, :, :]

        thermal_img = x[:, 3:6, :, :]

        

        rgb_feats = self.rgb_backbone(rgb_img)

        thermal_feats = self.thermal_backbone(thermal_img)

        

        # Mid-Fusion: Channel-wise concatenation at every multi-scale stage layer

        fused_outputs = []

        for r_feat, t_feat in zip(rgb_feats, thermal_feats):

            fused_feat = torch.cat([r_feat, t_feat], dim=1)

            fused_outputs.append(fused_feat)

            

        return tuple(fused_outputs)

