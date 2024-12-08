import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
import os
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from featup.util import unnorm, pca
from featup.model_utils.extractor_sd import load_model
from featup.model_utils.extractor_dino import ViTExtractor
from .modules.projection_network import AggregationNetwork
from featup.layers import ChannelNorm, UnitNorm

class SDDINOFeaturizer(nn.Module):
    def __init__(self, num_patches, diffusion_ver, extractor_name, aggre_net_weights_path, rot_inv=False, num_rotations=4):
        super(SDDINOFeaturizer, self).__init__()
        self.patch_size = 16
        self.num_features = 768
        self.rot_inv = rot_inv
        self.num_rotations = num_rotations
        self.angle = 360 / self.num_rotations
        assert num_patches % 4 == 0, "UNet requires input size to be multiple of 64. Please change num_patches to be multiple of 4."
        self.num_patches = num_patches
        self.sd_model = load_model(img_size=num_patches * 16, diffusion_ver=diffusion_ver, num_timesteps=50) # this sets eval mode
        print("loaded sd model")
        self.extractor_vit = ViTExtractor(extractor_name, stride=14)
        print("loaded vit model")
        self.aggre_net = AggregationNetwork(feature_dims=[640,1280,1280,768], projection_dim=768)
        assert os.path.exists(aggre_net_weights_path), f"AggreNet weights not found at {aggre_net_weights_path}"
        self.aggre_net.load_pretrained_weights(torch.load(aggre_net_weights_path, map_location='cuda'))
        self.mem_eff = False
        self.pca = False # for benchmarking, replace aggrenet with PCA

    def get_features(self, img_batch, apply_aggrenet=True):
        '''
        input:
            image_batch: torch.Tensor [bs, 3, H, W]. [-0.18, 0.26]. Have to be multiple of 64
        return:
            desc: torch.Tensor [bs, 768, H // 16, W // 16]    
        '''
        bs, _, H, W = img_batch.shape
        assert H == W == self.num_patches * self.patch_size
        with torch.no_grad():
            img_batch = unnorm(img_batch) # SD and ViT will normalize separately. This should be [0, 1]
            features_sd = self.sd_model(img_batch, raw=True)
            del features_sd['s2']

            img_dino_input = self.extractor_vit.resize_and_normalize(img_batch, self.num_patches * 14)
            features_dino = self.extractor_vit.extract_descriptors(img_dino_input, layer=11, facet='token')
            features_dino = features_dino.permute(0, 1, 3, 2).reshape(bs, -1, self.num_patches, self.num_patches)

            desc_gathered = torch.cat([
                features_sd['s3'],
                F.interpolate(features_sd['s4'], size=(self.num_patches, self.num_patches), mode='bilinear', align_corners=False),
                F.interpolate(features_sd['s5'], size=(self.num_patches, self.num_patches), mode='bilinear', align_corners=False),
                features_dino], dim=1) # [bs, 640+1280+1280+768, H // 16, W // 16]
            
        if apply_aggrenet:
            desc = self.aggre_net(desc_gathered)
            if os.environ.get("VERBOSE", False):
                print("aggrenet desc norm", desc.norm(dim=1).min(), desc.norm(dim=1).max(), desc.norm(dim=1).mean())
            return desc
        else:
            return desc_gathered

    def get_rotinv_features(self, image):
        '''
        First get rotational invariant features from backbone, then apply aggrenet
        '''
        B, _, H, W = image.shape
        # make batch of rotated images
        images = []
        with torch.no_grad():
            for idx in range(self.num_rotations):
                images.append(rotate(image, idx * self.angle, interpolation=InterpolationMode.BILINEAR))
            feat_rotinv = torch.zeros(B, sum(self.aggre_net.feature_dims), self.num_patches, self.num_patches).to(image)
            if not self.mem_eff:
                features = self.get_features(torch.cat(images, dim=0), apply_aggrenet=False)
                # rotate back
                features = list(features.chunk(self.num_rotations, dim=0))
                for idx in range(self.num_rotations):
                    feat_rotinv += rotate(features[idx], -idx * self.angle, interpolation=InterpolationMode.BILINEAR)
            else:
                for idx in range(self.num_rotations):
                    feature = self.get_features(images[idx], apply_aggrenet=False)
                    feat_rotinv += rotate(feature, -idx * self.angle, interpolation=InterpolationMode.BILINEAR)
            feat_rotinv /= self.num_rotations
        feat_rotinv_aggreneted = self.aggre_net(feat_rotinv)
        return feat_rotinv_aggreneted


    def forward(self, img):
        '''
        img: torch.Tensor [bs, 3, H, W]
        return: torch.Tensor [bs, 768, H // 16, W // 16]
        In the default context, backbones dont have grad, but aggrenet has grad
        '''
        if self.pca:
            features = self.get_features(img, apply_aggrenet=False)
            [features_pca], fit_pca = pca([features], dim=768)
            return features_pca
        
        if self.rot_inv:
            return self.get_rotinv_features(img)
        else:
            return self.get_features(img)