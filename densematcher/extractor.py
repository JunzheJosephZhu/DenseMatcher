import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from dift.models.dift_sd import SDFeaturizer
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
from .featurizers.SDDINO import SDDINOFeaturizer
from functools import partial
from .featurizers.util import resize
from featup.layers import ChannelNorm, UnitNorm
from featup.upsamplers import get_upsampler
import numpy as np
import re
import os
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class RotationInvariantExtractor(nn.Module):
    def __init__(self, num_rotations):
        super(RotationInvariantExtractor, self).__init__()
        self.num_rotations = num_rotations
        self.angle = 360 / num_rotations

    def forward(self, image, **kwargs):
        '''
        Base funtion for just running the extractor, without rotating images
        '''
        raise NotImplementedError

    def preprocess(self, image, smaller_edge_size):
        raise NotImplementedError
    
    @torch.no_grad()
    def get_rotinv_features(self, image, mask=None, **kwargs):
        '''
        Rotate images and do forward and rotate back
        args: image: (B, 3, H, W)
        return: features: (B, C, H // patch_size, W // patch_size)
        '''
        B, _, H, W = image.shape
        images = [] 
        for idx in range(self.num_rotations):
            images.append(rotate(image, idx * self.angle, interpolation=InterpolationMode.BILINEAR))
        images = torch.cat(images, 0)
        features = self(images, **kwargs) # [num_rotations * B, C, H // patch_size, W // patch_size]
        features = features.reshape(self.num_rotations, B, features.shape[-3], H // self.patch_size, W // self.patch_size)
        for idx in range(self.num_rotations):
            features[idx] = rotate(features[idx], -idx * self.angle, interpolation=InterpolationMode.BILINEAR)
        return features.mean(dim=0)
    
class DINOv2FeatureExtractor(RotationInvariantExtractor):
    def __init__(self, model_name, weights=None, checkpoint_key=None, num_rotations=4):
        super(DINOv2FeatureExtractor, self).__init__(num_rotations)
        self.backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=model_name).to("cuda")
        if weights is not None:
            self.load_pretrained_weights(self.backbone, weights, checkpoint_key)
        self.patch_size = self.backbone.patch_size

    @torch.no_grad()
    def forward(self, image):
        '''
        args: image: (B, 3, H, W)
        return: features: (B, C, H // patch_size, W // patch_size)
        '''
        grid_size = (image.shape[-2] // self.patch_size, image.shape[-1] // self.patch_size) # h x w (TODO: check)
        features = self.backbone.get_intermediate_layers(image)[0] # [1, H * W, C]
        return features.view(features.shape[0], grid_size[0], grid_size[1], features.shape[2]).permute(0, 3, 1, 2)

    @staticmethod
    def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # interpolate pos_embed
        source_num_patches = state_dict["pos_embed"].size(1) - 1
        source_h = source_w = int(source_num_patches ** 0.5)
        target_num_patches = model.patch_embed.num_patches
        target_h = target_w = int(target_num_patches ** 0.5)
        dim = state_dict["pos_embed"].shape[-1]
        class_pos_embed = state_dict["pos_embed"][:, :1]
        patch_pos_embed = F.interpolate(
            state_dict["pos_embed"][:, 1:].reshape(1, source_h, source_w, -1).permute(0, 3, 1, 2),
            size=(target_h, target_w),
            mode="bicubic",
            antialias=model.interpolate_antialias,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        state_dict["pos_embed"] = torch.cat((class_pos_embed, patch_pos_embed), dim=1)
        # load weights, TODO: make sure missing_keys == []
        msg = model.load_state_dict(state_dict, strict=False)
        print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))
    
    def preprocess(self, image: Image,
                  smaller_edge_size: float,
        ) -> torch.Tensor:
        '''
        Prepare image for DINOv2 extractor
        return: image: torch.Tensor [C, H, W] mask: torch.Tensor [1, H, W]
        '''
        smaller_edge_size = int(smaller_edge_size)
        interpolation_mode = transforms.InterpolationMode.BICUBIC

        transform = transforms.Resize(size=smaller_edge_size, interpolation=interpolation_mode, antialias=True)
        transform2 = transforms.Compose([  
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
        resized_image = transform(image)
        image_tensor = transform2(resized_image)
        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % self.patch_size, height - height % self.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]
        return image_tensor, torch.ones_like(image_tensor).mean(0, keepdim=True), resized_image
    
class DIFTFeatureExtractor(RotationInvariantExtractor):
    def __init__(self, ensemble_size, prompt, num_rotations=4):
        super(DIFTFeatureExtractor, self).__init__(num_rotations)
        self.backbone = SDFeaturizer()
        self.patch_size = 16
        self.ensemble_size = ensemble_size
        self.prompt = prompt
        
    @torch.no_grad()
    def forward(self, image, t=261):
        '''
        args: image: (B, 3, H, W)
        return: features: (B, C, H // patch_size, W // patch_size)
        '''
        features = []
        for idx in range(image.shape[0]):
            feature = self.backbone.forward(image[idx],
                                prompt=self.prompt,
                                ensemble_size=self.ensemble_size,
                                t=t)
            features.append(feature)
        features = torch.stack(features, dim=0)
        assert image.shape[-2] // self.patch_size == features.shape[-2]
        assert image.shape[-1] // self.patch_size == features.shape[-1]
        return features
    
    def preprocess(self, image: Image,
                smaller_edge_size: float,
        ) -> torch.Tensor:
        '''
        Prepare image for SD feature extractor. scale image to [-1, 1]
        return: image: torch.Tensor [C, H, W] mask: torch.Tensor [1, H, W]
        '''
        smaller_edge_size = int(smaller_edge_size)
        interpolation_mode = transforms.InterpolationMode.BICUBIC
        transform = transforms.Resize(size=smaller_edge_size, interpolation=interpolation_mode, antialias=True)
        transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # scale from [0, 1] to [-1, 1]
        ])
        resized_image = transform(image)
        image_tensor = transform2(resized_image)

        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % self.patch_size, height - height % self.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]
        return image_tensor, torch.ones_like(image_tensor).mean(0, keepdim=True), resized_image


class SDDINOFeatureExtractor(RotationInvariantExtractor):
    def __init__(self, pretrained_upsampler_path, aggre_net_weights_folder, num_rotations=4):
        '''
        pretrained_upsampler_path decides imsize & num_patches
        in odise clip.py and helper.py state_dict is an empty dict for clip and ldm
        '''
        super(SDDINOFeatureExtractor, self).__init__(num_rotations)
        self.patch_size = 16
        imgsize_match = re.search(r'imsize=(\d+)', pretrained_upsampler_path)
        channelnorm_match = re.search(r'channelnorm=(True|False)', pretrained_upsampler_path)
        unitnorm_match = re.search(r'unitnorm=(True|False)', pretrained_upsampler_path)
        rotinv_match = re.search(r'rotinv=(True|False)', pretrained_upsampler_path)
        self.imsize = int(imgsize_match.group(1))
        self.use_channelnorm = channelnorm_match.group(1) == 'True'
        self.use_unitnorm = unitnorm_match.group(1) == 'True'    
        # if rot_inv = True, then aggrenet will take rotational invariant features as input
        self.rot_inv = rotinv_match.group(1) == 'True' if rotinv_match is not None else False
        assert self.imsize % self.patch_size == 0
        self.num_patches = self.imsize // self.patch_size
        print("imsize", self.imsize, "Using channelnorm:", self.use_channelnorm, "Using unitnorm:", self.use_unitnorm, "Rot inv:", self.rot_inv)
        self.featurizer = SDDINOFeaturizer(num_patches=self.num_patches, 
                                           diffusion_ver='v1-5', 
                                           extractor_name='dinov2_vitb14', 
                                           aggre_net_weights_path=f'{aggre_net_weights_folder}/best_{self.num_patches * self.patch_size}.PTH', 
                                           rot_inv=self.rot_inv, 
                                           num_rotations=num_rotations)
        assert self.featurizer.patch_size == self.patch_size
        self.num_features = self.featurizer.num_features

        self.upsampler = get_upsampler("jbu_stack", self.num_features)
        pretrained_dict = torch.load(pretrained_upsampler_path, map_location="cpu")
        upsampler_state_dict = {}
        for k, v in pretrained_dict['state_dict'].items():
            if k.startswith('upsampler'):
                upsampler_state_dict[k.removeprefix("upsampler.")] = v
        self.upsampler.load_state_dict(upsampler_state_dict)
        num_upsampler_params = 0
        for param in self.upsampler.parameters():
            num_upsampler_params += param.numel()
        print(f"Featup has {num_upsampler_params} parameters.")


    def preprocess(self, image: Image.Image,
                   smaller_edge_size: float = None,
        ) -> torch.Tensor:
        '''
        Resize image while keeping aspect ratio.
        return: image: torch.Tensor [3, H, W] mask: torch.Tensor [1, H, W], pil_image_resized: PIL.
        '''
        assert isinstance(image, Image.Image), "input has to be pillow image"
        W, H = image.size
        pil_mask = resize(Image.fromarray(np.ones((H, W, 3), dtype=np.uint8) * 255), target_res=self.num_patches * 16, resize=True, to_pil=True)
        pil_image_resized = resize(image.convert('RGB'), target_res=self.num_patches * 16, resize=True, to_pil=True)
        image = T.Compose([
                    T.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ])(pil_image_resized) # [3, H, W]
        mask = T.Compose([
                    T.ToTensor(),
                ])(pil_mask).mean(dim=0, keepdim=True) # [1, H, W]
        return image, mask, pil_image_resized
        
    def forward(self, image, mask=None, return_everything=False):
        '''
        Do forwarding using the featurizer. If self.rot_inv = True, then the returned features would be rotation invariant
        args: image: (B, 3, H, W) mask: (B, 1, H, W)
        return: features_hr: (B, C, H, W)
        '''
        B, _, H, W = image.shape
        features_lr = self.featurizer(image)
        if self.use_unitnorm:
            features_lr = UnitNorm(self.num_features)(features_lr)
        if self.use_channelnorm:
            features_lr = ChannelNorm(self.num_features)(features_lr)
        features_hr = self.upsampler(features_lr, image)        
        if return_everything:
            return features_hr, features_lr 
        return features_hr

    @torch.no_grad()
    @torch.autocast("cuda")
    def get_rotinv_features_legacy(self, image, mask, return_everything=False):
        '''
        Legacy function. Get rotation invariant features by rotating image, extracting features, doing featup, then rotating back and averaging
        args:
            args: image: (B, 3, H, W) mask: (B, 1, H, W)
            return: feat_rotinv: (B, C, H, W)
                    features_hr: list of (B, C, H, W), all unrotated
                    features_lr: list of (num_rotations, B, C, H // patch_size, W // patch_size), each rotated by a different angle
                    masks: list of (B, 1, H, W), all unrotated
                    counts: (B, 1, H, W)
        '''
        B, _, H, W = image.shape
        # make batch of rotated images
        images, masks = [], []
        for idx in range(self.num_rotations):
            images.append(rotate(image, idx * self.angle, interpolation=InterpolationMode.BILINEAR))
            masks.append(rotate(mask, idx * self.angle, interpolation=InterpolationMode.BILINEAR))
        images = torch.cat(images, 0) # [num_rotations * B, num_features, H, W]
        masks = torch.cat(masks, 0) # [num_rotations * B, 1, H, W]
        # masks_lr = F.avg_pool2d(masks, kernel_size=16, stride=16, padding=0) 
        # masks_lr = (masks_lr > 0.9999).float()
        # masks = F.interpolate(masks_lr, size=(featurizer.num_patches * 16, featurizer.num_patches * 16), mode='nearest')
        # extract & normalize features
        features_lr = self.featurizer(images) # [num_rotations * B, C, H // patch_size, W // patch_size]]
        if self.use_unitnorm:
            features_lr = UnitNorm(self.num_features)(features_lr)
        if self.use_channelnorm:
            features_lr = ChannelNorm(self.num_features)(features_lr)
        # features_lr *= masks_lr
        # upsample features, do chunked forwarding to avoid OOM
        features_hr_list = [] # list of [B, C, H, W]
        features_lr_list = list(features_lr.chunk(self.num_rotations, dim=0)) # list of [B, C, H // patch_size, W // patch_size]
        images_list = images.chunk(self.num_rotations, dim=0)
        for f, im in zip(features_lr_list, images_list):
            # [B, C, H // patch_size, W // patch_size]
            features_hr_list.append(self.upsampler(f, im))        
        masks_list = list(masks.chunk(self.num_rotations, dim=0)) # list of [B, 1, H, W]
        feat_rotinv = torch.zeros_like(features_hr_list[0]) # [B, C, H, W]
        counts = torch.zeros_like(masks_list[0], dtype=int) # [B, 1, H, W]
        # rotate back
        for idx in range(self.num_rotations):
            features_hr_list[idx] = rotate(features_hr_list[idx], -idx * self.angle, interpolation=InterpolationMode.BILINEAR)
            masks_list[idx] = rotate(masks_list[idx], -idx * self.angle, interpolation=InterpolationMode.BILINEAR)
            masks_list[idx] = (masks_list[idx] > 0.5).to(features_hr_list[idx].dtype)
            feat_rotinv += features_hr_list[idx] * masks_list[idx]
            counts += masks_list[idx].int()
        feat_rotinv /= counts.clamp(min=1e-5)
        feat_rotinv = feat_rotinv
        if return_everything:
            return feat_rotinv, features_hr_list, features_lr_list, masks_list, counts
        return feat_rotinv

    @torch.no_grad()
    def get_naive_upsampled_features(self, image: torch.Tensor, mask=None, upsample_rate=4):
        B, _, H, W = image.shape
        assert self.patch_size % upsample_rate == 0
        assert upsample_rate % 2 == 0
        shift_step = self.patch_size // upsample_rate
        assert shift_step % 2 == 0
        features_upsampled = torch.zeros((B, self.num_features, self.num_patches * upsample_rate, self.num_patches * upsample_rate), device=image.device)                
        image_padded = (-torch.tensor(IMAGENET_DEFAULT_MEAN) / torch.tensor(IMAGENET_DEFAULT_STD)).to(image.device).reshape(1, 3, 1, 1).repeat(B, 1, H + self.patch_size, W + self.patch_size)
        image_padded[:, :, self.patch_size // 2: -self.patch_size // 2, self.patch_size // 2: -self.patch_size // 2] = image
        bound = int(shift_step * (upsample_rate / 2 - 0.5))
        # right shifted image should be encoded to fill the left side of the feature map
        for i, yshift in enumerate(range(bound, -bound - 1, -shift_step)):
            for j, xshift in enumerate(range(bound, -bound - 1, -shift_step)):
                image_padded_shifted = torch.roll(image_padded, shifts=(yshift, xshift), dims=(2, 3))
                image_shifted = image_padded_shifted[:, :, self.patch_size // 2: -self.patch_size // 2, self.patch_size // 2: -self.patch_size // 2]
                features = self.featurizer(image_shifted)
                features_upsampled[:, :, i::upsample_rate, j::upsample_rate] = features
        if self.use_unitnorm:
            features_upsampled = UnitNorm(self.num_features)(features_upsampled)
        if self.use_channelnorm:
            features_upsampled = ChannelNorm(self.num_features)(features_upsampled)
        return features_upsampled
    
    def get_naive_rotinv_features(self, image: torch.Tensor, mask=None, upsample_rate=4):
        '''
        For a featurizer without rot_inv, extract naively upsample features from rotated images, then rotate back and average
        For featurizers with rot_inv, just use ge_naive_upsampled_features, the featurizer does the rotation in its own loop
        '''
        assert self.rot_inv == False
        B, _, H, W = image.shape
        # make batch of rotated images
        images = []
        for idx in range(self.num_rotations):
            images.append(rotate(image, idx * self.angle, interpolation=InterpolationMode.BILINEAR))
        images = torch.cat(images, 0) # [num_rotations * B, num_features, H, W]
        features_mr = self.get_naive_upsampled_features(images, mask, upsample_rate) # [num_rotations * B, num_features, H // shift_step, W // shift_step]
        features_mr = features_mr.reshape(self.num_rotations, B, self.num_features, self.num_patches * upsample_rate, self.num_patches * upsample_rate)
        for idx in range(self.num_rotations):
            features_mr[idx] = rotate(features_mr[idx], -idx * self.angle, interpolation=InterpolationMode.BILINEAR)
        feat_rotinv = features_mr.mean(dim=0)
        return feat_rotinv
        