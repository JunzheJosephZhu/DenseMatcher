import torch
import numpy as np
import os
from PIL import Image
from .SDDINO import SDDINOFeaturizer

def get_featurizer(name, num_patches, rot_inv=False, aggre_net_weights_folder='checkpoints/SDDINO_weights', **kwargs):
    name = name.lower()
    if name == "sd_dino":
        patch_size = 16
        model = SDDINOFeaturizer(num_patches=num_patches, diffusion_ver='v1-5', extractor_name='dinov2_vitb14', aggre_net_weights_path=f'{aggre_net_weights_folder}/best_{num_patches * patch_size}.PTH', rot_inv=rot_inv)
        dim = 768
    else:
        raise ValueError("unknown model: {}".format(name))
    return model, patch_size, dim

def resize(img, target_res, resize=True, to_pil=True):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    canvas = np.zeros([target_res, target_res, original_channels], dtype=np.uint8) if original_channels > 1 else np.zeros([target_res, target_res], dtype=np.uint8)
    if original_height <= original_width:
        if resize:
            img = img.resize((target_res, int(np.round(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
        width, height = img.size
        img = np.asarray(img)
        vertical_padding = (target_res - height) // 2
        canvas[vertical_padding:vertical_padding+height, :] = img
    else:
        if resize:
            img = img.resize((int(np.round(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
        width, height = img.size
        img = np.asarray(img)
        horizontal_padding = (target_res - width) // 2
        canvas[:, horizontal_padding:horizontal_padding+width] = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas
