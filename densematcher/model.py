import torch
from torch import nn
from densematcher.projection import get_features_per_vertex
from densematcher.functional_map import compute_surface_map
from densematcher.extractor import DINOv2FeatureExtractor, DIFTFeatureExtractor, SDDINOFeatureExtractor
from PIL import Image
import numpy as np
import os
import os.path as osp
from densematcher import diffusion_net
from inspect import signature
import time

class MeshFeaturizer(nn.Module):
    def __init__(self, pretrained_upsampler_path, num_views, num_blocks, width, aggre_net_weights_folder="checkpoints/SDDINO_weights", reconstructor_layers=1, num_encoding_functions=8, num_hks=16):
        '''
        pretrained_upsampler_path: path to featup's .ckpt save file
        num_views: 2-tuple, (num_azimuth, num_elevation)
        Azimuth accounts for the whole circle. Elevation doesn't account for poles.
        '''
        super().__init__()
        self.pretrained_upsampler_path = pretrained_upsampler_path
        self.extractor_2d = SDDINOFeatureExtractor(pretrained_upsampler_path=pretrained_upsampler_path, aggre_net_weights_folder=aggre_net_weights_folder)
        self.H = self.W = self.extractor_2d.num_patches * 16
        self.num_views = num_views
        self.reconstructor_layers = reconstructor_layers
        self.num_encoding_functions = num_encoding_functions
        self.num_hks = num_hks
        self.pos_enc_size = 3 * (2 * num_encoding_functions + 1)
        self.diffusion_in_channels = self.extractor_2d.featurizer.num_features + self.num_hks + self.pos_enc_size
        self.extractor_3d = diffusion_net.layers.DiffusionNet(
            C_in=self.diffusion_in_channels,
            C_out=width,
            C_width=width,
            N_block=num_blocks,
            dropout=True,
        )
        # for reconstruction loss of original features
        if reconstructor_layers == -1: # mirror arch
            self.reconstructor = diffusion_net.layers.DiffusionNet(
            C_in=width,
            C_out=self.extractor_2d.featurizer.num_features,
            C_width=width,
            N_block=num_blocks,
            dropout=True,
        )
        elif reconstructor_layers == 1:
            self.reconstructor = nn.Linear(width, self.extractor_2d.featurizer.num_features)
        else:
            reconstructor_blocks = []
            for _ in range(reconstructor_layers - 1):
                reconstructor_blocks.append(nn.Linear(width, width))
                reconstructor_blocks.append(nn.ReLU())
            reconstructor_blocks.append(nn.Linear(width, self.extractor_2d.featurizer.num_features))
            self.reconstructor = nn.Sequential(*reconstructor_blocks)
        # override for ablation study
        self.use_lr_features = False
                
    @property
    def device(self):
        return next(self.parameters()).device
    
    def positional_encoding(self, tensor, include_input=True, log_sampling=True) -> torch.Tensor:
        r"""
        Yanked from TinyNeRF
        Apply positional encoding to the input.

        Args:
            tensor (torch.Tensor): Input tensor to be positionally encoded. # [..., d], between [-pi, pi]
            encoding_size (optional, int): Number of encoding functions used to compute
                a positional encoding (default: 6).
            include_input (optional, bool): Whether or not to include the input in the
                positional encoding (default: True).

        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """
        if os.environ.get("VERBOSE", False):
            print("vertex position range:", tensor.min(), tensor.max())
        num_encoding_functions = self.num_encoding_functions
                
        # TESTED
        # Trivially, the input tensor is added to the positional encoding.
        encoding = [tensor] if include_input else []
        frequency_bands = None
        if log_sampling:
            frequency_bands = 2.0 ** torch.linspace(
                0.0,
                num_encoding_functions - 1,
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
            frequency_bands = torch.linspace(
                2.0 ** 0.0,
                2.0 ** (num_encoding_functions - 1),
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )

        for freq in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)
    
    def extract_features_2d(self, img):
        '''
        img: torch.Tensor, (H, W, 3), on a cuda device
        return: features, torch.Tensor, [1, C, H, W], on a cuda device
        '''
        img_pil = Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
        image_tensor, mask_tensor, resized_img_pil = self.extractor_2d.preprocess(img_pil)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        mask_tensor = mask_tensor.unsqueeze(0).to(self.device)
        features_hr, features_lr = self.extractor_2d.forward(image_tensor, mask_tensor, return_everything=True)
        return features_hr, features_lr
    
    def forward(self, mesh_raw, mesh_simp, operators, cameras, return_mvfeatures=False):
        '''
        mesh_raw: whatever the fuck format objaverse provided. Probably has shit topology. Use to render multiview images
        mesh_simp: simplified mesh with a reasonable number of vertices. Use to compute vertex features
        cameras: cameras extrinsics, tuple of torch.Tensor R and t with shape [N, 3, 3] and [N, 3].
        NOTE: R is **transpose** of normal R!!!!! This is because pytorch3d has weird ass conventions 
        output normalized per-vertex features
        '''
        mesh_raw = mesh_raw.to(self.device)
        mesh_simp = mesh_simp.to(self.device)


        with torch.no_grad():
            mv_features = get_features_per_vertex(# import diffusion_net.utils as utils
                device=self.device,
                extractor=self.extract_features_2d,
                mesh=mesh_raw,
                mesh_simp=mesh_simp,
                num_views=self.num_views,
                H=self.H,
                W=self.W,
                cameras=cameras,
                render_dir=os.environ.get("RENDER_DIR", None),
                use_lr_features=self.use_lr_features,
            ) # [V, 768]

         
        start_s = time.time()
        # assuming mesh scale is 0.3, normalized by extent and centered, it would be [-1/6, 1/6]. We rescale to [-pi, pi]
        pos_enc = self.positional_encoding(mesh_simp.verts_list()[0] * np.pi * 6) # [V, 51] 

        frames, massvec, L_list, evals, evecs, gradX, gradY = (op.to(mesh_simp.device) for op in operators)
        hks = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, self.num_hks) # [V, 16]
        if os.environ.get("TIMEIT", False):
            print("computing XYZ pos enc and HKS took", time.time() - start_s)

        start_s = time.time()
        # sparse mm doesnt work with half
        in_features = torch.cat([mv_features, pos_enc, hks], dim=1) # [V, diffusion_in_channels]
        out_features = self.extractor_3d(in_features, massvec, L_list, evals, evecs, gradX, gradY, edges=None, faces=None)
        if os.environ.get("TIMEIT", False):
            print("diffusion net forward took", time.time() - start_s, "with", mv_features.shape[0], "vertices")
        # normalize
        out_features_normalized = out_features / out_features.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        if return_mvfeatures:
            return out_features_normalized, out_features, mv_features

        return out_features_normalized