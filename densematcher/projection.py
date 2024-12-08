import os
import shutil
import torch
from PIL import Image
from torchvision.utils import make_grid
import numpy as np
from densematcher.render import batch_render
from pytorch3d.ops import ball_query
from tqdm import tqdm
import time
import random
from pytorch3d.structures.meshes import Meshes
import matplotlib.pyplot as plt
import os

def save_renders(batched_renderings, render_dir):
    '''
    batched_renderings: [total_views, H, W, 3]
    '''
    if os.path.exists(render_dir):
        shutil.rmtree(render_dir)
    os.makedirs(render_dir, exist_ok=True)
    for i in range(len(batched_renderings)):
        plt.imsave(f"{render_dir}/view{i}.png", batched_renderings[i].cpu().numpy().clip(0, 1))

@torch.autocast('cuda', enabled=True)
def get_features_per_vertex(
    device,
    extractor,
    mesh,
    mesh_simp,
    num_views,
    ft_dim=768,
    H=512,
    W=512,
    cameras=None,
    center=torch.zeros((1, 3)),
    dummy_testing=False,
    render_dir=None,
    fts=None, # [num_views, C, H, W]
    use_lr_features=False,
):
    '''
    take a mesh, render different views, and project features onto vertices.
    if ft is not None, project onto the mesh using the given features
    render_dir: str, path to save renders
    num_views: 2-tuple, (num_azimuth, num_elevation). If cameras are provided, this is ignored
    cameras: camera extrinsic matrices, (R, T), R: [num_views, 3, 3], T: [num_views, 3]
    No normalization is applied here
    '''
    mesh_vertices = mesh_simp.verts_list()[0] # [V, 3]
    mesh_faces = mesh_simp.faces_list()[0] # [F, 3]
    
    start_s = time.time()
    
    # batch_render has no gradient
    batched_renderings, camera, depth, _, _ = batch_render(
        device, mesh, num_views, H, W, cameras=cameras, center=center
    ) # torch.tensor [bs, H, W, 4], last channel is mask, rgb is between [0, 1]
    batched_renderings_simp, _, _, pix2face, _ = batch_render(
        device, mesh_simp, num_views, H, W, cameras=cameras, center=center
    ) # pix2face is should contain *simplified* mesh's face indices
    batched_renderings, batched_renderings_simp = batched_renderings[..., :3], batched_renderings_simp[..., :3]

    if os.environ.get("TIMEIT", False):
        print("rendering took", time.time() - start_s, "for", num_views, "views", (H, W), "resolution")

    if render_dir is not None and torch.cuda.current_device() == 0:
        save_renders(batched_renderings, render_dir)
        save_renders(batched_renderings_simp, render_dir + "_simp")
    torch.cuda.empty_cache()
    ft_per_vertex = torch.zeros((len(mesh_vertices), ft_dim), dtype=torch.float32, device=device)
    ft_per_vertex_count = torch.zeros(len(mesh_vertices), dtype=torch.float32, device=device)
    if os.environ.get("VERBOSE", False):
        pbar = tqdm(range(len(batched_renderings)))
    else:
        pbar = range(len(batched_renderings))
        
    start_s = time.time()
    for idx in pbar:
        if dummy_testing: # project a cat onto the mesh's idx-th feature channel
            ft  = torch.zeros((1, ft_dim, H, W), device=device)
            dummy_img = np.asarray(Image.open("assets/images/cat.png").convert("L").resize((192, 192))) / 255.0
            ft[0, idx, :, :] = 1 # torch.from_numpy(dummy_img).to(device)
        elif fts is None:
            ft_hr, ft_lr = extractor(batched_renderings[idx]) # [1, C, H, W]
            if render_dir is not None and torch.cuda.current_device() == 0:
                torch.save(ft_hr, f"{render_dir}/feat{idx}.pt")
                torch.save(ft_lr, f"{render_dir}/feat_lr{idx}.pt")
            ft = ft_lr if use_lr_features else ft_hr
        else:
            ft = fts[idx:idx + 1]
        # Indices of unique visible faces
        fg_mask = pix2face[idx] != -1
        visible_faces = pix2face[idx][fg_mask].unique()   # (num_visible_faces )
        # Get Indices of unique visible verts using the vertex indices in the faces
        visible_verts = mesh_faces[visible_faces].unique() # 1D tensor of indices
        projected_verts = camera[idx].get_full_projection_transform().transform_points(mesh_vertices[visible_verts])[:, :2] # [V, 2], pytorch3D NDC(up+ left+)
        # only keep those between -1 and 1. Not all vertices on visible faces are visible
        visible_verts_mask = ((-1 < projected_verts) & (projected_verts < 1)).all(dim=1) # [len(projected_verts)]
        visible_verts = visible_verts[visible_verts_mask] # [V, 3]
        projected_verts = projected_verts[visible_verts_mask] # [V, 2]
        dummy_grid = -1 * projected_verts[None, None, :, :2] # the coordinates are [x, y]
        visible_vertex_features = torch.nn.functional.grid_sample(ft, dummy_grid, align_corners=True).squeeze().T # [V, C]
        ft_per_vertex[visible_verts] += visible_vertex_features
        if dummy_testing:
            sil = torch.zeros((192, 192, 3)).to(mesh_vertices)
            sil[..., 0] = 1 - torch.from_numpy(dummy_img)
            projected_x, projected_y = -projected_verts[:, 0], -projected_verts[:, 1]
            sil[(projected_y * 96 + 96).int(), (projected_x * 96 + 96).int(), 1] = 1
            plt.imsave("tmp.png", sil.cpu().numpy())
            sil2 = torch.zeros((192, 192, 3)).to(mesh_vertices)
            sil2[..., 0] = 1 - torch.from_numpy(dummy_img)
            sil2[(projected_y * 96 + 96).int(), (projected_x * 96 + 96).int(), 1] = 1 - visible_vertex_features[:, idx]
            plt.imsave("tmp2.png", sil2.cpu().numpy())
            ft_per_vertex_count[visible_verts] = 1
        else:
            ft_per_vertex_count[visible_verts] += 1
    filled_indices = ft_per_vertex_count != 0
    missing_indices = ft_per_vertex_count == 0
    ft_per_vertex[filled_indices, :] = ft_per_vertex[filled_indices, :] / ft_per_vertex_count[filled_indices][..., None]
    missing_features = sum(missing_indices)
    # don't fill missing vertices, let diffusionet take care of those indices
    if os.environ.get("TIMEIT", False):
        print("computing multiview 2D features took", time.time() - start_s, "for", num_views, "views", (H, W), "resolution")

    if os.environ.get("VERBOSE", False):
        print("Number of missing features: ", missing_features)

    return ft_per_vertex
