import torch
import numbers
import numpy as np
import argparse
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures, Materials
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
import meshplot as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorsys
import scipy.io as sio
import os
import pickle
from plyfile import PlyData
import trimesh
import pytorch3d
import warnings
import copy
import time
import pymeshlab as pml
# import gpytoolbox
import igraph as ig
import traceback
import pywavefront
from pytorch3d.renderer.mesh.textures import TexturesUV, TexturesVertex
from densematcher.diffusion_net.utils import hash_arrays
from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras

def generate_colors(n):
    hues = [i / n for i in range(n)]
    saturation = 1
    value = 1
    colors = [colorsys.hsv_to_rgb(hue, saturation, value) for hue in hues]
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    return colors

def plot_mesh(myMesh,cmap=None):
    mp.plot(myMesh.vert, myMesh.face,c=cmap)
    
def double_plot(myMesh1,myMesh2,cmap1=None,cmap2=None):
    d = mp.subplot(myMesh1.verts_list()[0].cpu().numpy(), myMesh1.faces_list()[0].cpu().numpy(), c=cmap1, s=[2, 2, 0])
    mp.subplot(myMesh2.verts_list()[0].cpu().numpy(), myMesh2.faces_list()[0].cpu().numpy(), c=cmap2, s=[2, 2, 1], data=d)

def get_colors(vertices):
    min_coord,max_coord = np.min(vertices,axis=0,keepdims=True),np.max(vertices,axis=0,keepdims=True)
    cmap = (vertices-min_coord)/(max_coord-min_coord)
    return cmap


def load_pytorch3d_mesh(filename, device=torch.device("cpu"), remesh_verts=-1, check_manifold=False, load_with_o3d=False):
    '''

    1. good explanation of obj file: https://pytorch3d.readthedocs.io/en/latest/modules/io.html#pytorch3d.io.load_obj
    2. In pytorch 3d, UV map is flipped by the Y-axis
    Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/mesh/textures.py#L1220
    '''
    mesh = load_objs_as_meshes([filename], device=device)
    # for trimesh, process=False to not merge vertices, since pytorch3d doesnt merge vertices
    # for pywavefront, create_materials=True in case some materials encoded in usemtl in .obj file are missing
    wavefront = pywavefront.Wavefront(filename, create_materials=True)
    vertex_colors = np.array([vertex[3:6] for vertex in wavefront.vertices])
    assert len(vertex_colors) == len(mesh.verts_list()[0]), "Number of vertex colors does not match number of vertices"
    assert 0 <= vertex_colors.min() and vertex_colors.max() <= 1, "Vertex colors are not in [0, 1]"
    # pytorch3d only loads uv texture from objs. if UV texture is missing, use vertex colors 
    if mesh.textures is None :
        mesh.textures = Textures(verts_rgb=[torch.tensor(vertex_colors, dtype=torch.float32)])
    return mesh

def recenter(mesh, mesh_simp):
    '''
    move a pair of meshes to the center of the first mesh's bounding box
    mesh & mesh simp: pytorch3d Meshes
    '''
    bbox = mesh_simp.get_bounding_boxes() # [1, 3, 2]
    center = bbox.mean(2)
    n1 = mesh.verts_list()[0].shape[0]
    n2 = mesh_simp.verts_list()[0].shape[0]
    mesh.offset_verts_(-center.repeat(n1, 1))
    mesh_simp.offset_verts_(-center.repeat(n2, 1))

def get_uniform_SO3_RT(num_azimuth, num_elevation, distance, center, device="cpu", add_angle_azi=0, add_angle_ele=0):
    '''
    Get a bunch of camera extrinsics centered towards center with uniform distance in polar coordinates(elevation and azimuth)
    Args:
        num_elevation: int, number of elevation angles, excluding the poles
        num_azimuth: int, number of azimuth angles
        distance: radius of those transforms
        center: center around which the transforms are generated. Needs to be torch.tensor of shape [1, 3]
    Returns:
        rotation: torch.tensor of shape [num_views, 3, 3]
        translation: torch.tensor of shape [num_views, 3]
        Weirdly in pytorch3d y-axis is for world coordinate's up axis
        pytorch3d also has a weird as convention where R is right mulplied, so its actually the inverse of the normal rotation matrix
    '''
    grid = torch.zeros((num_elevation, num_azimuth, 2)) # First channel azimuth, second channel elevation
    azimuth = torch.linspace(0, 360, num_azimuth + 1)[:-1] 
    elevation = torch.linspace(-90, 90, num_elevation + 2)[1:-1] 
    grid[:, :, 0] = azimuth[None, :]
    grid[:, :, 1] = elevation[:, None]
    grid = grid.view(-1, 2)
    top_down = torch.tensor([[0, -90], [0, 90]]) # [2, 2]
    grid = torch.cat([grid, top_down], dim=0) # [num_views, 2]
    azimuth = grid[:, 0] + add_angle_azi
    elevation = grid[:, 1] + add_angle_ele
    
    rotation, translation = look_at_view_transform(
        dist=distance, azim=azimuth, elev=elevation, device=device, at=center
    )
    return rotation, translation, azimuth, elevation

def get_distance_between_groups(geodesic_distmat, group1, group2):
    '''
    geodesic_distmat: np.ndarray, distance matrix of [V, V]
    ring1, ring2: lists of vertex indices
    return: a scalar
    '''
    if len(group1) == 0 or len(group2) == 0:
        if os.environ.get('VERBOSE', False):
            print("Warning: empty group when computing distance between groups")
        return 0
    group_distmat = geodesic_distmat[np.array(group1)[:, None], np.array(group2)[None, :]] # geodesic_distmat[np.ix_(ring1, ring2)])
    id1, id2 = linear_sum_assignment(group_distmat)
    return group_distmat[id1, id2].mean()

def get_groups_dmtx(geodesic_distmat, groups):
    '''
    get a [num_rings, num_rings] distance matrix between rings
    '''
    num_rings = len(groups)
    d_rings = np.full((num_rings, num_rings), -1.0)
    for i in range(num_rings):
        for j in range(num_rings):
            if i == j:
                d_rings[i, j] = 0
            elif d_rings[j, i] != -1:
                d_rings[i, j] = d_rings[j, i]
            else:
                d_rings[i, j] = get_distance_between_groups(geodesic_distmat, groups[i], groups[j])
    return d_rings
