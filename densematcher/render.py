from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings, MeshRasterizer
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer.lighting import PointLights
from densematcher.utils import get_uniform_SO3_RT
import torch
import math
import time

@torch.autocast('cuda', enabled=False) # autocasting causes dtype issues
@torch.no_grad()
def run_rendering(device, mesh, num_views, H, W, cameras, center, add_angle_azi=0, add_angle_ele=0):
    '''
    num_views: 2-tuple, (num_azimuth, num_elevation). Ignored if cameras are provided
    add_angle_azi: float, offset azimuth angle
    by default, cameras face toward the center of the object
    return center so that this can be reused when rendering a pair of meshes
    '''
    assert len(mesh) == 1
    if cameras is None:
        bbox = mesh.get_bounding_boxes() # [1, 3, 2]
        radius = bbox.abs().max()
        if center is None:
            center = bbox.mean(2)
        rotation, translation, _, _ = get_uniform_SO3_RT(num_azimuth=num_views[0], num_elevation=num_views[1], center=center, distance=radius * 2, device=device)
    else:
        rotation, translation = cameras
    total_views = len(rotation)

    camera = PerspectiveCameras(R=rotation, T=translation, device=device)
    rasterization_settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0
    )
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=rasterization_settings)
    camera_centre = camera.get_camera_center()
    lights = PointLights(
        diffuse_color=((0.4, 0.4, 0.5),),
        ambient_color=((0.6, 0.6, 0.6),),
        specular_color=((0.01, 0.01, 0.01),),
        location=camera_centre,
        device=device,
    )
    shader = HardPhongShader(device=device, cameras=camera, lights=lights)
    batch_renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    batch_mesh = mesh.extend(total_views)
    batched_renderings = batch_renderer(batch_mesh)
    fragments = rasterizer(batch_mesh)
    depth = fragments.zbuf
    pix2face = fragments.pix_to_face # index of packed faces, not batched faces
    # ref: https://github.com/facebookresearch/pytorch3d/issues/684#issuecomment-846174601
    pix2face_offset = (torch.arange(0, total_views, dtype=torch.int64) * len(mesh.faces_list()[0]))[:, None, None, None].to(pix2face.device)
    pix2face = torch.where(pix2face == -1, pix2face, pix2face - pix2face_offset) # reduce all fg by pix2face_offset
    return batched_renderings, camera, depth, pix2face, center

def batch_render(device, mesh, num_views, H, W, cameras, center):
    trials = 0
    add_angle_azi = 0
    add_angle_ele = 0
    while trials < 5:
        try:
            return run_rendering(device, mesh, num_views, H, W, cameras, center, add_angle_azi=add_angle_azi, add_angle_ele=add_angle_ele)
        except torch.linalg.LinAlgError as e:
            trials += 1
            print("lin alg exception at rendering, retrying ", trials)
            add_angle_azi = torch.randn(1)
            add_angle_ele = torch.randn(1)
            continue
