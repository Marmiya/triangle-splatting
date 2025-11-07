"""
BRDF Triangle Renderer using diff-triangle-rasterization

Uses Triangle Splatting's native rasterizer with window function
BRDF colors are pre-computed in Python and passed to CUDA rasterizer

Architecture:
  Python: BRDF shading (ue5_default_lit_brdf + ue5_environment_brdf)
  CUDA: Triangle rasterization with window function (sigma parameter)

Author: VCCSim Team
"""

import torch
import torch.nn.functional as F
import math
from diff_triangle_rasterization import TriangleRasterizationSettings, TriangleRasterizer
from utils.brdf_utils import ue5_default_lit_brdf, ue5_environment_brdf


def compute_brdf_colors(triangles, base_color, roughness, metallic, camera_center, pipe):
    """
    Pre-compute BRDF colors for each triangle in Python

    Args:
        triangles: [N, 3, 3] triangle vertices
        base_color: [N, 3] per-triangle base color
        roughness: [N, 1] per-triangle roughness
        metallic: [N, 1] per-triangle metallic
        camera_center: [3] camera position in world space
        pipe: Pipeline config with lighting parameters

    Returns:
        colors: [N, 3] per-triangle BRDF colors
    """
    N = triangles.shape[0]

    # Compute per-triangle normals
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = torch.cross(edge1, edge2, dim=1)  # [N, 3]
    normals = F.normalize(normals, dim=1)

    # Compute per-triangle view directions (from triangle center to camera)
    triangle_centers = triangles.mean(dim=1)  # [N, 3]
    view_dirs = F.normalize(camera_center.unsqueeze(0) - triangle_centers, dim=1)  # [N, 3]

    # Get lighting configuration
    if hasattr(pipe, 'light_direction') and pipe.light_direction is not None:
        light_dir = pipe.light_direction.unsqueeze(0).expand(N, -1)  # [N, 3]
    else:
        light_dir = torch.tensor([[0.5773, 0.5773, -0.5773]], device="cuda").expand(N, -1)

    if hasattr(pipe, 'light_color') and pipe.light_color is not None:
        light_color = pipe.light_color.unsqueeze(0).expand(N, -1)  # [N, 3]
    else:
        light_color = torch.tensor([[1.0, 1.0, 1.0]], device="cuda").expand(N, -1)

    if hasattr(pipe, 'ambient_color') and pipe.ambient_color is not None:
        ambient_color = pipe.ambient_color.unsqueeze(0).expand(N, -1)  # [N, 3]
    else:
        ambient_color = torch.tensor([[0.3, 0.3, 0.3]], device="cuda").expand(N, -1)

    # Compute BRDF shading
    # Direct lighting (main light source)
    direct_lighting = ue5_default_lit_brdf(
        base_color, roughness, metallic,
        normals, view_dirs,
        light_dir, light_color
    )  # [N, 3]

    # Ambient/environment lighting
    ambient_lighting = ue5_environment_brdf(
        base_color, roughness, metallic,
        normals, view_dirs, ambient_color
    )  # [N, 3]

    # Combine lighting
    final_colors = direct_lighting + ambient_lighting  # [N, 3]
    final_colors = torch.clamp(final_colors, 0.0, 1.0)

    return final_colors


def render_brdf(viewpoint_camera, triangle_model, pipe, bg_color, scaling_modifier=1.0):
    """
    Render BRDF triangles using diff-triangle-rasterization

    Args:
        viewpoint_camera: Camera object with matrices and parameters
        triangle_model: TriangleBRDFModel instance
        pipe: Pipeline configuration
        bg_color: Background color tensor [3]
        scaling_modifier: Scaling modifier (default 1.0)

    Returns:
        dict: Rendering results
    """
    # Validate triangle data
    if triangle_model.get_triangles_points.ndim < 3:
        print(f"[ERROR] Triangle points tensor has insufficient dimensions: {triangle_model.get_triangles_points.ndim}")
        print(f"[ERROR] Expected shape: (N, 3, 3), got shape: {triangle_model.get_triangles_points.shape}")
        raise ValueError(f"Triangle points tensor should have 3 dimensions, got {triangle_model.get_triangles_points.ndim}")

    # Get triangle data
    triangles = triangle_model.get_triangles_points  # [N, 3, 3]
    base_color = triangle_model.get_base_color       # [N, 3]
    roughness = triangle_model.get_roughness         # [N, 1]
    metallic = triangle_model.get_metallic           # [N, 1]
    opacity = triangle_model.get_opacity             # [N, 1]
    sigma = triangle_model.get_sigma                 # [N, 1]

    # Pre-compute BRDF colors in Python
    colors_precomp = compute_brdf_colors(
        triangles, base_color, roughness, metallic,
        viewpoint_camera.camera_center, pipe
    )  # [N, 3]

    # Create zero tensor for screenspace points (required by original API)
    screenspace_points = torch.zeros_like(
        triangle_model.get_triangles_points[:, 0, :].squeeze(),
        dtype=triangle_model.get_triangles_points.dtype,
        requires_grad=True,
        device="cuda"
    ) + 0

    scaling = torch.zeros_like(
        triangle_model.get_triangles_points[:, 0, 0].squeeze(),
        dtype=triangle_model.get_triangles_points.dtype,
        requires_grad=True,
        device="cuda"
    ).detach()

    density_factor = torch.zeros_like(
        triangle_model.get_triangles_points[:, 0, 0].squeeze(),
        dtype=triangle_model.get_triangles_points.dtype,
        requires_grad=True,
        device="cuda"
    ).detach()

    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = TriangleRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,  # Not using SH (BRDF instead)
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug if hasattr(pipe, 'debug') else False
    )

    rasterizer = TriangleRasterizer(raster_settings=raster_settings)

    # Apply mask (for densification/pruning)
    if hasattr(triangle_model, '_mask'):
        mask = ((torch.sigmoid(triangle_model._mask) > 0.01).float() -
                torch.sigmoid(triangle_model._mask)).detach() + torch.sigmoid(triangle_model._mask)
        opacity = opacity * mask

    # Rasterize with BRDF colors
    rendered_image, radii, scaling, density_factor, allmap, max_blending = rasterizer(
        triangles_points=triangle_model.get_triangles_points_flatten,
        sigma=sigma,
        num_points_per_triangle=triangle_model.get_num_points_per_triangle,
        cumsum_of_points_per_triangle=triangle_model.get_cumsum_of_points_per_triangle,
        number_of_points=triangle_model.get_number_of_points,
        shs=None,  # Not using SH
        colors_precomp=colors_precomp,  # Use pre-computed BRDF colors
        opacities=opacity,
        means2D=screenspace_points,
        scaling=scaling,
        density_factor=density_factor
    )

    rets = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "scaling": scaling,
        "density_factor": density_factor,
        "max_blending": max_blending
    }

    # Extract additional outputs from allmap
    render_alpha = allmap[1:2]

    # Get normal map (transform from view space to world space)
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1, 2, 0) @
                    (viewpoint_camera.world_view_transform[:3, :3].T)).permute(2, 0, 1)

    # Get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # Get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # Get depth distortion map
    render_dist = allmap[6:7]

    # Pseudo surface depth (for regularization)
    from utils.point_utils import depth_to_normal
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    # Pseudo surface normal
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2, 0, 1)
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update({
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
    })

    return rets
