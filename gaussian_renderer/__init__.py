#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene import Scene
from utils.sh_utils import eval_sh

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render_facemask(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc._opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # scales = None
    # rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None #pc.get_features
    colors_precomp = pc._features_dc

    # if override_color is None:
    #     if pipe.convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    #         dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #     else:
    #         print("using SH")
    #         shs = pc.get_features
    # else:
    #     colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    del rasterizer
    del screenspace_points
    del colors_precomp
    del rotations
    del opacity
    del means2D
    del means3D
    del raster_settings
    del tanfovx
    del tanfovy
    torch.cuda.empty_cache()
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            # "viewspace_points": screenspace_points,
            # "visibility_filter" : radii > 0,
            # "radii": radii,
            "scaling": scales,
            }

def render_wSH(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc._opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # scales = None
    # rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = pc.get_SH #[render_numGauss, int(sh_degree+1)**2, 3]
    colors_precomp = None

    # if override_color is None:
    #     if pipe.convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    #         dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #     else:
    #         print("using SH")
    #         shs = pc.get_features
    # else:
    #     colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    del rasterizer
    del screenspace_points
    del colors_precomp
    del rotations
    del opacity
    del means2D
    del means3D
    del raster_settings
    del tanfovx
    del tanfovy
    torch.cuda.empty_cache()
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            # "viewspace_points": screenspace_points,
            # "visibility_filter" : radii > 0,
            # "radii": radii,
            "scaling": scales,
            }


def render_wVC(viewpoint_camera, current_scene:Scene, current_gaussians : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, current_xyz=None, current_feat=None, IL=None, region=None, mask=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!

    current_feat:  [#plane, 3007, #featdim]
    
    """   

    # compute view-direction and distance
    anchor_pos = current_xyz[:IL.num_anchors, :] #[3007, 3]
    # mean_pos = anchor_pos.mean()

    ob_view = anchor_pos - viewpoint_camera.camera_center[None] #[3007, 3]
    
    # ob_view = mean_pos - viewpoint_camera.camera_center #[3]
    # ob_view = ob_view.unsqueeze(0) #[1, 3]
    
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True) #[3007, 1]
    # view
    ob_view = ob_view / ob_dist #[3007, 3]

    # noise = torch.randn_like(ob_view)
    # ob_view = ob_view + noise*1e-4

    if IL.dist_encoding:
        ob_dist = ob_dist.unsqueeze(0).repeat([IL.num_planes, 1, 1]) #[#plane, 3007, 1]
        
    print("dist encoding: ", IL.dist_encoding)
    ob_view = ob_view.unsqueeze(0).repeat([IL.num_planes, 1, 1]) #[#plane, 3007, 3]

    # mean_view direction case
    # ob_view = ob_view.unsqueeze(0).repeat([IL.num_planes, IL.num_anchors, 1]) #[#plane, 3007, 3]

    # concatenate with current feat
    if IL.dist_encoding:
        wVC_feat = torch.cat([current_feat, ob_view, ob_dist], dim = -1) #[#plane, 3007, #featdim+3+1]
    else:
        wVC_feat = torch.cat([current_feat, ob_view], dim = -1) #[#plane, 3007, #featdim+3]

    # compute current neural gaussians attributes
    # [#plane, 3007, 64] -> [3007, 3*4]
    if IL.num_planes>1:
        s_fdc = IL.color_decoder(wVC_feat[0])
    else:
        s_fdc = IL.color_decoder(wVC_feat)

    s_fdc = s_fdc.reshape(IL.render_numGauss, IL.fdc_ch) 
    _fdc = IL.pp_fdc(s_fdc) #still RGB [0, 1]

    if IL.num_planes>1:
        s_scale = IL.scale_decoder(wVC_feat[1])
    else:
        s_scale = IL.scale_decoder(wVC_feat)

    s_scale = s_scale.reshape(IL.render_numGauss, IL.scale_ch)
    _scales = IL.pp_scale(s_scale + IL.mu_scale)

    # [#plane, 3007, 64] -> [3007, 4*4]
    if IL.num_planes>1:
        s_rot = IL.rot_decoder(wVC_feat[2])
    else:
        s_rot = IL.rot_decoder(wVC_feat)

    s_rot = s_rot.reshape(IL.render_numGauss, IL.rot_ch) 
    _rots = IL.pp_rot(s_rot+IL.mu_rot)

    # [#plane, 3007, 64] -> [3007, 1*4]
    if IL.num_planes>1:
        s_opac = IL.opac_decoder(wVC_feat[3])
    else:
        s_opac = IL.opac_decoder(wVC_feat)

    s_opac = s_opac.reshape(IL.render_numGauss, IL.opac_ch)   
    _opacities = IL.pp_opac(s_opac + IL.mu_opac)

    _frest = None#torch.zeros(self.render_numGauss, int((((self.sh_degree+1)**2)*3-3)/3), 3).float().cuda()
    torch.cuda.empty_cache()
    
    current_fdc = _fdc.squeeze(0)
    current_frest = _frest
    current_scales = _scales.squeeze(0)
    current_rots = _rots.squeeze(0)
    current_opacities = _opacities.squeeze(0)

    eps = 1e-8
    opac_eps = 1e-5

    # validate fdc[1e-8, 1], opacity[1e-8, 1]
    nudge_fdc = (abs(current_fdc)<eps)*eps
    current_fdc = current_fdc + nudge_fdc #[1e-8, 1]
    
    nudge_opac = (abs(current_opacities)<opac_eps)*opac_eps
    current_opacities = current_opacities + nudge_opac #[1e-8, 1]

    with torch.no_grad():
        print("RGB: ", current_fdc.mean())
        print("RGB MAX/MIN: ", current_fdc.max(), current_fdc.min())
        
        print("OPAC: ", current_opacities.mean())
        print("OPAC MAX/MIN: ", current_opacities.max(), current_opacities.min())
        
    # set current neural gaussians attributes and xyz
    current_scene.set_gaussians(gaussians=current_gaussians, input_format='learnable_blenshape', index_in_batch=0, xyz=current_xyz, fdc=current_fdc, frest=current_frest, scale=current_scales, rot=current_rots, opac=current_opacities)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(current_gaussians.get_xyz, dtype=current_gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=current_gaussians.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = current_gaussians._xyz
    means2D = screenspace_points
    opacity = current_gaussians._opacity

    cov3D_precomp = None
    scales = current_gaussians._scaling
    rotations = current_gaussians._rotation
    shs = None #pc.get_features
    colors_precomp = current_gaussians._features_dc

    if region!=None and (region == "face" or region == "face_neck"):
        facemask = (mask==True)
        # facemask = facemask.view(-1)
        print("mask: ", facemask.shape)
        means3D = means3D[facemask]
        means2D = means2D[facemask]
        opacity = opacity[facemask]
        scales = scales[facemask]
        rotations = rotations[facemask]
        colors_precomp = colors_precomp[facemask]
    elif region!= None and region == "hair_torso":
        ht_mask = (mask==False)
        # ht_mask = ht_mask.view(-1)
        print("mask :", ht_mask.shape)
        means3D = means3D[ht_mask]
        means2D = means2D[ht_mask]
        opacity = opacity[ht_mask]
        scales = scales[ht_mask]
        rotations = rotations[ht_mask]
        colors_precomp = colors_precomp[ht_mask]
    else:
        print("fullhead rendering")

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    del rasterizer
    del screenspace_points
    del colors_precomp
    del rotations
    del opacity
    del means2D
    del means3D
    del raster_settings
    del tanfovx
    del tanfovy
    del current_fdc
    del current_frest
    del current_scales
    del current_rots
    del current_opacities
    torch.cuda.empty_cache()
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            # "viewspace_points": screenspace_points,
            # "visibility_filter" : radii > 0,
            # "radii": radii,
            "scaling": scales,
            }


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, region=None, mask=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!

    regions: [face, hair_torso]
    mask = [#render_Gauss]: IL.original_facemask_bool
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc._xyz
    means2D = screenspace_points
    opacity = pc._opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # scales = None
    # rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    scales = pc._scaling
    rotations = pc._rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None #pc.get_features
    colors_precomp = pc._features_dc

    if region!=None and (region == "face" or region == "face_neck"):
        facemask = (mask==True)
        # facemask = facemask.view(-1)
        print("mask: ", facemask.shape)
        means3D = means3D[facemask]
        means2D = means2D[facemask]
        opacity = opacity[facemask]
        scales = scales[facemask]
        rotations = rotations[facemask]
        colors_precomp = colors_precomp[facemask]
    elif region!= None and region == "hair_torso":
        ht_mask = (mask==False)
        # ht_mask = ht_mask.view(-1)
        print("mask :", ht_mask.shape)
        means3D = means3D[ht_mask]
        means2D = means2D[ht_mask]
        opacity = opacity[ht_mask]
        scales = scales[ht_mask]
        rotations = rotations[ht_mask]
        colors_precomp = colors_precomp[ht_mask]
    else:
        print("fullhead rendering")

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    del rasterizer
    del screenspace_points
    del colors_precomp
    del rotations
    del opacity
    del means2D
    del means3D
    del raster_settings
    del tanfovx
    del tanfovy
    torch.cuda.empty_cache()
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            # "viewspace_points": screenspace_points,
            # "visibility_filter" : radii > 0,
            # "radii": radii,
            "scaling": scales,
            }

def original_render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
