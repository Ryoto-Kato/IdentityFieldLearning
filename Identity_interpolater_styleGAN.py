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

import os
import torch
import torchvision
import random
from torch import nn
from torch.nn import functional as F

import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.util import get_expon_weight_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.general_utils import inverse_sigmoid
from identity_learner_simple_styleGAN import IdentityInterpolater
# from utils.vgg_loss import VGGLoss
# vgg = VGGLoss().to("cuda")

# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
# LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize = True).cuda()

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
import matplotlib.pyplot as plt
import trimesh
from torchimize.functions import lsq_lma


from skimage.restoration import (
    denoise_tv_chambolle
)

# get current date and year
now = datetime.now()
date = now.strftime("%d") + now.strftime("%m") + now.strftime("%Y")
# print(date)
time = now.strftime("%H_%M")
# print("time:", time)
# path_to_tdImage=os.path.join(os.getcwd(), "output", "images", date+"_"+time[:2])
# path_to_tdMesh = os.path.join(os.getcwd(), "output", "meshes", date+"_"+time[:2])
# if not os.path.exists(path_to_tdImage):
#     os.mkdir(path_to_tdImage)
# if not os.path.exists(path_to_tdMesh):
#     os.mkdir(path_to_tdMesh)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

path_to_3WI = os.path.join(os.getcwd(), os.pardir, "3DSSL-WS23_IntuitiveAnimation")
sys.path.append(os.path.join(path_to_3WI, 'src'))
from utils.OBJ_helper import OBJ
from utils.Dataset_handler import Filehandler
from utils.pickel_io import dump_pckl


def identity_interpolating(dataset, pipe, path2blendshape, expName, unique_str, path_to_output = "", scale=1.0, configs = {"COG": 1e10, "CS": 1e10, "TV": 1.0}):
    """
    input:
        path2blendshape: PATH to blendshape .pth

    """
    id_frame = 0
    path2blendshape = configs["path2blendshape"]
    n_frames = configs["n_frames"]
    loop = configs["loop"]
    set_id = configs["set_id"]
    render_camInterpolation = configs["render_camInterpolation"]
    
    if set_id == 0:
        limited_subject_names = ["018", "024"]
    elif set_id == 1:
        limited_subject_names = ["036", "085"]
    elif set_id == 2:
        limited_subject_names = ["109", "201"]
    elif set_id == 3:
        limited_subject_names = ["222", "238"]
    elif set_id == 4:
        limited_subject_names = ["244", "232"]
    elif set_id == 5:
        limited_subject_names = ["074", "182"]
    elif set_id == -1:
        limited_subject_names = ["018", "024", "036", "085", "109", "201", "222", "238"]
    elif set_id == -2:
        limited_subject_names = ["244", "024", "074", "182", "180", "137", "175", "170"]
    elif set_id == -3:
        limited_subject_names = ["082", "024", "104", "108", "106", "199", "195", "200"]

    log_text = f"interp_{limited_subject_names[0]}-{limited_subject_names[-1]}"

    first_iter = 0
    progress_bar = tqdm(range(first_iter, n_frames*(len(limited_subject_names)-1)), desc="Training progress")
    first_iter += 1

    unique_str = unique_str
    model_path = os.path.join(path_to_output, unique_str, expName, str(id_frame))
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    II = IdentityInterpolater(path2blendshape=path2blendshape, subject_names=limited_subject_names, n_frames=n_frames*(len(limited_subject_names)-1), loop = loop)
    sh_degree = II.sh_degree

    subject_name = limited_subject_names[1]
    current_gaussians = GaussianModel(sh_degree=sh_degree)
    current_scene=Scene(dataset, expName = expName, ALLcam=True, frame_counter=id_frame, subject_id=subject_name, scale=scale, Center=True, Render_Interp_CAM=True)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if render_camInterpolation:
        viewpoint_stack = current_scene.getInterpCameras().copy() # interp among all test cameras
    else:
        viewpoint_stack = current_scene.getTrainCameras().copy() # center camera


    frame_rate = 30

    for frame_id in range(II.max_frames):

        if not viewpoint_stack:
            if render_camInterpolation:
                viewpoint_stack = current_scene.getInterpCameras().copy() # interp among all test cameras
            else:
                viewpoint_stack = current_scene.getTrainCameras().copy() # center camera
        viewpoint_cam = viewpoint_stack.pop(0)

        print("frame_id: ", frame_id)
        # get UVs
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        iter_start.record()

        current_xyz, current_fdc, current_frest, current_scales, current_rots, current_opacities= II(frame_id=frame_id)

        current_scene.set_gaussians(gaussians=current_gaussians, input_format='learnable_blenshape',index_in_batch=0,
                            xyz=current_xyz,
                            fdc=current_fdc,
                            frest=current_frest,
                            rot=current_rots,
                            opac=current_opacities,
                            scale=current_scales)

        renderArgs = (pipe, background)
        with torch.no_grad():
            image = render(viewpoint_cam, current_scene.gaussians, *renderArgs)["render"].clamp_max_(1.0)
            torchvision.utils.save_image(image, os.path.join(model_path, f"{frame_id}.png"))

        iter_end.record()
        progress_bar.set_postfix({"frame_id": f"{frame_id}"})
        progress_bar.update(1)
        torch.cuda.empty_cache()
