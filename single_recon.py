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
import random
import torchvision
from torch import nn
from torch.nn import functional as F

import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import original_render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.util import get_expon_weight_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.general_utils import inverse_sigmoid
# from identity_learner_simple_styleGAN import IdentityTracker
# from utils.vgg_loss import VGGLoss
# vgg = VGGLoss().to("cuda")

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize = True).cuda()

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
import matplotlib.pyplot as plt
import trimesh
from heat_kernel import HeatKernel
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

eliminated_subjects = ["018", '252', '251']


def identity_tracking(dataset, opt, pipe, testing_iterations, path2blendshape, list_ids, expName, unique_str, path_to_output = "", scale=1.0, configs = {"COG": 1e10, "CS": 1e10, "TV": 1.0}):
    """
    input:
        path2blendshape: PATH to blendshape .pth

    """
    l1_list, lpips_list, ssim_list, psnr_list = [], [], [], []

    id_frame = 0
    max_iters = configs["max_iters"]
    path2data = configs["path2data"]
    path2blendshape = configs["path2blendshape"]
    coeffs_reg_weight = configs["COEFF_REG"]
    optimizer_coeffs = configs["optimizer"]
    set_id = configs["set"]
    render_numGauss = configs["render_numGauss"]
    subd = configs["subd"]

    limited_subject_names = None
    if set_id == 0:
        limited_subject_names = ['017', '018', '030', '033', '038', '113', '124', '126', '211']
    elif set_id == 1:
        limited_subject_names = ['018', '250', '252', '251', '253', '256', '264', '266', '272', '289', '290']
    elif set_id == 2:
        limited_subject_names = ['018', '252', '251', '260', '261', '262', '263', '267', '269', '270', '271']
    elif set_id == -1:
        # limited_subject_names = ['030', '038', '113']
        mini_batches = [
                        ['017', '030', '033', '038', '113', '124', '126', '211'],
                        ['250', '252', '251', '253', '256', '264', '266', '272', '289', '290'],
                        ['252', '251', '260', '261', '262', '263', '267', '269', '270', '271']
                        ]
    elif set_id == 101:
        limited_subject_names = ["flame1", "flame2"]
        scale = 1.0
        
    if set_id != -1:
        mini_batches = [limited_subject_names]

    eval_dict = {"l1": {}, "psnr": {}, "lpips": {}, "ssim": {}}

    for sn in limited_subject_names:
        if sn not in eliminated_subjects:
            eval_dict['l1'].update({sn: None})
            eval_dict['psnr'].update({sn: None})
            eval_dict['lpips'].update({sn: None})
            eval_dict['ssim'].update({sn: None})

    average_l1 = 0.0
    average_lpips = 0.0
    average_psnr = 0.0
    average_ssim = 0.0

    for mid, mb_subjects in enumerate(mini_batches):

        print("current mb_subject list: ", mb_subjects)

        for id_subject, subject_name in enumerate(mb_subjects):
            limited_subject_names = [subject_name]

            first_iter = 0
            viewpoint_stack = None
            progress_bar = tqdm(range(first_iter, max_iters*len(limited_subject_names)), desc="Training progress")
            first_iter += 1

            tb_writer, model_path = prepare_output_and_logger(dataset, unique_str, expName=expName, memo = "None", frame_counter = id_frame, path_to_output=path_to_output)
            l1_test, psnr_test, lpips_test = 0.0, 0.0, 0.0

            limited_subjects_ids = []
            viewpoint_stacks_list = {}
            gaussians_list = {}
            scenes_list = {}

            for i, subject_id in enumerate(limited_subject_names):
                _gaussians = GaussianModel(sh_degree=3)

                if set_id != 101:
                    _scene=Scene(dataset, expName = expName, ALLcam=True, frame_counter=id_frame, subject_id=subject_id, scale=scale, Center=True, render_numGauss = render_numGauss, subd = subd, Render_Interp_CAM=True)
                else:
                    _scene=Scene(dataset, expName = expName, ALLcam=True, frame_counter=id_frame, subject_id=subject_id, scale=scale, Center=True, render_numGauss = render_numGauss, subd = subd, Render_Interp_CAM=False)

                _scene.set_gaussians(gaussians=_gaussians, input_format='pcd')

                _gaussians.training_setup(opt)

                gaussians_list.update({subject_id: _gaussians})
                scenes_list.update({subject_id: _scene})
                _subject_id = list_ids.index(subject_id)
                viewpoint_stacks_list.update({subject_id: _scene.getTrainCameras().copy()})
                limited_subjects_ids.append(_subject_id)

            print(f"limited subject ids: {limited_subjects_ids}")


            sh_degree = 0
            pre_active_blendshape = 0
            next_start_id_im = 0

            path2gaussianprop_folder = os.path.join(model_path, "gaussian_props")

            if not os.path.exists(path2gaussianprop_folder):
                os.makedirs(path2gaussianprop_folder, exist_ok=True)

            # adaptive_weight_coeffs = torch.ones(IT.num_blendshape).float().cuda()

            for iteration in range(first_iter, max_iters+1):
                print("subject_ids: ", limited_subject_names)

                # get UVs
                for index_in_batch, subject_name in enumerate(limited_subject_names):
                    coeffs_reg = 0.0
                    photometric_loss = 0.0
                    loss = 0.0

                    iter_start = torch.cuda.Event(enable_timing = True)
                    iter_end = torch.cuda.Event(enable_timing = True)
                    iter_start.record()

                    # print("current blenshape: ", current_blenshapes.shape)
                    photometric_loss=0.0

                    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    bg = torch.rand((3), device="cuda") if opt.random_background else background

                    print("render: ", subject_name)
                    
                    # current_gaussians = GaussianModel(sh_degree=sh_degree)
                    # current_scene=Scene(dataset, expName = expName, ALLcam=True, frame_counter=id_frame, subject_id=subject_name, scale=scale, Center=True)

                    current_gaussians = gaussians_list[subject_name]
                    current_gaussians.update_learning_rate(iteration)

                    current_scene = scenes_list[subject_name]

                    # Pick a random Camera
                    viewpoint_stack = viewpoint_stacks_list[subject_name]
                    if not viewpoint_stack:
                        viewpoint_stack = current_scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

                    render_pkg = original_render(viewpoint_cam, current_gaussians, pipe, bg)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                    # print("radii:", radii)
                    # Loss
                    gt_image = viewpoint_cam.original_image.cuda()

                    Ll1 = l1_loss(image, gt_image)
                    ssim_loss = 1.0 - ssim(image, gt_image)

                    photometric_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (ssim_loss)

                    loss = photometric_loss

                    # update parameters
                    current_gaussians.optimizer.zero_grad(set_to_none = True)
                    loss.backward()
                    iter_end.record()
                    current_gaussians.optimizer.step()

                    with torch.no_grad():
                        progress_bar.set_postfix({"id": f"{subject_name}", "p_loss": f"{photometric_loss}"})
                        progress_bar.update(1)    
                        additional_losses = {"p_loss": photometric_loss}
                        l1_test = 0.0
                        lpips_test = 0.0
                        ssim_test = 0.0
                        psnr_test = 0.0

                        if iteration % (2) == 0:
                            # next_start_id_im = render_camInterpolation(scene=current_scene, gaussians = current_gaussians ,pipe=pipe, model_path=model_path, subject_name=subject_name, sh_degree=sh_degree, dataset=dataset, expName=expName, frame_counter=id_frame, scale=scale, ALLcam=True, Center=False, next_start_id_im=next_start_id_im)
                            # im_save_path = os.path.join(model_path, "im_result")
                            training_report(eval_dict, tb_writer, iteration, Ll1, photometric_loss, l1_loss, iter_start.elapsed_time(iter_end), additional_losses, testing_iterations, current_scene, original_render, (pipe, background), expName=expName, max_iter= max_iters, frame_counter=id_frame, im_save_path=None, subject_id = subject_name)                            
                            # tb_writer.add_histogram(f"{subject_name}/coeffs", IT.coeffs[index_in_batch], iteration)

                        if iteration == max_iters-1:
                            render_scene=Scene(dataset, expName = expName, ALLcam=True, frame_counter=id_frame, subject_id="024", scale=scale, Center=True, render_numGauss = render_numGauss, subd = subd, Render_Interp_CAM=True)
                            next_start_id_im = render_camInterpolation(scene=render_scene, gaussians = current_gaussians, pipe=pipe, model_path=model_path, subject_name=subject_name, sh_degree=sh_degree, dataset=dataset, expName=expName, frame_counter=id_frame, scale=scale, ALLcam=True, Center=False, next_start_id_im=next_start_id_im)
                            # for subject_name in limited_subject_names:
                            # im_save_path = os.path.join(model_path, "im_result")
                            # render_camInterpolation(IT = IT, pipe=pipe, model_path=model_path, subject_name=subject_name, sh_degree=sh_degree, dataset=dataset, expName=expName, frame_counter=id_frame, subject_id=subject_id, scale=scale, ALLcam=True, Center=False, iterations=iteration)
                            training_report(eval_dict, tb_writer, iteration, Ll1, photometric_loss, l1_loss, iter_start.elapsed_time(iter_end), additional_losses, testing_iterations, current_scene, original_render, (pipe, background), expName=expName, max_iter= max_iters, frame_counter=id_frame, im_save_path=None, subject_id = subject_name)
                            # current_scene.save(iteration, subject_id = subject_name, test=True)
                            path2subject_gaussianprop = os.path.join(path2gaussianprop_folder, subject_name)
                            if not os.path.exists(path2subject_gaussianprop):
                                os.makedirs(path2subject_gaussianprop, exist_ok=True)

                            current_gaussians.save_GaussianProp(path=path2subject_gaussianprop)

        print("eval_dict: ", eval_dict)
        # average among subjects in current mini-batch
        sum_l1 = 0
        sum_lpips = 0
        sum_ssim = 0
        sum_psnr = 0

        sn_counter = 0
        ave_target_ids = []

        if mid == len(mini_batches)-1: # 0=1-1
            "take average of all subjects"
            for mids in mini_batches:
                for sn in mids:
                    ave_target_ids.append(sn)
        else:
            "take average of subjects in minibatch"
            ave_target_ids = limited_subject_names

        for sn in ave_target_ids:
            if sn not in eliminated_subjects:
                sum_l1 += eval_dict['l1'][sn]
                sum_lpips += eval_dict['lpips'][sn]
                sum_psnr += eval_dict['psnr'][sn]
                sum_ssim += eval_dict['ssim'][sn]
                sn_counter = sn_counter + 1
                
        average_l1 = sum_l1/sn_counter
        average_lpips = sum_lpips/sn_counter
        average_ssim = sum_ssim/sn_counter
        average_psnr = sum_psnr/sn_counter

        if mid == len(mini_batches)-1:
            tb_writer.add_scalar('summary/metrics/ave_l1', average_l1, mid)
            tb_writer.add_scalar('summary/metrics/ave_lpips', average_lpips, mid)
            tb_writer.add_scalar('summary/metrics/ave_ssim', average_ssim, mid)
            tb_writer.add_scalar('summary/metrics/ave_l1_psnr', average_psnr, mid)
        else:
            tb_writer.add_scalar('metrics/ave_l1', average_l1, mid)
            tb_writer.add_scalar('metrics/ave_lpips', average_lpips, mid)
            tb_writer.add_scalar('metrics/ave_ssim', average_ssim, mid)
            tb_writer.add_scalar('metrics/ave_l1_psnr', average_psnr, mid)


        return average_l1, average_lpips, average_ssim, average_psnr, eval_dict
        

def prepare_output_and_logger(args, _unique_id=None, expName="", memo = "", frame_counter=0, path_to_output=""):
    unique_str = _unique_id
    args.model_path = os.path.join(path_to_output, unique_str, expName, str(frame_counter))

    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))) + "\n" + memo)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer, args.model_path

def render_camInterpolation(scene, gaussians, pipe, model_path, subject_name, sh_degree, dataset, expName, frame_counter, scale, ALLcam=True, Center=True, next_start_id_im=0):
    
    with torch.no_grad():

        im_save_path = os.path.join(model_path, f"{subject_name}_interp")
        if not os.path.exists(im_save_path):
            os.makedirs(im_save_path, exist_ok=True)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        viewpoint_stack = scene.getInterpCameras().copy()
        renderArgs = (pipe, background)

        for i, cam in enumerate(viewpoint_stack):
            image = original_render(cam, gaussians, *renderArgs)["render"].clamp_max_(1.0)
            torchvision.utils.save_image(image, os.path.join(im_save_path, f"{i}.png"))
            next_start_id_im += 1
            del image
            torch.cuda.empty_cache()
        
        del bg_color
        del background
        del viewpoint_stack
        del renderArgs
        del scene
        del gaussians
        torch.cuda.empty_cache()

        return next_start_id_im


def training_report(eval_dict, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, additional_losses, testing_iterations, scene : Scene, renderFunc, renderArgs, expName, max_iter, frame_counter, im_save_path, subject_id):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        if len(additional_losses.keys()) != 0:
            for key in additional_losses.keys():
                if key.endswith("attention"):
                    # print(additional_losses[key].shape)
                    print(additional_losses[key])
                    tb_writer.add_histogram('scene/'+str(key), additional_losses[key], iteration)
                else:
                    tb_writer.add_scalar('train_loss_patches/'+str(key), additional_losses[key], iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    l1_list = []
    psnr_list = []
    ssim_list = []
    lpips_list = []

    with torch.no_grad():
        if iteration <= max_iter-1:
            validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                                {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    lpips_test = 0.0
                    ssim_test = 0.0
                    for idx, viewpoint in enumerate(config['cameras']):
                        # gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    
                        gt_image = viewpoint.original_image.to("cuda").clamp_max_(1.0)
                        image = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"].clamp_max_(1.0)
                        
                        # resize for interpretable metrics
                        # 3, H, W

                        if "flame" not in subject_id:
                            resized_gt_image = gt_image[:, 302:1302, :]
                            resized_image = image[:, 302:1302, :]
                        else:
                            resized_gt_image = gt_image
                            resized_image = image

                        print("gt_image: ", resized_gt_image.shape)
                        print("image: ", resized_image.shape)
                        
                        # print("gt_image min/max: ", torch.min(gt_image), torch.max(gt_image))
                        # print("image min/max: ", torch.min(image), torch.max(image))
                        # image = torch.clamp(gt_image + image, 0.0, 1.0)
                        # print(image.shape)
                        # print(gt_image.shape)

                        # if iteration == max_iter:
                            # plt.subplot(1, 2, 1)
                            # plt.imshow(image.detach().cpu().permute(1,2,0).numpy())
                            # plt.title(f"Rendered image at {iteration}")
                            # plt.subplot(1, 2, 2)
                            # plt.imshow(gt_image.detach().cpu().permute(1, 2, 0).numpy())
                            # plt.title(f"GT at {iteration}")
                            # plt.suptitle(f"Camera ID: {viewpoint.colmap_id}")
                            # path_to_plotImage = im_save_path
                            # if not os.path.exists(path_to_plotImage):
                            #     os.mkdir(path_to_plotImage)
                            # if not os.path.exists(os.path.join(path_to_plotImage, str(frame_counter))):
                            #     os.mkdir(os.path.join(path_to_plotImage, str(frame_counter)))
                            # print(viewpoint.colmap_id)
                            # print("output path: ", os.path.join(path_to_plotImage, str(frame_counter)))
                            # plt.savefig(os.path.join(path_to_plotImage, str(frame_counter), str(viewpoint.colmap_id)+ "_" + str(iteration) +'.png'))
                        # if tb_writer and (idx < 5):
                        #     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        #     if iteration == testing_iterations[0]:
                        #         tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                        tb_writer.add_images(f"{subject_id}/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), resized_image[None], global_step=iteration)
                        tb_writer.add_images(f"{subject_id}/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), resized_gt_image[None], global_step=iteration)

                        _l1 = l1_loss(resized_image, resized_gt_image).mean().double()
                        _psnr = psnr(resized_image, resized_gt_image).mean().double()
                        _ssim = ssim(resized_image, resized_gt_image).mean().double()
                        _lpips = LPIPS(resized_image[None], resized_gt_image[None])

                        if config['name']=="test":
                            l1_test += _l1
                            psnr_test += _psnr
                            ssim_test += _ssim
                            lpips_test += _lpips

                        if tb_writer:
                            tb_writer.add_scalar(f"{subject_id}/"+config['name'] + "_view_{}/l1".format(viewpoint.image_name), _l1, iteration)
                            tb_writer.add_scalar(f"{subject_id}/"+config['name'] + "_view_{}/psnr".format(viewpoint.image_name), _psnr, iteration)
                            tb_writer.add_scalar(f"{subject_id}/"+config['name'] + "_view_{}/lpips".format(viewpoint.image_name), _lpips, iteration)
                            tb_writer.add_scalar(f"{subject_id}/"+config['name'] + "_view_{}/ssim".format(viewpoint.image_name), _ssim, iteration)
                
                        # shape should be: B, 3, h, w
                        
                        # print("LPIPS")
                        # print(gt_image.shape)
                        # print(resized_image.shape)
                    if config['name']=="test":
                        psnr_test /= len(config['cameras'])
                        l1_test /= len(config['cameras'])
                        ssim_test /= len(config['cameras'])
                        lpips_test /= len(config['cameras'])

                        if subject_id not in eliminated_subjects:
                            eval_dict["l1"][subject_id] = l1_test
                            eval_dict["psnr"][subject_id] = psnr_test
                            eval_dict["lpips"][subject_id] = lpips_test
                            eval_dict["ssim"][subject_id] = ssim_test

                    # if iteration == max_iter:
                    #     lpips_test /=len(config['cameras'])
                    #     # lpips_test = lpips_test.detach().cpu().numpy() + 0

                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, lpips_test, ssim_test))
                    if tb_writer:
                        tb_writer.add_scalar(f"{subject_id}/" + config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar(f"{subject_id}/" + config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                        tb_writer.add_scalar(f"{subject_id}/" + config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                        tb_writer.add_scalar(f"{subject_id}/" + config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
            
            if tb_writer:
                # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
                tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            torch.cuda.empty_cache()

    return l1_test, lpips_test, ssim_test, psnr_test

