import os
import torch
import random
from tqdm import tqdm

from random import randint
from gaussian_renderer import render_facemask
from scene import Scene, GaussianModel
from argparse import Namespace
from torch_utils import misc

from utils.util import get_expon_weight_func
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr

from identity_learner_simple_styleGAN import IdentityLearner

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def identity_learning(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, list_ids, expName, unique_str, ALLcam=False, NofFrame = 5, path_to_output = "", scale=1.0, configs = {"COG": 1e10, "CS": 1e10, "TV": 1.0}, flag_densification=False):
    id_frame = 0
    scene = None
    max_iters = configs["max_iters"]
    batch_size = configs["batch_size"]
    path2data = configs["path2data"]
    training_args = configs["training_args"]
    max_blendshapes = configs["num_blendshape"]
    cs_reg_weight = configs["cs"]
    cog_reg_weight = configs["cog"]
    ore_weight = configs["ORE"]
    coeffreg_weight = configs["COEFF_REG"]
    volume_reg_weight = configs["VOLUME"]
    distortion_reg_weight = configs["DISTORTION"]
    triplane = configs["triplane"]
    laplace = configs["laplace"]
    num_planes = configs["num_planes"]
    xyz_method = configs["xyz_method"]
    tex_reso = configs["tex_reso"]
    anchorsubd_type = configs["anchorsubd_type"]
    offset_decoder_output_type = configs["offset_decoder_output_type"]
    delayed_noise_start_iter = configs["delayed_noise_start_iter"]
    num_train_subjects = configs["num_train_subjects"]
    xyz_lr = configs["xyz_lr"]
    gt_type = configs["gt_type"]
    # sparse_weight = configs["SPARSE"]
    m = configs["m"]
    K = configs["K"]
    embed_dim = configs["embed_dim"]
    
    eps = 1e-8
    opac_eps = 1e-5

    sh_degree = 0
    mini_batch_size = 8
    mini_batch_iters = 10

    first_iter = 0
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, max_iters*batch_size), desc="Training progress")
    first_iter += 1

    render_every = 100//mini_batch_iters # 10 where mini_batch_iters = 10
    delayed_noise_start_iter = delayed_noise_start_iter//mini_batch_iters
    save_every = 200//mini_batch_iters # 100 wherer mini_batch_iters = 10
    max_iters = max_iters // mini_batch_iters # 2000-> 200 where mini_batch_iters = 10


    limited_subject_names = list_ids

    if num_train_subjects > 128:
        # limited_subject_names = ['018', '032', '035', '037', '040', '042', '056', '058']
        test_subject_names = ['017', '030', '033', '038', '113', '124', '126', '211','250', '252', '251', '253', '256', '264', '266', '272', '289', '290', '260', '261', '262', '263', '267', '269', '270', '271']
        for tn in test_subject_names:
            if tn in limited_subject_names:
                limited_subject_names.remove(tn)
        # subjects which are not having tracking
        removed_subject_names = ['278', '280', '281', '288', '308']
    
        for rn in removed_subject_names:
            if tn in limited_subject_names:
                limited_subject_names.remove(rn)
    else:
        # limited_subject_names = ['018', '032', '035', '037', '040', '042', '056', '058']
        test_subject_names = ['017', '030', '033', '038', '113', '124', '126', '211']
        for tn in test_subject_names:
            limited_subject_names.remove(tn)

        # either open mouth, too long bang, or eye glasses
        removed_subject_names = ['032', '035', '043', '055', '067', '068', '070', '072', '073', '078', '079', '080', '081', '088', '090', '091', '092', '096', '107', '110', '116', '117', '120', '129',
                                '131', '138', '141', '142', '143', '144', '150', '151', '162', '164', '166', '168', '171', '173', '174', '176', '177', '180', '184', '185', '186', '193', '209', '241',
                                '246', '255', '371']
    
        for rn in removed_subject_names:
            limited_subject_names.remove(rn)


    if gt_type == "torso":
        # torso segmentation mask is not available in a view
        not_qualified_subjects =  ["057", "075", "078", "091", "092", "113", "145", "182", "183", "204", "259", "282", "295", "316", "327", "328"]

        for qn in not_qualified_subjects:
            if qn in limited_subject_names:
                limited_subject_names.remove(qn)

    # max_id: 248
    # toal 264
    limited_subject_names = limited_subject_names[:num_train_subjects]
    num_subjects = len(limited_subject_names)
    batch_size = num_subjects
    
    print("#train subjects: ", len(limited_subject_names))
    print("train subjects ids: ", limited_subject_names)
    print("#test subject ids: ", len(test_subject_names))
    print("batch size:", batch_size)
    # limited_subject_names = ['018', '032', '035']
    
    eval_dict = {"train": {"l1": {}, "psnr": {}, "lpips": {}, "ssim": {}},
                 "test": {"l1": {}, "psnr": {}, "lpips": {}, "ssim": {}}}

    for split in ["train", "test"]:
        for sn in limited_subject_names:
            eval_dict[split]['l1'].update({sn: 0})
            eval_dict[split]['psnr'].update({sn: 0})
            eval_dict[split]['lpips'].update({sn: 0})
            eval_dict[split]['ssim'].update({sn: 0})

    tb_writer, model_path = prepare_output_and_logger(dataset, unique_str, expName=expName, memo = "None", frame_counter = id_frame, path_to_output=path_to_output)
    coeffs_weight_func = lambda x: coeffreg_weight if x > 0 else 0.0

    limited_subjects_ids = []
    for i, id in enumerate(limited_subject_names):
        _subject_id = list_ids.index(id)
        limited_subjects_ids.append(_subject_id)
        # viewpoint_stacks_list.update({id: []})

    print(f"limited subject ids: {limited_subjects_ids}")

    IL = IdentityLearner(path2data=path2data, blendshape_size = max_blendshapes, K = K, m = m, embed_dim=embed_dim, tex_reso=tex_reso, anchorsubd_type = anchorsubd_type, subject_names=limited_subject_names, sh_degree=sh_degree, num_planes = num_planes, device="cuda", triplane=triplane, laplace=laplace, xyz_method = xyz_method, offset_decoder_output_type = offset_decoder_output_type, reload_subjects=True).cuda()
    IL.training_setup(training_args={"xyz_lr": xyz_lr})

    gaussians_list = {}
    scenes_list = {}
    viewpoint_stacks_list = {}

    final_iter = False
    final_step_id = 231


    average_l1 = 0
    average_lpips = 0
    average_ssim = 0
    average_psnr = 0

    eval_flag = False


    for iteration in range(first_iter, max_iters+1):
        eval_flag = False
        if iteration >= final_step_id:
            final_iter = True

        accum_photometric_loss = 0.0
        current_batch_ids = random.sample(limited_subjects_ids, batch_size)
    
        # current_batch_ids = limited_subjects_ids
        print("current batch: ", current_batch_ids)
        subject_ids = [list_ids[i] for i in current_batch_ids]

        print("current_batch_ids: ", current_batch_ids)
        print("subject_ids: ", subject_ids)
        pre_loss = 0.0

        mini_batche_subject_ids = [] #16-sets of mini-batch
        last_id = subject_ids[-1]
        
        for i in range(batch_size//mini_batch_size):
            _mini_batch = []
            _mini_batch = subject_ids[i*mini_batch_size:i*mini_batch_size+mini_batch_size]
            mini_batche_subject_ids.append(_mini_batch)


        if tb_writer:
            tb_writer.add_histogram(f"mu_scale", IL.mu_scale.prod(dim=1), iteration)
            tb_writer.add_histogram(f"mu_opac", IL.mu_opac, iteration)
            tb_writer.add_histogram(f"mu_rot", IL.mu_scale.prod(dim=1), iteration)

        
        for mb_index, mb in enumerate(mini_batche_subject_ids):

            print("current mb: ", mb)

            for i, subject_id in enumerate(mb):
                _gaussians = GaussianModel(sh_degree=sh_degree)
                gaussians_list.update({subject_id: _gaussians})
                _scene=Scene(dataset, expName = expName, ALLcam=True, frame_counter=id_frame, subject_id=subject_id, scale=scale, gt_type=gt_type)
                scenes_list.update({subject_id: _scene})
                _subject_id = list_ids.index(subject_id)
                viewpoint_stacks_list.update({subject_id: _scene.getTrainCameras().copy()})
                del _gaussians
                del _scene
                torch.cuda.empty_cache()

            active_blendshape=max_blendshapes

            for mb_iter in range(mini_batch_iters):
                # get UVs
                for index_in_batch, subject_name in enumerate(mb):
                    if delayed_noise_start_iter > 0 and delayed_noise_start_iter <= iteration:
                        current_xyz, current_fdc, current_frest, current_scales, current_rots, current_opacities= IL(batch_names=[subject_name], noise_mode='random')
                    else:
                        current_xyz, current_fdc, current_frest, current_scales, current_rots, current_opacities= IL(batch_names=[subject_name]) #noise_mode='const' by default
                    
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
                        
                    loss = 0.0
                    tv_loss = 0.0
                    photometric_loss=0.0
                    ORE = 0.0
                    cog_reg_term = 0.0
                    cs_reg_term = 0.0
                    coeffs_reg = 0.0
                    vgg_loss = 0.0

                    additional_losses = {}
                    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    bg = torch.rand((3), device="cuda") if opt.random_background else background

                    print("render: ", subject_name)
                    id_frame=0
                    
                    current_gaussians = gaussians_list[subject_name]
                    current_scene = scenes_list[subject_name]
                    current_scene.set_gaussians(gaussians=current_gaussians, input_format='learnable_blenshape', index_in_batch=0, xyz=current_xyz, fdc=current_fdc, frest=current_frest, scale=current_scales, rot=current_rots, opac=current_opacities)

                    if cog_reg_weight > 0.0:
                        ref_cog = torch.from_numpy(current_scene.flame_tracking_pcd.points).float().cuda()

                    iter_start = torch.cuda.Event(enable_timing = True)
                    iter_end = torch.cuda.Event(enable_timing = True)

                    iter_start.record()

                    # Pick a random Camera
                    if not viewpoint_stacks_list[subject_name]:
                        viewpoint_stacks_list[subject_name] = current_scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stacks_list[subject_name].pop(randint(0, len(viewpoint_stacks_list[subject_name])-1))

                    # Render
                    if (iteration - 1) == debug_from:
                        pipe.debug = True

                    render_pkg = render_facemask(viewpoint_cam, current_gaussians, pipe, bg)
                    image, final_G_scales = render_pkg["render"], render_pkg["scaling"]

                    gt_image = viewpoint_cam.original_image.cuda()
                    # pred (B, C, H, W), gt (B, C, H, W)

                    # Loss
                    Ll1 = l1_loss(image, gt_image)
                    ssim_loss = 1.0 - ssim(image, gt_image)


                    photometric_loss = (1.0 - opt.lambda_dssim)*Ll1 + opt.lambda_dssim*ssim_loss #+ lpips_score
                    loss = photometric_loss

                    if cog_reg_weight > 0.0:
                        cog_reg_term = ((((ref_cog-current_xyz)**2).sum(dim=1))**2).mean()
                        loss = loss + cog_reg_term * cog_reg_weight

                    if cs_reg_weight > 0.0:
                        covariance_scale = final_G_scales
                        cs_reg_term = (((covariance_scale**2).sum(dim=1))**2).mean()
                        loss = loss + cs_reg_term * cs_reg_weight

                    if coeffreg_weight > 0.0:
                        coeffs_reg = (IL.coeffs[limited_subject_names.index(subject_name)]**2).mean()
                        loss = loss + coeffs_reg * coeffreg_weight

                    if ore_weight > 0.0:
                        INN = torch.matmul(IL.embeddings.reshape(IL.num_blendshape, -1), IL.embeddings.reshape(-1, IL.num_blendshape))
                        ORE = torch.norm(torch.eye(INN.shape[0]).cuda() - INN)
                        loss = loss + ORE*ore_weight

                    if volume_reg_weight > 0.0:
                        scaling_reg = final_G_scales.prod(dim=1).mean()
                        loss = loss + volume_reg_weight * scaling_reg                        
                        
                    loss = loss * (len(limited_subject_names)**-1)

                    accum_photometric_loss+=loss
                    IL.optimizer.zero_grad(set_to_none = True)
                    
                    # compute gradients via backprop
                    loss.backward()

                    # set the nan to num
                    with torch.no_grad():
                        for param in IL.parameters():
                            if param.grad is not None:
                                misc.nan_to_num(param.grad, nan=0.0, posinf=1e9, neginf=-1e9, out=param.grad)

                    # update parameters
                    IL.optimizer.step()
                    iter_end.record()

                    with torch.no_grad():
                        test_subject_ids = []
                        test_subject_ids.append(subject_name)
                        
                        if subject_name == last_id:
                            test_subject_ids.append(-1)

                        for i, test_subject_id in enumerate(test_subject_ids):
                            if test_subject_id == -1:
                                if (iteration % (render_every) == 0 and mb_iter == (mini_batch_iters-1)) or iteration == max_iters-1:
                                    pass
                            else:
                                additional_losses = {"ave_loss": loss, "L1": Ll1, "ssim": ssim_loss, "p_loss": photometric_loss, "active_b": active_blendshape, "cog_reg": cog_reg_term, "coeffs_weight": coeffs_weight_func(iteration)}
                                progress_bar.set_postfix({"id": f"{test_subject_id}", "l1": f"{Ll1}", "ssim": f"{ssim_loss}", "p_loss": f"{photometric_loss}", "xyz_method": f"{xyz_method}", "gt_type": f"{gt_type}", "cs_reg_weight": f"{cs_reg_weight}"})
                                progress_bar.update(1)

                                if (iteration % (render_every) == 0 and mb_iter == (mini_batch_iters-1)) or iteration == max_iters-1:
                                    eval_flag = True
                                    im_save_path = os.path.join(model_path, "im_result")
                                    training_report(eval_dict, tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), additional_losses, testing_iterations, current_scene, render_facemask, (pipe, background), expName=expName, max_iter= max_iters, frame_counter=id_frame, im_save_path=im_save_path, subject_id = test_subject_id, IL=IL)
                                    tb_writer.add_histogram(f"{test_subject_id}/coeffs", IL.coeffs[limited_subject_names.index(test_subject_id)], iteration)
                                    tb_writer.add_histogram(f"{test_subject_id}/colors/RED", current_fdc[..., 0], iteration)
                                    tb_writer.add_histogram(f"{test_subject_id}/colors/GREEN", current_fdc[..., 1], iteration)
                                    tb_writer.add_histogram(f"{test_subject_id}/colors/BLUE", current_fdc[..., 2], iteration)

                        if (iteration == max_iters-1 or iteration % (save_every) == 0) and last_id==subject_name:
                            eval_flag = True
                            additional_losses = {"ave_loss": accum_photometric_loss, "p_loss": accum_photometric_loss}
                            im_save_path = os.path.join(model_path, "im_result")
                            training_report(eval_dict, tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), additional_losses, testing_iterations, current_scene, render_facemask, (pipe, background), expName=expName, max_iter= max_iters, frame_counter=id_frame, im_save_path=im_save_path, subject_id = subject_name, IL=IL)
                            tb_writer.add_histogram(f"{subject_name}/coeffs", IL.coeffs[limited_subject_names.index(subject_name)], iteration)
                            print("\n[ITER {}] Saving Checkpoint".format(iteration))
                            torch.save(IL, current_scene.model_path + f"/blendshape_{int(iteration*mini_batch_iters)}.pth")
                    
                    del render_pkg
                    del image
                    del gt_image
                    del final_G_scales
                    del viewpoint_cam

                    del current_xyz
                    del current_fdc
                    del current_frest
                    del current_scales
                    del current_rots
                    del current_opacities

                    del current_gaussians
                    del current_scene

                    torch.cuda.empty_cache()

            gaussians_list.clear()
            scenes_list.clear()
            viewpoint_stacks_list.clear()
            torch.cuda.empty_cache()

        if eval_flag or final_iter:
            # average among subjects in current mini-batch
            sum_l1 = {"train": 0, "test": 0}
            sum_lpips = {"train": 0, "test": 0}
            sum_ssim = {"train": 0, "test": 0}
            sum_psnr = {"train": 0, "test": 0}

            sn_counter = {"train": 0, "test":0}
            ave_target_ids = []

            ave_target_ids = limited_subject_names

            for split in ["train", "test"]:
                for sn in ave_target_ids:
                    sum_l1[split] += eval_dict[split]['l1'][sn]
                    sum_lpips[split] += eval_dict[split]['lpips'][sn]
                    sum_psnr[split] += eval_dict[split]['psnr'][sn]
                    sum_ssim[split] += eval_dict[split]['ssim'][sn]
                    sn_counter[split] = sn_counter[split] + 1
                    
                average_l1 = sum_l1[split]/sn_counter[split]
                average_lpips = sum_lpips[split]/sn_counter[split]
                average_ssim = sum_ssim[split]/sn_counter[split]
                average_psnr = sum_psnr[split]/sn_counter[split]

                tb_writer.add_scalar(f'metrics/{split}/ave_l1', average_l1, iteration)
                tb_writer.add_scalar(f'metrics/{split}/ave_lpips', average_lpips, iteration)
                tb_writer.add_scalar(f'metrics/{split}/ave_ssim', average_ssim, iteration)
                tb_writer.add_scalar(f'metrics/{split}/ave_l1_psnr', average_psnr, iteration)

        if final_iter:
            break

    return average_l1, average_psnr, average_lpips, average_ssim, eval_dict
    

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

def training_report(eval_dict, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, additional_losses, testing_iterations, scene : Scene, renderFunc, renderArgs, expName, max_iter, frame_counter, im_save_path, subject_id, IL):
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

    l1_test, psnr_test, lpips_test = 0.0, 0.0, 0.0

    # Report test and samples of training set
    with torch.no_grad():
        if iteration < max_iter-1:
            train_cam_id = 0
            validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                                 {'name': 'train', 'cameras' : [scene.getTrainCameras()[train_cam_id]]})
            
            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    lpips_test = 0.0
                    ssim_test = 0.0

                    l1_train = 0.0
                    psnr_train = 0.0
                    lpips_train = 0.0
                    ssim_train = 0.0 
                    for idx, viewpoint in enumerate(config['cameras']):                    
                        gt_image = viewpoint.original_image.to("cuda").clamp_max_(1.0)
                        image = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"].clamp_max_(1.0)

                        tb_writer.add_images(f"{subject_id}/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f"{subject_id}/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                        if config['name'] == "test":
                            l1_test += l1_loss(image, gt_image).mean().float().cpu().numpy()
                            psnr_test += psnr(image, gt_image).mean().float().cpu().numpy()
                            ssim_test += ssim(image, gt_image).mean().float().cpu().numpy()
                            # lpips_test += LPIPS(image, gt_image).mean().double()
                        else:
                            l1_train += l1_loss(image, gt_image).mean().float().cpu().numpy()
                            psnr_train += psnr(image, gt_image).mean().float().cpu().numpy()
                            ssim_train += ssim(image, gt_image).mean().float().cpu().numpy()
                            # lpips_test += LPIPS(image, gt_image).mean().double()
                        del gt_image
                        del image
                        torch.cuda.empty_cache()

                    if config['name'] == "test":
                        psnr_test /= len(config['cameras'])
                        l1_test /= len(config['cameras'])
                        ssim_test /= len(config['cameras'])
                        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, lpips_test, ssim_test))

                        eval_dict[config['name']]["l1"][subject_id] = l1_test
                        eval_dict[config['name']]["psnr"][subject_id] = psnr_test
                        eval_dict[config['name']]["lpips"][subject_id] = lpips_test
                        eval_dict[config['name']]["ssim"][subject_id] = ssim_test
                    
                        if tb_writer:
                            tb_writer.add_scalar(f"{subject_id}/" + config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                            tb_writer.add_scalar(f"{subject_id}/" + config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                            tb_writer.add_scalar(f"{subject_id}/" + config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                            tb_writer.add_scalar(f"{subject_id}/" + config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    else:
                        psnr_train /= len(config['cameras'])
                        l1_train /= len(config['cameras'])
                        ssim_train /= len(config['cameras'])
                        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS {} SSIM {}".format(iteration, config['name'], l1_train, psnr_train, lpips_train, ssim_train))

                        eval_dict[config['name']]["l1"][subject_id] = l1_train
                        eval_dict[config['name']]["psnr"][subject_id] = psnr_train
                        eval_dict[config['name']]["lpips"][subject_id] = lpips_train
                        eval_dict[config['name']]["ssim"][subject_id] = ssim_train
                    
                        if tb_writer:
                            tb_writer.add_scalar(f"{subject_id}/" + config['name'] + '/loss_viewpoint - l1_loss', l1_train, iteration)
                            tb_writer.add_scalar(f"{subject_id}/" + config['name'] + '/loss_viewpoint - psnr', psnr_train, iteration)
                            tb_writer.add_scalar(f"{subject_id}/" + config['name'] + '/loss_viewpoint - lpips', lpips_train, iteration)
                            tb_writer.add_scalar(f"{subject_id}/" + config['name'] + '/loss_viewpoint - ssim', ssim_train, iteration)
            if tb_writer:
                # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
                tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

