import os
import torch
import sys
import numpy as np
from datetime import datetime
from styleGAN_identity_learning_large import identity_learning

from utils.dataset_handler import Filehandler
from utils.general_utils import safe_state

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

MAX_TRAIN_SUBJECTS = 264

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    # load hyper parameters
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    max_iterations =  30_000
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(np.arange(0, max_iterations, 500)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=list(np.arange(0, max_iterations, 500)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--ALLcam", action='store_true', default=False)
    parser.add_argument("--flag_densification", action='store_true', default=False)
    parser.add_argument("--recompute_laplacian", type = bool , default=False)
    parser.add_argument("--K", type=int, default=12)
    parser.add_argument("--num_train_subjects", type=int, default=128)
    parser.add_argument("--max_iters", type=int, default=30_000)
    parser.add_argument("--IL_name", type=str, default="")
    parser.add_argument("--anchorsubd_type", type=str, default="subd0")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--tex_reso", type=int, default=64)
    parser.add_argument("--num_blendshape", type=int, default=80)
    parser.add_argument("--xyz_lr", type=float, default=0.00000016)
    parser.add_argument("--xyz_method", type=str, default="default")
    parser.add_argument("--gt_type", type=str, default="")
    parser.add_argument("--cs_reg_weight", type=float, default=0.0)


    args = parser.parse_args(sys.argv[1:])
    args.iterations = max_iterations
    args.random_background = False
    args.test_iterations.append(0)
    args.test_iterations.append(args.iterations)
    args.save_iterations.append(0)
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)

    if args.num_train_subjects > MAX_TRAIN_SUBJECTS:
        print("There are no more than 264 (24 #test, 240 #train, 5 useless) subjects for training")
        exit()

    path_to_dataset = '/mnt/hdd/dataset/269-single-timestep-EXP-1-head'
    EXP = "EXP-1-head"

    now = datetime.now()
    date = now.strftime("%d%m")
    time = now.strftime("%H%M")
    title_experiment = f"AFTER_GR_{args.num_train_subjects}subjects_styleGAN_noisemode_const_common_plane_K{args.K}tex{args.tex_reso}embed{args.embed_dim}blendshape{args.num_blendshape}_anchor_{args.anchorsubd_type}_xyzmethod_{args.xyz_method}_learnableMEAN_{args.max_iters}iters_{args.IL_name}_gt_type{args.gt_type}"+time+"_"+date
    # title_experiment = "ablation0406_TEST_"+time+"_"+date
    path2exp_out = os.path.join("/mnt/hdd/GRFinal", title_experiment)
    scale = 0.5

    list_ids, list_paths = Filehandler.dirwalker_InFolder(path_to_folder = path_to_dataset, prefix="")
    list_ids = list_ids[:-1]
    num_subjects = len(list_ids)
    path2refmesh = "/home/kato/Photorealistic-3DMM/DeformationLearning_3DGS/samples/flame_meshes/flame_facemask_subd.ply"
    training_args = {}
    # max K: 15
    configs = {"max_iters":args.max_iters, "gt_type": args.gt_type, "num_train_subjects": args.num_train_subjects, "num_planes": 1, "batch_size": 25, "num_blendshape":args.num_blendshape, "embed_dim":args.embed_dim, "tex_reso": args.tex_reso, "anchorsubd_type": args.anchorsubd_type, "K": args.K, "xyz_lr": args.xyz_lr, "cs": args.cs_reg_weight, "cog": 0.0, "ORE": 0.0, "COEFF_REG": 0.0, "VOLUME": 0.0, "DISTORTION": 0.0, "m": 1, "delayed_noise_start_iter": -1, "triplane": False, "laplace": False, "xyz_method": args.xyz_method, "offset_decoder_output_type": "XYZ", "path2data": path_to_dataset, "path2refmesh": path2refmesh, "training_args": training_args}
    str_config=str(configs)

    path_to_output = os.path.join(path2exp_out)
    os.makedirs(path_to_output, exist_ok = True)
    # path_to_output = path2exp_out

    print(configs)
    with open(os.path.join(path2exp_out, "configs.txt"), "w") as f:
        f.write(str(configs))

    l1_test_list = []
    psnr_test_list = []
    lpips_test_list = []
    average_l1 = 0
    average_lpips = 0
    average_psnr = 0


    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    _ALLcam = args.ALLcam
    numOfFrames = 1

    l1_test, psnr_test, lpips_test, ssim_test, eval_dict = identity_learning(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, list_ids = list_ids, expName=EXP, unique_str="", ALLcam=_ALLcam, NofFrame=numOfFrames, path_to_output=path_to_output, scale=0.5, configs = configs, flag_densification=args.flag_densification)
    print(f"FINAL l1 {l1_test}, PSNR {psnr_test} LPIPS {lpips_test} SSIM {ssim_test}")

    final_test_result = {"config": str_config, "num_sub": num_subjects, "average_L1": l1_test, "average_PSNR": psnr_test, "average_LPIPS": lpips_test, "average_ssim": ssim_test}
    with open(os.path.join(path_to_output, "summary.txt"), "w") as summary_f:
        summary_f.write(str(final_test_result))

    with open(os.path.join(path_to_output, "eval_dict.txt"), "w") as eval_dict_fs:
        eval_dict_fs.write(str(eval_dict))

    args.recompute_laplacian = False


