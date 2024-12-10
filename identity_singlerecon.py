import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from render_3DGS_mesh import render_3DGS_mesh
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
from single_recon import identity_tracking

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# [TODO] set the path to "3DSSL-WS23_IntuitiveAnimation"
path_to_3WI = os.path.join(os.getcwd(), os.pardir)
sys.path.append(os.path.join(path_to_3WI, 'src'))

from utils.OBJ_helper import OBJ
from utils.Dataset_handler import Filehandler
from utils.pickel_io import dump_pckl

from datetime import datetime

if __name__ == "__main__":

    # args for render 3DGS in mesh
    render_parser = ArgumentParser(description="render 3dgs in mesh")
    render_parser.add_argument("--path_to_3DGS", type = str, default="")
    render_parser.add_argument("--all", type = bool, default=True)
    render_parser.add_argument("--render", type = bool, default=True)
    render_parser.add_argument("--iter", type = int, default=15000)
    render_parser.add_argument("--path2blendshape", type = str, default="")
    render_parser.add_argument("--set_id", type=int, default=0)
    render_parser.add_argument("--IL_name", type=str, default="")

    render_args = render_parser.parse_args(sys.argv[1:])

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    # load hyper parameters
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    max_iterations =  15_000
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
    parser.add_argument("--path2blendshape", type = str, default="")
    parser.add_argument("--set_id", type=int, default=0)
    parser.add_argument("--IL_name", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    args.iterations = max_iterations
    args.random_background = False
    args.test_iterations.append(0)
    args.test_iterations.append(args.iterations)
    args.save_iterations.append(0)
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)


    path_to_dataset = '/mnt/hdd/dataset/269-single-timestep-EXP-1-head'
    EXP = "EXP-1-head"

    K = 12
    render_numGauss = 3007 * K
    subd = -1

    now = datetime.now()
    date = now.strftime("%d%m")
    time = now.strftime("%H%M")
    title_experiment = f"Identity_{args.IL_name}_singlerecon_numGauss{render_numGauss}_set{args.set_id}_defaultlr"+time+"_"+date

    path2exp_out = os.path.join("/mnt/hdd/EVAL24", title_experiment)
    scale = 0.5

    list_ids, list_paths = Filehandler.dirwalker_InFolder(path_to_folder = path_to_dataset, prefix="")
    list_ids = list_ids[:-1]
    num_subjects = len(list_ids)
    path2refmesh = "/home/kato/Photorealistic-3DMM/DeformationLearning_3DGS/samples/flame_meshes/flame_facemask_subd.ply"
    configs = {"max_iters": 30_000, "COEFF_REG": 0.0, "set": args.set_id, "render_numGauss": render_numGauss, "subd": subd, "optimizer": "ADAM", "path2data": path_to_dataset, "path2refmesh": path2refmesh, "path2blendshape": args.path2blendshape}
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

    average_l1, average_lpips, average_ssim, average_psnr, eval_dict = identity_tracking(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, path2blendshape=args.path2blendshape, list_ids = list_ids, expName=EXP, unique_str="", path_to_output=path_to_output, scale=0.5, configs = configs)
    print(f"FINAL l1 {average_l1}, PSNR {average_psnr} LPIPS {average_lpips} SSIM {average_ssim}")

    render_args.path_to_3DGS = path_to_output
    render_3DGS_mesh(render_args)
    
    final_test_result = {"config": str_config, "num_sub": num_subjects, "average_L1": average_l1, "average_PSNR": average_psnr, "average_LPIPS": average_lpips, "average_SSIM": average_ssim}
    with open(os.path.join(path_to_output, "summary.txt"), "w") as summary_f:
        summary_f.write(str(final_test_result))

    with open(os.path.join(path_to_output, "eval_dict.txt"), "w") as eval_dict_fs:
        eval_dict_fs.write(str(eval_dict))

    args.recompute_laplacian = False


