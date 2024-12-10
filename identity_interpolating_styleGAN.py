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
from Identity_interpolater_styleGAN import identity_interpolating

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
    parser.add_argument("--render_camInterpolation", type=bool, default=False)


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

    now = datetime.now()
    date = now.strftime("%d%m")
    time = now.strftime("%H%M")

    if not args.render_camInterpolation:
        title_experiment = f"Identity_interpolating_styleGAN_{args.IL_name}_set{args.set_id}_"+time+"_"+date+"_centercam"
    else:
        title_experiment = f"Identity_interpolating_styleGAN_{args.IL_name}_set{args.set_id}_"+time+"_"+date

    path2exp_out = os.path.join("/mnt/hdd/output", title_experiment)
    scale = 0.5
    
    path2refmesh = "/home/kato/Photorealistic-3DMM/DeformationLearning_3DGS/samples/flame_meshes/flame_facemask_subd.ply"
    configs = {"n_frames": 90, "loop": False, "render_camInterpolation": args.render_camInterpolation, "set_id": args.set_id, "path2blendshape": args.path2blendshape}
    str_config=str(configs)

    path_to_output = os.path.join(path2exp_out)
    os.makedirs(path_to_output, exist_ok = True)
    # path_to_output = path2exp_out

    print(configs)
    with open(os.path.join(path2exp_out, "configs.txt"), "w") as f:
        f.write(str(configs))

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    _ALLcam = args.ALLcam
    numOfFrames = 1

    identity_interpolating(lp.extract(args), pp.extract(args), path2blendshape=args.path2blendshape, expName=EXP, unique_str="", path_to_output=path_to_output, scale=0.5, configs = configs)

    #ffmpeg -i %d.png -c:v libx264 -r 30 output.mp4


