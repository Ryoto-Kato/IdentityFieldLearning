import argparse

parser = argparse.ArgumentParser(description="FLAME model")

parser.add_argument("--K", type=int, default=12)
parser.add_argument("--num_train_subjects", type=int, default=128)
parser.add_argument("--max_iters", type=int, default=30_000)
parser.add_argument("--IL_name", type=str, default="")
parser.add_argument("--anchorsubd_type", type=str, default="subd0")
parser.add_argument("--embed_dim", type=int, default=64)
parser.add_argument("--tex_reso", type=int, default=64)
parser.add_argument("--num_blendshape", type=int, default=80)
parser.add_argument("--full_head", action="store_true")
parser.add_argument("--gt_type", type=str, default="flame")
parser.add_argument("--cs_reg_weight", type=float, default=0.0)
parser.add_argument("--delayed_noise_start_iter", type=int, default=-1)
parser.add_argument("--xyz_lr", type = str, default="COMMON")
parser.add_argument("--xyz_method", type=str, default="default")
parser.add_argument("--mask_part", type=str, default="FACE")
parser.add_argument("--cog_reg_weight", type=float, default=1e-2)
parser.add_argument("--opac_reg_weight", type = float, default=1.0)
parser.add_argument("--num_planes", type = int, default=1)

parser.add_argument(
    "--flame_model_path",
    type=str,
    default="./data/flame2023.pkl",
    help="flamqe model path",
)

parser.add_argument(
    "--static_landmark_embedding_path",
    type=str,
    default="./data/flame_static_embedding.pkl",
    help="Static landmark embeddings path for FLAME",
)

parser.add_argument(
    "--dynamic_landmark_embedding_path",
    type=str,
    default="./data/flame_dynamic_embedding.npy",
    help="Dynamic contour embedding path for FLAME",
)

# FLAME hyper-parameters

parser.add_argument(
    "--shape_params", type=int, default=300, help="the number of shape parameters"
)

parser.add_argument(
    "--expression_params",
    type=int,
    default=100,
    help="the number of expression parameters",
)

parser.add_argument(
    "--pose_params", type=int, default=6, help="the number of pose parameters"
)

# Training hyper-parameters

parser.add_argument(
    "--use_face_contour",
    default=True,
    type=bool,
    help="If true apply the landmark loss on also on the face contour.",
)

parser.add_argument(
    "--use_3D_translation",
    default=False,  # Flase for RingNet project
    type=bool,
    help="If true apply the landmark loss on also on the face contour.",
)

parser.add_argument(
    "--optimize_eyeballpose",
    default=True,  # False for For RingNet project
    type=bool,
    help="If true optimize for the eyeball pose.",
)

parser.add_argument(
    "--optimize_neckpose",
    default=True,  # False For RingNet project
    type=bool,
    help="If true optimize for the neck pose.",
)

parser.add_argument("--num_worker", type=int, default=4, help="pytorch number worker.")

parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")

parser.add_argument("--ring_margin", type=float, default=0.5, help="ring margin.")

parser.add_argument(
    "--ring_loss_weight", type=float, default=1.0, help="weight on ring loss."
)

parser.add_argument("--path2output", type=str, default='./outputs/')

parser.add_argument("--separate_transform", type=bool, default =True)

def get_config():
    config = parser.parse_args()
    return config
