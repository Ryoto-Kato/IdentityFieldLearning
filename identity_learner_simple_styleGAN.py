import os
import torch
import numpy as np
from torch import nn
from simple_knn._C import distCUDA2
from utils.sh_utils import eval_sh
from networks.postprocess_flameforward import get_verts_from_codes
from networks.flame_pytorch import FLAME, get_config
from utils.pkl_handler import load_from_memory

import trimesh
from scene.gaussian_model import torch_GaussianProp
from utils.vis_tools import VisTrimesh
from utils.FLAME_helper import npz2torch
from utils.dataset_handler import Filehandler
from utils.interpolation import interpolate_codes, interpolate_xyzs
from networks.stylegan2 import Generator
from tqdm import tqdm
device = "cuda"

def vecnorm(vector):
    """return L2 norm (vector length) along the last axis
        Compute the length of the vector
    """
    return torch.sqrt(torch.sum(vector**2, axis = -1))


def cos_sim_loss(vector1, vector2):
    return (1 - (torch.dot(vector1, vector2)/(vecnorm(vector1)*vecnorm(vector2))))**2

def concat_gp(gp:torch_GaussianProp, attributes = ['xyz']):
    cat_gp = gp.xyz.reshape(-1).cuda()
    return cat_gp

class flame_manager():
    def __init__(self, path2data, subject_names = []):
        self.path2data = path2data
        self.meshes = None
        self.normals = None

        # flame coeffs
        self.shape_codes = None
        self.exp_codes = None
        self.other_codes = []

        self.coeffs = None
        self.subject_names = subject_names
        self.num_subjects = len(subject_names)

        self.x_min = None
        self.y_min = None
        self.z_min = None
        self.x_max = None
        self.y_max = None
        self.z_max = None

        self.x_mean = None
        self.y_mean = None
        self.z_mean = None

    def get_npz_path(self, subject_id, exp_name):
        path2npz = os.path.join(self.path2data, subject_id, "sequences", exp_name, "annotations", "tracking",  "FLAME2023_v2", "tracked_flame_params.npz")
        return path2npz
    
    def get_coeffs(self, subject_id, exp_name, id_timestep):
        path2npz = self.get_npz_path(subject_id, exp_name)
        # get torch.cuda parameters
        shape_params, expression_params, pose_params, neck_pose, eye_pose, transl, rot_mat, scale=npz2torch(path2npz, id_timestep=int(id_timestep))
        print("shape params: ", shape_params.shape)
        return shape_params, expression_params, pose_params, neck_pose, eye_pose, transl, rot_mat, scale

    def get_flame_facemask_paths(self, subject_id, exp_name, id_timestep):
        path2facemask = os.path.join(self.path2data, subject_id, "sequences", exp_name, "timesteps", "frame_"+id_timestep, id_timestep+"_facemask.ply")
        path2facemask_subd = os.path.join(self.path2data, subject_id, "sequences", exp_name, "timesteps", "frame_"+id_timestep, id_timestep+"_facemask_subd.ply")
        path2facemask_subd2 = os.path.join(self.path2data, subject_id, "sequences", exp_name, "timesteps", "frame_"+id_timestep, id_timestep+"_facemask_subd2.ply")
        path2facemask_subd3 = os.path.join(self.path2data, subject_id, "sequences", exp_name, "timesteps", "frame_"+id_timestep, id_timestep+"_facemask_subd3.ply")
        return path2facemask, path2facemask_subd, path2facemask_subd2, path2facemask_subd3 

    def get_facemask(self, subject_id, exp_name, id_timestep, subd_type="none"):
        path2facemask, path2facemask_subd, path2facemask_subd2, path2facemask_subd3 = self.get_flame_facemask_paths(subject_id, exp_name, id_timestep)
        target_mesh = None
        if subd_type=="subd":
            target_mesh = trimesh.load(path2facemask_subd)
        elif subd_type == "subd2":
            target_mesh = trimesh.load(path2facemask_subd2)
        elif subd_type == "subd3":
            target_mesh = trimesh.load(path2facemask_subd3)
        elif subd_type == "none":
            target_mesh = trimesh.load(path2facemask)

        print(target_mesh.vertices.shape)

        return target_mesh

    def loader(self, target_exp="SEN-10-port_strong_smokey", target_timestep = "00000", subd_type = "none", datatype="codes"):
        sub_dirNames, sub_dirPaths = Filehandler.dirwalker_InFolder(path_to_folder=self.path2data, prefix='')
        self.num_subjects =0
        for subject_id, subject_path in zip(sub_dirNames, sub_dirPaths):
            if subject_id not in self.subject_names:
                continue

            print(f"current subject: {subject_id}")
            path2sequences = os.path.join(subject_path, "sequences")

            list_expNames, list_expPaths = Filehandler.dirwalker_InFolder(path_to_folder=path2sequences, prefix='')

            for exp_name, exp_path in zip(list_expNames, list_expPaths):
                if exp_name != target_exp:
                    continue
                print(f"\t{'-'}current_exp: {exp_name}")

                path2timesteps = os.path.join(exp_path, "timesteps")

                list_ts_Names, list_ts_Paths = Filehandler.dirwalker_InFolder(path_to_folder=path2timesteps, prefix='frame_')

                for id_timestep, path_timestep in zip(list_ts_Names, list_ts_Paths):
                    print(f"\t\t{'-'}current_ts: {id_timestep}")
                    if id_timestep[-5:] != target_timestep:
                        continue

                    if datatype == "codes":
                        shape_params, expression_params, pose_params, neck_pose, eye_pose, transl, rot_mat, scale = self.get_coeffs(subject_id=subject_id, exp_name=exp_name, id_timestep=id_timestep[-5:])

                        print("shape code: ", shape_params.shape)
                        print("exp code: ", expression_params.shape)

                        if self.shape_codes == None:
                            self.shape_codes = shape_params.unsqueeze(0)
                        else:
                            self.shape_codes = torch.cat((self.shape_codes, shape_params.unsqueeze(0)), 0)

                        if self.exp_codes == None:
                            self.exp_codes = expression_params.unsqueeze(0)
                        else:
                            self.exp_codes = torch.cat((self.exp_codes, expression_params.unsqueeze(0)), 0)

                        if self.other_codes == None:
                            self.other_codes = [pose_params, neck_pose, eye_pose, transl, rot_mat, scale]
                        else:
                            self.other_codes.append([pose_params, neck_pose, eye_pose, transl, rot_mat, scale])
                    elif datatype == "meshes":
                        # loading flame computed mesh
                        mesh  = self.get_facemask(subject_id=subject_id, exp_name=exp_name, id_timestep=id_timestep[-5:], subd_type=subd_type)
                        
                        print(mesh.vertices.shape)
                        print(mesh.vertex_normals.shape)

                        if self.meshes == None:
                            self.meshes = torch.from_numpy(mesh.vertices).unsqueeze(0).float()
                        else:
                            self.meshes = torch.cat((self.meshes, torch.from_numpy(mesh.vertices).unsqueeze(0).float()), 0)

                        if self.normals == None:
                            self.normals = torch.from_numpy(mesh.vertex_normals).unsqueeze(0).float()
                        else:
                            self.normals = torch.cat((self.normals, torch.from_numpy(mesh.vertex_normals).unsqueeze(0).float()), 0)


        # self.coeffs = (self.coeffs-torch.mean(self.coeffs,dim=0)[None, :])/torch.std(self.coeffs,dim=0)[None, :]
        # print(f"coeffs mean: {torch.mean(self.coeffs,0)}")
        # print(f"coeffs std: {torch.std(self.coeffs,0)}")
        # print(f"coeffs: {self.coeffs.shape}")

        if self.meshes != None:
            self.x_mean = self.meshes[:, :, 0].mean()
            self.y_mean = self.meshes[:, :, 1].mean()
            self.z_mean = self.meshes[:, :, 2].mean()
            self.x_min = self.meshes[:, :, 0].min()
            self.x_max = self.meshes[:, :, 0].max()
            self.y_min = self.meshes[:, :, 1].min()
            self.y_max = self.meshes[:, :, 1].max()
            self.z_min = self.meshes[:, :, 2].min()
            self.z_max = self.meshes[:, :, 2].max()
            print(f"meshes: {self.meshes.shape}")
        print(f"num_subjects: {self.num_subjects}")
    
    def batcher_params(self, selected_ids = [0, 2, 3, 4, 5]):
        corresp_params = self.coeffs[selected_ids]
        return corresp_params
    
    def get_tracking_mesh(self, id, device = "cpu"):
        return self.meshes[id].to(device)

class IdentityTracker(nn.Module):
    def __init__(self, path2data, path2blendshape, subject_names=[], set_id = 0, device = "cuda", method="ADAM", refinement=False, reload_subjects = False):
        super().__init__()
        self.path2data = path2data
        self.path2blendshape = path2blendshape
        self.method = method
        self.set_id = set_id

        self.subject_id2name = {}
        self.subject_name2id = {}
        self.num_subjects = len(subject_names)

        for i, subject_name in enumerate(subject_names):
            self.subject_id2name.update({i: subject_name})
            self.subject_name2id.update({subject_name: i})

        self.IL = torch.load(path2blendshape).cpu()
        self.IL.eval()

        self.pos_encode_mode = self.IL.pos_encode_mode
        self.xyz_method = self.IL.xyz_method
        self.offset_decoder_output_type = self.IL.offset_decoder_output_type
        self.ng_subd_type = self.IL.ng_subd_type

        self.sh_degree = self.IL.sh_degree
        self.num_blendshape = self.IL.num_blendshape
        self.optimizers = None
        self.device = device
        self.K = self.IL.K
        
        # xyz
        self.latent_embed_dim = self.IL.latent_embed_dim
        self.num_anchors = self.IL.num_anchors
        self.render_numGauss = self.num_anchors*self.K # 12028, K = 4 # sample from subd2

        # learnable xyz
        # initialize with flame tracking mesh
        path2mesh_normal_npy = os.path.join(path2data, f"tracking_mesh_test_{self.num_subjects}_set{self.set_id}_{self.ng_subd_type}__wNormal.npy")
        path2bbox_npy = os.path.join(path2data, f"tracking_mesh_train_{self.num_subjects}_{self.ng_subd_type}_bbox.npy")
        init_xyz = torch.empty(0)
        self.manager = None
        self.bbox = None
        self.xyz_mean = None

        self.manager = flame_manager(path2data=path2data, subject_names = subject_names)
        self.manager.loader(target_exp="EXP-1-head", target_timestep="00000", subd_type=self.ng_subd_type, datatype="meshes")
        init_xyz = self.manager.meshes.cpu() #subjects, #vert_in subd2, 3
        init_normal = self.manager.normals.cpu()
        np.save(path2mesh_normal_npy, [init_xyz.cpu().numpy(), init_normal.cpu().numpy()], allow_pickle=True)
        bbox = np.array([[self.manager.x_min, self.manager.x_max, self.manager.y_min, self.manager.y_max, self.manager.z_min, self.manager.z_max],[self.manager.x_mean, self.manager.y_mean, self.manager.z_mean]])
        np.save(path2bbox_npy, bbox, allow_pickle=True)
        print("bbox: ", type(bbox))
        self.bbox = torch.Tensor(bbox[0]).float().cuda()
        self.xyz_mean = torch.Tensor(bbox[1]).float().cuda()

        if self.xyz_method == "default":
            """
                - subjejct specific xyz [#render_numGauss, 3]
                - subject specific coeffs [80]
            """
            self.xyz = nn.Parameter(init_xyz[:, :self.render_numGauss, :].float().cpu().requires_grad_(True)) #subjects, #render_numGauss, 3

        self.coeffs = nn.Parameter(0.01*torch.randn(self.num_subjects, self.num_blendshape).float().cuda()).requires_grad_(True)
        self.blenshape = self.IL.blenshape
        
    def training_setup(self, training_args={}):
        
        if self.xyz_method == "default":
            l = [[
                    {'params': [self.xyz], 'lr': 0.00000016, "name": "xyz"},
                    {'params': [self.coeffs], 'lr': 1e-3, "name": "coeffs"}
            ]for i in range(self.num_subjects)]
    
        self.optimizers = []
        for i in range(self.num_subjects):
            if self.method == "ADAM":
                self.optimizers.append(torch.optim.Adam(l[i], lr=0.0, eps = 1e-15))
            elif self.method == "LBFGS":
                self.optimizers.append(torch.optim.LBFGS(l[i], lr=0.0, max_iter = 20, line_search_fn = 'strong_wolfe'))

    def xyz_forward(self, current_batch_ids):
        torch.cuda.empty_cache()
        print("device: ", self.xyz.device)
        return (self.xyz[current_batch_ids].cuda(), None, None, None)
    
    def forward(self, batch_names, noise_mode='const', mu=False):
        """
        B: batch size
        Input:
            batch_ids: subject name e.g., "018"
        Output:
            latent_gauss_attribs: xyz, fdc, frest, scale, rot, opac
        """

        if not mu and batch_names != None:
            current_batch_ids = []
            for name in batch_names:
                current_batch_ids.append(self.subject_name2id[name])

            coeffs = self.coeffs[current_batch_ids]
            xyz, normals, shape_code, exp_code = self.xyz_forward(current_batch_ids=current_batch_ids)#self.xyz[current_batch_ids]
            
            coeffs = coeffs.cuda()
            # xyz = xyz.cuda()
        else:
            coeffs = torch.zeros(1, self.num_blendshape).float().cuda()
            xyz = torch.mean(self.xyz, dim = 0).float().cuda() #render_Gauss, 3

        latent_gauss_attribs = self.blenshape(_coeffs=coeffs, _xyz = xyz, _normal = normals, _shape_code = shape_code, _exp_code = exp_code, noise_mode=noise_mode)
        
        return latent_gauss_attribs
    

class IdentityInterpolater(nn.Module):
    def __init__(self, path2blendshape, subject_names=[], n_frames = 10, loop = False):
        super().__init__()
        self.path2blendshape = path2blendshape
        self.loop = loop
        self.n_frames = n_frames

        self.subject_id2name = {}
        self.subject_name2id = {}
        self.num_subjects = len(subject_names)

        self.IL = torch.load(path2blendshape).cuda()
        self.IL.eval()

        self.sh_degree = self.IL.sh_degree

        self.trained_subject_id2name = self.IL.subject_id2name
        self.trained_subject_name2id = self.IL.subject_name2id
        self.trained_subject_names = self.trained_subject_name2id.keys()

        print(self.trained_subject_name2id)

        self.latent_codes = []
        self.xyz = []

        for i, subject_name in enumerate(subject_names):
            id = self.trained_subject_name2id[subject_name]
            self.latent_codes.append(self.IL.coeffs[id].float().cpu())
            self.xyz.append(self.IL.xyz[id].float().cpu())

        # interpolation
        self.interp_codes = interpolate_codes(latent_codes = self.latent_codes, n_frames = self.n_frames, loop=self.loop)
        self.interp_xyzs = interpolate_xyzs(xyzs = self.xyz, n_frames = self.n_frames, loop=self.loop)

        self.blenshape = self.IL.blenshape

        self.max_frames = len(self.interp_codes)


    def forward(self, frame_id, noise_mode = 'const', mu=False):
        """
        B: batch size
        Input:
            batch_ids: subject name e.g., "018"
        Output:
            latent_gauss_attribs: xyz, fdc, frest, scale, rot, opac
        """

        coeffs = self.interp_codes[frame_id].float().unsqueeze(0).cuda()
        print(coeffs.shape)
        xyz = self.interp_xyzs[frame_id].float().cuda()
        print(xyz.shape)

        latent_gauss_attribs = self.blenshape(_coeffs=coeffs, _xyz = xyz, _normal = None, _shape_code = None, _exp_code = None, noise_mode = noise_mode)

        return latent_gauss_attribs

class IdentityLearner(nn.Module):
    def __init__(self, path2data, blendshape_size = 80, K = 4, embed_dim = 16, m = 10, tex_reso = 64, anchorsubd_type = "subd0", subject_names=[], sh_degree = 0, num_planes = 1, device = "cuda", triplane=False, laplace = True, xyz_method = None, offset_decoder_output_type = "XYZ", reload_subjects=False):
        super().__init__()
        self.path2data = path2data
        if triplane:
            self.pos_encode_mode = "triplane"
        else:
            self.pos_encode_mode = "embedding"

        self.subject_id2name = {}
        self.subject_name2id = {}
        self.num_subjects = len(subject_names)

        for i, subject_name in enumerate(subject_names):
            self.subject_id2name.update({i: subject_name})
            self.subject_name2id.update({subject_name: i})

        self.sh_degree = sh_degree
        self.num_blendshape = blendshape_size
        self.optimizer = None
        self.device = device
        self.K = K

        # xyz
        self.latent_embed_dim = embed_dim
        self.anchorsubd_type = anchorsubd_type

        if self.anchorsubd_type == "subd0":
            self.num_anchors = 3007
            if self.K == 1:
                self.num_anchors = int(3007 * 12)
                self.anchorsubd_type = "subd3"
        elif self.anchorsubd_type == "subd1":
            self.num_anchors = 11839

        print("Num of anchors: ", self.num_anchors)

        self.ng_subd_type = "subd3"
        if self.K == 1:
            self.render_numGauss = self.num_anchors
        else:
            self.render_numGauss = self.num_anchors*self.K # 12028 K = 4 # sample from subd2
        print("Num of rendered Gaussians: ", self.render_numGauss)

        # FLAME mean facemask (reference to obtain mean attributes of 3D Gaussians)

        # 3007 (subd0)
        # path2refmesh3007 = "../data/flame_meshes/flame_facemask.ply"
        
        # 11839 (subd1)
        # path2refmesh_subd1 = "../data/flame_meshes/flame_facemask_subd.ply"
        
        # 46987 (subd2)
        path2refmesh = "../data/flame_meshes/flame_facemask_subd2.ply"
        
        # 187219 (subd3)
        # path2refmesh = "../data/flame_meshes/flame_facemask_subd3.ply"
        
        # 747427 (subd4)
        # path2refmesh = "../data/flame_meshes/flame_facemask_subd4.ply"


        refmesh = trimesh.load(path2refmesh, process= False)
        self.ref_mean = torch.from_numpy(refmesh.vertices).float()
        self.ref_mean = self.ref_mean[:self.render_numGauss, :]

        # scale init
        dist2 = torch.clamp_min(distCUDA2(self.ref_mean.cuda()), 0.0000001)
        sample_scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        self.mu_scale = nn.Parameter(sample_scales.float().clone()).requires_grad_(True)
        
        # rot init
        mu_rot = torch.zeros((self.render_numGauss, 4)).cuda()
        mu_rot[:, 0] = 1
        self.mu_rot = nn.Parameter(mu_rot.float()).requires_grad_(True)

        # opac init
        self.mu_opac = nn.Parameter(0.1*torch.ones((self.render_numGauss, 1)).float().cuda()).requires_grad_(True)

        self.xyz_method = xyz_method
        self.offset_decoder_output_type = offset_decoder_output_type
        self.offset_SH = 0

        if self.offset_decoder_output_type == "SH":
            self.offset_SH = 3

        path2codes_npy = os.path.join(path2data, f"tracking_mesh_train_{self.num_subjects}_codes.npy")
        path2mesh_normal_npy = os.path.join(path2data, f"tracking_mesh_train_{self.num_subjects}_{self.ng_subd_type}_wNormal.npy")
        
        path2bbox_npy = os.path.join(path2data, f"tracking_mesh_train_{self.num_subjects}_{self.ng_subd_type}_bbox.npy")

        init_xyz = torch.empty(0)
        self.manager = None
        self.bbox = None
        self.xyz_mean = None
        
        if not os.path.exists(path2mesh_normal_npy) or not os.path.exists(path2bbox_npy) or reload_subjects:
            self.manager = flame_manager(path2data=path2data, subject_names = subject_names)
            self.manager.loader(target_exp="EXP-1-head", target_timestep="00000", subd_type=self.ng_subd_type, datatype="meshes")
            init_xyz = self.manager.meshes.cpu() #subjects, #vert_in subd2, 3
            init_normal = self.manager.normals.cpu()
            np.save(path2mesh_normal_npy, [init_xyz.cpu().numpy(), init_normal.cpu().numpy()], allow_pickle=True)
            bbox = np.array([[self.manager.x_min, self.manager.x_max, self.manager.y_min, self.manager.y_max, self.manager.z_min, self.manager.z_max],[self.manager.x_mean, self.manager.y_mean, self.manager.z_mean]])
            np.save(path2bbox_npy, bbox, allow_pickle=True)
            print("bbox: ", type(bbox))
            self.bbox = torch.Tensor(bbox[0]).float().cuda()
            self.xyz_mean = torch.Tensor(bbox[1]).float().cuda()
        else:
            [np_init_xyz, np_init_normal] = np.load(path2mesh_normal_npy, allow_pickle=True)
            init_xyz = torch.from_numpy(np_init_xyz).float().cpu()
            init_normal = torch.from_numpy(np_init_normal).float().cpu()

            np_bbox = np.load(path2bbox_npy, allow_pickle=True)
            self.bbox = torch.Tensor(np_bbox[0]).float().cuda()
            self.xyz_mean = torch.Tensor(np_bbox[1]).float().cuda()

        if triplane:
            xmin = self.bbox[0]
            xmax = self.bbox[1]
            ymin = self.bbox[2]
            ymax = self.bbox[3]
            zmin = self.bbox[4]
            zmax = self.bbox[5]

            radius = torch.sqrt((xmax-xmin)**2+(ymax-ymin)**2+(zmax-zmin)**2).float().cuda()
            hyper_scale = 1.0
        
        self.anchors = None
        self.offsets = None

        self.shape_codes = None
        self.exp_codes = None
        self.other_codes = None

        self.flamelayer = None

        # scaled the bbox into [-1, 1]
        if triplane:
            aligned_bbox = torch.Tensor([xmin-xmin, xmax-xmin, ymin-ymin, ymax-ymin, zmin-zmin, zmax-zmin]).float().cuda()
            scaled_bbox = (1/(radius*hyper_scale))*aligned_bbox

            self.center_of_bbox = 0.5*torch.Tensor([scaled_bbox[1]-scaled_bbox[0], scaled_bbox[3]-scaled_bbox[2], scaled_bbox[5]-scaled_bbox[4]]).float().cuda()
            self.bbox_min = torch.Tensor([xmin, ymin, zmin]).float().cuda()

        if self.xyz_method == "default":
            self.xyz = nn.Parameter(init_xyz[:, :self.render_numGauss, :].float().cpu().requires_grad_(True)) #subjects, #render_numGauss, 3

        # learnable coeffs
        self.coeffs = nn.Parameter(0.01*torch.randn(self.num_subjects, self.num_blendshape).float().cuda()).requires_grad_(True)

        self.pos_embedding = None
        self.triplanes = None

        # styleGAN configuration
        self.num_planes = num_planes
        self.z_dim = self.latent_embed_dim     # input latnet z dimensionality (after linear blending)
        self.w_dim = self.z_dim  # output from mapping network f which is not used in this case
        self.c_dim = 0           # Conditioning label

        self.latent_tex_reso = tex_reso #Output resolution [4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]
        self.latent_tex_channels = self.z_dim


        if triplane:
            self.axes = self.generate_planes()
            self.spatial_reso = 256
            self.n_planes = 3
            #[0: xy, 1: yz, 2: xz]
            self.triplanes = nn.Parameter(torch.randn(self.n_planes, self.spatial_reso*self.spatial_reso*self.latent_embed_dim).float().cuda().requires_grad_(True)) #[3, [256, 256, 48]]
        else:
            self.pos_embedding = nn.Parameter(torch.randn((self.num_anchors, self.latent_embed_dim)).float().cuda().requires_grad_(True))

        self.G = Generator(z_dim=self.z_dim, c_dim=self.c_dim, w_dim=self.w_dim, img_resolution=self.latent_tex_reso, img_channels=self.latent_tex_channels).cuda()

        path2uv_coords = f"../data/flame_meshes/FLAME_texture_{self.anchorsubd_type}_facemask.npy"
        
        uv_coords = np.load(path2uv_coords, allow_pickle=True)
        if self.K==1:
            uv_coords = uv_coords[:self.num_anchors, :]
        uv_coords = torch.from_numpy(uv_coords).float().unsqueeze(0)
 
        self.uv_coords = uv_coords.repeat([self.num_planes, 1, 1, 1]).cuda()

        self.embeddings=nn.Parameter(torch.randn((self.latent_embed_dim, self.num_planes, self.num_blendshape)).float().cuda().requires_grad_(True))

        self.fdc_ch = 3
        self.scale_ch = 3
        self.rot_ch = 4
        self.opac_ch = 1

        # chs:11 (3+3+4+1)
        self.other_ch = self.scale_ch+self.rot_ch+self.opac_ch

        self.color_decoder = nn.Sequential(
            nn.Linear(self.latent_embed_dim, self.latent_embed_dim),
            nn.ReLU(True),
            nn.Linear(self.latent_embed_dim, self.fdc_ch*self.K),
        ).cuda()

        self.scale_decoder = nn.Sequential(
            nn.Linear(self.latent_embed_dim, self.latent_embed_dim),
            nn.ReLU(True),
            nn.Linear(self.latent_embed_dim, self.scale_ch*self.K),
        ).cuda()

        self.rot_decoder = nn.Sequential(
            nn.Linear(self.latent_embed_dim, self.latent_embed_dim),
            nn.ReLU(True),
            nn.Linear(self.latent_embed_dim, self.rot_ch*self.K),
        ).cuda()

        self.opac_decoder = nn.Sequential(
            nn.Linear(self.latent_embed_dim, self.latent_embed_dim),
            nn.ReLU(True),
            nn.Linear(self.latent_embed_dim, self.opac_ch*self.K),
        ).cuda()


        self.pp_fdc = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.pp_opac = nn.Sigmoid()
        self.latent_triplanes = None

    def sample_from_planes(self, triplane, mode='bilinear', padding_mode='zeros'):
        """
            triplane: [3. 16. 64, 64]
                16: self.latent_tex_channels
                64: self.latent_tex_reso
            
            uv_coordinate: [3, 1, 3007, 2]
        """

        uv_coordinate = self.uv_coords
        print("uv coordinate: ", uv_coordinate.shape)
        output_features = torch.nn.functional.grid_sample(triplane, uv_coordinate, mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(self.num_planes, self.num_anchors, self.latent_tex_channels)
        return output_features

    def apply_laplace(self, embedding):
        _ms = self.sigmoid(self.ms) + self.base_m
        print("ms: ", _ms)
        self.operator = self.mel.get_operator(ms = _ms) #3007
        print("operator: ", self.operator.shape)
        _embed = embedding.reshape(self.num_anchors, -1)
        sm_embed = torch.matmul(self.operator, _embed)
        sm_embed = sm_embed.reshape(self.num_anchors, self.latent_embed_dim*self.num_planes)
        return sm_embed

    def initialize_weight(self, module):
        if isinstance(module, (nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.zero_()

    def training_setup(self, training_args={"xyz_lr": 0.00000016}):
        
        if self.pos_encode_mode == "triplane":
            l = [
                {'params': self.color_decoder.parameters(), 'lr': 0.002, "name": "color_decoder"},
                {'params': self.scale_decoder.parameters(), 'lr': 0.001, "name": "scale_decoder"},
                {'params': self.rot_decoder.parameters(), 'lr': 0.001,"name": "rot_decoder"},
                {'params': self.opac_decoder.parameters(), 'lr': 0.001, "name": "opac_decoder"},
                {'params': self.G.parameters(), 'lr': 0.004, "name": "Generator"},
                {'params': [self.embeddings], 'lr': 1e-3, "name": "embedding"},
                {'params': [self.triplanes], 'lr': 1e-3, "name": "triplanes"},
                {'params': [self.coeffs], 'lr': 1e-3, "name": "coeffs"},
                {'params': [self.mu_scale], 'lr': 1e-4, "name": "mu_scale"},
                {'params': [self.mu_opac], 'lr': 1e-4, "name": "mu_opac"},
                {'params': [self.mu_rot], 'lr': 1e-4, "name": "mu_rot"},
            ]
        else:
            l = [
                {'params': self.color_decoder.parameters(), 'lr': 0.002, "name": "color_decoder"},
                {'params': self.scale_decoder.parameters(), 'lr': 0.001, "name": "scale_decoder"},
                {'params': self.rot_decoder.parameters(), 'lr': 0.001, "name": "rot_decoder"},
                {'params': self.opac_decoder.parameters(), 'lr': 0.001, "name": "opac_decoder"},
                {'params': self.G.parameters(), 'lr': 0.004, "name": "Generator"},
                {'params': [self.embeddings], 'lr': 1e-3, "name": "embedding"},
                {'params': [self.pos_embedding], 'lr': 1e-3, "name": "triplanes"},
                {'params': [self.coeffs], 'lr': 1e-3, "name": "coeffs"},
                {'params': [self.mu_scale], 'lr': 1e-4, "name": "mu_scale"},
                {'params': [self.mu_opac], 'lr': 1e-4, "name": "mu_opac"},
                {'params': [self.mu_rot], 'lr': 1e-4, "name": "mu_rot"},
            ]

        if self.xyz_method == "default":
            l.append({'params': [self.xyz], 'lr': training_args["xyz_lr"], "name": "xyz"})
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps = 1e-15)
            
    def blenshape(self, _coeffs, _xyz, _normal, _shape_code, _exp_code, noise_mode='const'):
        B = 1
        print("xyz_method: ", self.xyz_method)
        xyz = _xyz.reshape(self.render_numGauss, 3)

        z = torch.einsum('bl,mkl->bmk', _coeffs, self.embeddings).squeeze(0).reshape(self.num_planes, self.latent_embed_dim)

        if self.pos_encode_mode == "triplane":
            coordinate_anchors = _xyz[:3007, :]
            pos_embed = self.sample_from_planes(coordinates=coordinate_anchors) #3007, #embed_dims
        else:
            pos_embed = self.pos_embedding

        # [#plane, 64, 128, 128]
        latent_triplanes = self.G(z=z, c = None, noise_mode=noise_mode)
        self.latent_triplanes = latent_triplanes

        # [#plane, 3007, 64]
        sampled_latent_codes_by_anchors = self.sample_from_planes(triplane=latent_triplanes)        

        _embed = pos_embed[None] + sampled_latent_codes_by_anchors

        final_embed = _embed.clone()

        # [#plane, 3007, 64] -> [3007, 3*4]
        if self.num_planes>1:
            s_fdc = self.color_decoder(final_embed[0])
        else:
            s_fdc = self.color_decoder(final_embed)

        s_fdc = s_fdc.reshape(self.render_numGauss, self.fdc_ch) 
        _fdc = self.pp_fdc(s_fdc) #still RGB [0, 1]

        if self.num_planes>1:
            s_scale = self.scale_decoder(final_embed[1])
        else:
            s_scale = self.scale_decoder(final_embed)

        s_scale = s_scale.reshape(self.render_numGauss, self.scale_ch)
        _scales = s_scale + self.mu_scale

        # [#plane, 3007, 64] -> [3007, 4*4]
        if self.num_planes>1:
            s_rot = self.rot_decoder(final_embed[2])
        else:
            s_rot = self.rot_decoder(final_embed)

        s_rot = s_rot.reshape(self.render_numGauss, self.rot_ch) 
        _rots = s_rot+self.mu_rot

        # [#plane, 3007, 64] -> [3007, 1*4]
        if self.num_planes>1:
            s_opac = self.opac_decoder(final_embed[3])
        else:
            s_opac = self.opac_decoder(final_embed)

        s_opac = s_opac.reshape(self.render_numGauss, self.opac_ch)   
        _opacities = self.pp_opac(s_opac + self.mu_opac)

        _frest = None#torch.zeros(self.render_numGauss, int((((self.sh_degree+1)**2)*3-3)/3), 3).float().cuda()
        torch.cuda.empty_cache()
        return xyz, _fdc.squeeze(0), _frest, _scales.squeeze(0), _rots.squeeze(0), _opacities.squeeze(0)

    def xyz_forward(self, current_batch_ids):
        torch.cuda.empty_cache()
        print("device: ", self.xyz.device)
        return (self.xyz[current_batch_ids].cuda(), None, None, None)

    def forward(self, batch_names, noise_mode='const', mu=False):
        """
        B: batch size
        Input:
            batch_ids: subject name e.g., "018"
        Output:
            latent_gauss_attribs: xyz, fdc, frest, scale, rot, opac
        """

        if not mu and batch_names != None:
            current_batch_ids = []
            for name in batch_names:
                current_batch_ids.append(self.subject_name2id[name])

            coeffs = self.coeffs[current_batch_ids]
            xyz, normals, shape_code, exp_code = self.xyz_forward(current_batch_ids=current_batch_ids)#self.xyz[current_batch_ids]
            
            coeffs = coeffs.cuda()
        else:
            coeffs = torch.zeros(1, self.num_blendshape).float().cuda()
            xyz = torch.mean(self.xyz, dim = 0).float().cuda() #render_Gauss, 3

        latent_gauss_attribs = self.blenshape(_coeffs=coeffs, _xyz = xyz, _normal = normals, _shape_code = shape_code, _exp_code = exp_code, noise_mode=noise_mode)

        return latent_gauss_attribs
