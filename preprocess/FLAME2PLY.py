import os
import sys
import numpy as np
import torch
import trimesh

from networks.flame_pytorch import FLAME, get_config
from utils.ply_helper import save_ply
from utils.FLAME_helper import npz2torch_cuda
from utils.dataset_handler import Filehandler
import matplotlib.pyplot as plt

from argparse import ArgumentParser

# def nha2nersemble(additional_dict):
    


def get_PLY_from_npz(path2npz='./data/sample_flame_params.npz', path2output = "", id_timestep = "00000", save_name = "sample.ply", flame_mean=False, data_format="nersemble"):
    config = get_config(data_format=data_format)
    print("config: ", config.flame_model_path)

    radian = np.pi / 180.0
    flamelayer = FLAME(config, texture_mode="FLAME")
    additional_dict = None
    # get torch.cuda parameters
    if flame_mean:
        shape_params, expression_params, pose_params, neck_pose, eye_pose, transl, rot_mat, scale, additional_dict=npz2torch_cuda(path2npz=path2npz, id_timestep=int(id_timestep), data_format=data_format)
        
        shape_params = torch.zeros_like(shape_params).cuda()
        expression_params = torch.zeros_like(expression_params).cuda()
        pose_params = torch.zeros_like(pose_params).cuda()
        neck_pose = torch.zeros_like(neck_pose).cuda()
        eye_pose = torch.zeros_like(eye_pose).cuda()
        transl = torch.zeros_like(transl).cuda()
        rot_mat = torch.zeros_like(rot_mat).cuda()
        scale = torch.ones_like(scale).cuda()
    else:
        shape_params, expression_params, pose_params, neck_pose, eye_pose, transl, rot_mat, scale, additional_dict=npz2torch_cuda(path2npz, id_timestep=int(id_timestep), data_format=data_format)
        expression_params = torch.zeros_like(expression_params).cuda()
    
    flamelayer.cuda()

    # print(f"rot mat: {rot_mat}")
    scale_mat = torch.eye(3).cuda()
    scale_mat *=scale
    print(f"scale {scale}")
    # print(f"scale mat: {scale_mat}")

    # Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework
    flame_vertice, flame_landmark = flamelayer(shape_params = shape_params, expression_params = expression_params, pose_params = pose_params, neck_pose = neck_pose, eye_pose= eye_pose, transl = None if config.separate_transform else transl)
    # print(flame_vertice.size(), flame_landmark.size())

    # Visualize Landmarks
    # This visualises the static landmarks and the pose dependent dynamic landmarks used for RingNet project
    faces = flamelayer.faces
    vertices = flame_vertice.detach().cpu().numpy().squeeze()
    cano_vertices = vertices.copy()
    rot_mat = rot_mat.detach().cpu().numpy().squeeze()
    scale_mat = scale_mat.detach().cpu().numpy().squeeze()
    translation = transl.detach().cpu().numpy().squeeze()
    # translation = translation.reshape(3, 1)
    cano2world=np.zeros((3, 4))
    scaled_rot_mat = np.matmul(scale_mat, rot_mat) # world to canonical, .T: cononical to world
    print(f"scaled rotation:\n {scaled_rot_mat}")
    cano2world[:3, :3] = scaled_rot_mat
    cano2world[:3, 3] = translation
    print(f"traslation:\n {translation}")
    print(cano2world)

    vertices = np.matmul(vertices, scale_mat.T)

    world2cano_txt_path = os.path.join(path2output, "flame_cano2world.txt") 
    with open(world2cano_txt_path, "w") as f:
        for row in cano2world:
            f.write(" ".join([str(x) for x in row])+"\n")

    print("world2canonical space")
    print(cano2world)

    joints = flame_landmark.detach().cpu().numpy().squeeze()
    vertex_colors = np.zeros([vertices.shape[0], 4])

    mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)

    vertices = np.asarray(mesh.vertices)
    print("mesh vertices: ", vertices.shape)
    vertex_normals = np.asarray(mesh.vertex_normals)
    faces = np.asarray(mesh.faces)

    if data_format == "nersemble":
        new_mouth_faces = np.array([
                    [1666, 3514, 2783],
                    [1665, 1666, 2783],
                    [1665, 2783, 2782],
                    [1665, 2782, 1739],
                    [1739, 2782, 2854],
                    [1739, 2854, 1742],
                    [1742, 2854, 2857],
                    [1742, 2857, 1747],
                    [1747, 2857, 2862],
                    [1747, 2862, 1746],
                    [1746, 2862, 2861],
                    [1746, 2861, 1595],
                    [1595, 2861, 2731],
                    [1595, 2731, 1594],
                    [1594, 2731, 2730],
                    [1594, 2730, 1572],
                    [1572, 2730, 2708],
                    [1572, 2708, 1573],
                    [1573, 2708, 2709],
                    [1573, 2709, 1860],
                    [1860, 2709, 2943],
                    [1860, 2943, 1862],
                    [1862, 2943, 2945],
                    [1862, 2945, 1830],
                    [1830, 2945, 2930],
                    [1830, 2930, 1835],
                    [1835, 2930, 2933],
                    [1835, 2933, 1852],
                    [1852, 2933, 2941],
                    [1852, 2941, 3497]
                ])

        print(f"before closing mouth: {faces.shape}")
        faces = np.concatenate([faces, new_mouth_faces])
        print(f"after closing mouth: {faces.shape}")

    uv_coords = flamelayer.texture_coordinates_by_vertex

    vt=flamelayer.vt
    ft=flamelayer.ft
    print("vt: ", vt.shape)
    print("ft: ", ft.shape)
    print("uv_coords:", uv_coords.shape)

    path2subd0_uv_coords_save = os.path.join(os.path.dirname(flamelayer.path2tex_space_path), "FLAME_texture_subd0")
    path2subd0_vt_save = os.path.join(os.path.dirname(flamelayer.path2tex_space_path), "FLAME_texture_vt")
    path2subd0_ft_save = os.path.join(os.path.dirname(flamelayer.path2tex_space_path), "FLAME_texture_ft")
    if not os.path.exists(path2subd0_uv_coords_save):
        np.save(path2subd0_uv_coords_save, arr = uv_coords, allow_pickle=True)
    if not os.path.exists(path2subd0_vt_save):
        np.save(path2subd0_vt_save, arr = vt, allow_pickle=True)
    if not os.path.exists(path2subd0_ft_save):
        np.save(path2subd0_ft_save, arr = ft, allow_pickle=True)

    vertex_colors = np.asarray(vertex_colors)
    # path2npz: /exp_folder/annotations/tracking/FLAME2023_v2/.npz
    # path2output: /exp_folder/timesteps/frame_00000/00000.ply
    # print(save_name)
    save_ply(os.path.join(path2output, save_name), vertices = vertices, faces = faces, vertex_normals = vertex_normals, vertex_colors = vertex_colors, only_points=False)

    plt.scatter(uv_coords[:, 0], uv_coords[:, 1], marker='o')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(os.path.dirname(flamelayer.path2tex_space_path), f"FLAME_texture_subd0.png"))
    plt.close()

    # in HIFI3D uv coordinate, [2.0, 1.0]-> eye regions

    if flamelayer.texture_mode == "HIFI3D":
        eyes_vertID = np.argwhere(uv_coords[:, 0] > 1.0)
        print("eyes_vertID: ", eyes_vertID.shape)

        np.save(file = os.path.join(os.path.dirname(flamelayer.path2tex_space_path), "FLAME_eyes_vertID"), arr = eyes_vertID, allow_pickle=True)

        eyes_uv_coords = np.array([[x[0]-2.0, x[1]] for x in uv_coords if x[0] > 1.0])
        print("eyes_uv: ", eyes_uv_coords.shape)

        plt.scatter(eyes_uv_coords[:, 0], eyes_uv_coords[:, 1], marker='o')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(os.path.join(os.path.dirname(flamelayer.path2tex_space_path), f"FLAME_texture_sub0_eyes.png"))
        plt.close()

        np.save(file = os.path.join(os.path.dirname(flamelayer.path2tex_space_path), "FLAME_texture_subd0_eyes"), arr = eyes_uv_coords, allow_pickle=True)

    faces_uv_coords = np.array([[x[0], x[1]] for x in uv_coords if x[0] <= 1.0])

    plt.scatter(faces_uv_coords[:, 0], faces_uv_coords[:, 1], marker='o')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(os.path.dirname(flamelayer.path2tex_space_path), f"FLAME_texture_sub0_faces.png"))
    plt.close()

    np.save(file = os.path.join(os.path.dirname(flamelayer.path2tex_space_path), "FLAME_texture_subd0_faces"), arr = faces_uv_coords, allow_pickle=True)

    # save two subdivision versions

    # subd1
    subd_verts, subd_faces, subd_dict = trimesh.remesh.subdivide(vertices=vertices, faces = faces, face_index=None, vertex_attributes={"normal": vertex_normals, "color":vertex_colors, "uv_coords":uv_coords}, return_index=True)
    subd_vertex_normals = subd_dict["normal"]
    subd_vertex_colors = subd_dict["color"]
    subd_uv_coords = subd_dict["uv_coords"]
    print("subd1_uv_coords: ", subd_uv_coords.shape)
    print(save_name[:-4] +'_subd.ply')
    print("subd verts: ", subd_verts.shape)
    save_ply(os.path.join(path2output, save_name[:-4] +'_subd.ply'), vertices=subd_verts, faces=subd_faces, vertex_normals=subd_vertex_normals, vertex_colors=subd_vertex_colors, only_points=False)

    plt.scatter(subd_uv_coords[:, 0], subd_uv_coords[:, 1], marker='o')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(os.path.dirname(flamelayer.path2tex_space_path), f"FLAME_texture_subd1.png"))
    plt.close()

    #save uv_coords
    path2subd1_uvcoords_save = os.path.join(os.path.dirname(flamelayer.path2tex_space_path), "FLAME_texture_subd1")
    if not os.path.exists(path2subd1_uvcoords_save):
        np.save(file = path2subd1_uvcoords_save, arr = subd_uv_coords, allow_pickle=True) 

    # subd2
    subd_verts, subd_faces, subd_dict = trimesh.remesh.subdivide(vertices=subd_verts, faces = subd_faces, face_index=None, vertex_attributes={"normal": subd_vertex_normals, "color":subd_vertex_colors, "uv_coords":subd_uv_coords}, return_index=True)
    subd_vertex_normals = subd_dict["normal"]
    subd_vertex_colors = subd_dict["color"]
    subd_uv_coords = subd_dict["uv_coords"]
    print("subd2_uv_coords: ", subd_uv_coords.shape)
    print(save_name[:-4]  +'_subd2.ply')
    print("subd verts: ",subd_verts.shape)
    save_ply(os.path.join(path2output, save_name[:-4]  +'_subd2.ply'), vertices=subd_verts, faces=subd_faces, vertex_normals=subd_vertex_normals, vertex_colors=subd_vertex_colors, only_points=False)

    plt.scatter(subd_uv_coords[:, 0], subd_uv_coords[:, 1], marker='o')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(os.path.dirname(flamelayer.path2tex_space_path), f"FLAME_texture_subd2.png"))
    plt.close()

    #save uv_coords
    path2subd2_uvcoords_save = os.path.join(os.path.dirname(flamelayer.path2tex_space_path), "FLAME_texture_subd2")
    if not os.path.exists(path2subd2_uvcoords_save):
        np.save(file = path2subd2_uvcoords_save, arr = subd_uv_coords, allow_pickle=True)

    # subd3
    subd_verts, subd_faces, subd_dict = trimesh.remesh.subdivide(vertices=subd_verts, faces = subd_faces, face_index=None, vertex_attributes={"normal": subd_vertex_normals, "color":subd_vertex_colors, "uv_coords":subd_uv_coords}, return_index=True)
    subd_vertex_normals = subd_dict["normal"]
    subd_vertex_colors = subd_dict["color"]
    subd_uv_coords = subd_dict["uv_coords"]
    print("subd3_uv_coords: ", subd_uv_coords.shape)
    print(save_name[:-4]  +'_subd3.ply')
    print("subd verts: ",subd_verts.shape)
    save_ply(os.path.join(path2output, save_name[:-4]  +'_subd3.ply'), vertices=subd_verts, faces=subd_faces, vertex_normals=subd_vertex_normals, vertex_colors=subd_vertex_colors, only_points=False)

    plt.scatter(subd_uv_coords[:, 0], subd_uv_coords[:, 1], marker='o')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(os.path.dirname(flamelayer.path2tex_space_path), f"FLAME_texture_subd3.png"))
    plt.close()

    #save uv_coords
    path2subd2_uvcoords_save = os.path.join(os.path.dirname(flamelayer.path2tex_space_path), "FLAME_texture_subd3")
    if not os.path.exists(path2subd2_uvcoords_save):
        np.save(file = path2subd2_uvcoords_save, arr = subd_uv_coords, allow_pickle=True)

    # if data_format == "nha":
        # nha2nersemble()


if __name__ == "__main__":

    parser = ArgumentParser(description="FLAME2PLY")
    parser.add_argument('--path2data', type = str, help="path to nersemble data parent_dir contains subjects")
    parser.add_argument('--subject_name', type = str, default=None, help="subject name if you want to process specific person")
    parser.add_argument('--flameMEAN', action="store_true", default=False, help="if true: outputs FLAME mean mesh")
    args = parser.parse_args(sys.argv[1:])

    flameMEAN=args.flameMEAN #False
    subject_name=args.subject_name #None
    data_format="nersemble"

    sub_dirNames, sub_dirPaths = Filehandler.dirwalker_InFolder(path_to_folder=args.path2data, prefix='')

    # print(sub_dirNames)

    if not flameMEAN:
        for subject_id, subject_path in zip(sub_dirNames, sub_dirPaths):
            if subject_name != None and subject_id != subject_name:
                continue

            print(f"current subject: {subject_id}")
            path2sequences = os.path.join(subject_path, "sequences")

            list_expNames, list_expPaths = Filehandler.dirwalker_InFolder(path_to_folder=path2sequences, prefix='')

            for exp_name, exp_path in zip(list_expNames, list_expPaths):
                print(f"\t{'-'}current_exp: {exp_name}")
                path2annot=os.path.join(exp_path, "annotations")
                path2npz = os.path.join(path2annot, "tracking", "FLAME2023_v2", "tracked_flame_params.npz")

                path2timesteps = os.path.join(exp_path, "timesteps")

                list_ts_Names, list_ts_Paths = Filehandler.dirwalker_InFolder(path_to_folder=path2timesteps, prefix='frame_')

                for id_timestep, path_timestep in zip(list_ts_Names, list_ts_Paths):
                    print(f"\t\t{'-'}current_ts: {id_timestep}")
                    get_PLY_from_npz(path2npz=path2npz, path2output=path_timestep, id_timestep=id_timestep[-5:], save_name=id_timestep[-5:]+".ply", data_format=data_format)
    else:
        path2npz=None
        path_timestep=None
        id_timestep=None
        get_PLY_from_npz(flame_mean=flameMEAN, path2output="./data/flame", save_name="flame_mean.ply", data_format=data_format)

