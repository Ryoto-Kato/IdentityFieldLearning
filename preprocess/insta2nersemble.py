import os, sys
import numpy as np
import torch
import shutil
import trimesh
import json

from argparse import ArgumentParser
from utils.dataset_handler import Filehandler
from utils.ply_helper import save_ply
from utils.obj_helper import OBJ
from PIL import Image, ImageFilter, ImageChops

"""
Conversion from insta (publicly available pre-processed data) to NeRSemble
"""


def insta2nersemble(path2tracking:str, path2save:str, exp="EXP-1-head", timestep = 0):
    """
    convert insta to nersemble dataformat
    """

    os.makedirs(path2save, exist_ok=True)

    # set up folders
    path2cal = os.path.join(path2save, "calibration")
    path2seq = os.path.join(path2save, "sequences")
    os.makedirs(path2cal, exist_ok=True)
    os.makedirs(path2seq, exist_ok=True)
    path2exp = os.path.join(path2seq, exp)
    os.makedirs(path2exp, exist_ok=True)

    path2anno = os.path.join(path2exp, "annotations")
    path2timesteps = os.path.join(path2exp, "timesteps")
    os.makedirs(path2anno, exist_ok=True)
    os.makedirs(path2timesteps, exist_ok=True)

    path2target_ts = os.path.join(path2timesteps, 'frame_{0:05}'.format(timestep))
    os.makedirs(path2target_ts, exist_ok=True)

    path2oroginal_im_dir = os.path.join(path2target_ts)
    os.makedirs(path2oroginal_im_dir, exist_ok=True)

    path2im_dir = os.path.join(path2target_ts, "images")
    os.makedirs(path2im_dir, exist_ok=True)

    path2img = os.path.join(path2tracking, "00000.png")
    shutil.copy(src=path2img, dst = os.path.join(path2oroginal_im_dir, "original.jpg"))

    # add white background
    # get image size (512x512)

    fore_ground = Image.open(path2img)
    _bg = Image.new("RGB", fore_ground.size, "WHITE")
    _bg.paste(fore_ground, (0, 0), fore_ground)
    _bg.convert("RGB")
    _bg.save(os.path.join(path2im_dir, "cam_222200037.jpg"))

    path2mesh = os.path.join(path2tracking, "00000.obj")
    # save mesh and subdivide 
    # mesh = trimesh.load(path2mesh, process=False, maintain_order=True)
    mesh = OBJ(filename=path2mesh)
    ref_mesh = trimesh.load(file_obj="/home/kato/VCAI-utils/data/flame/flame_mean.ply", process=False, maintain_order=True)
    
    vertices = np.asarray(mesh.vertices)
    print("original mesh vertices: ", vertices.shape)
    faces = np.asarray(ref_mesh.faces)
    print("ref faces:", faces.shape)
    vertex_colors = np.zeros([vertices.shape[0], 4])

    mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors, process=False)

    vertices = np.asarray(mesh.vertices)
    print("mesh vertices: ", vertices.shape)
    vertex_normals = np.asarray(mesh.vertex_normals)
    faces = np.asarray(mesh.faces)
    
    save_name = "{0:05}.ply".format(timestep)
    save_ply(os.path.join(path2target_ts, save_name), vertices = vertices, faces = faces, vertex_normals = vertex_normals, vertex_colors = vertex_colors, only_points=False)

    # subd1
    subd_verts, subd_faces, subd_dict = trimesh.remesh.subdivide(vertices=vertices, faces = faces, face_index=None, vertex_attributes={"normal": vertex_normals, "color":vertex_colors}, return_index=True)
    subd_vertex_normals = subd_dict["normal"]
    subd_vertex_colors = subd_dict["color"]
    print(save_name[:-4] +'_subd.ply')
    print("subd verts: ", subd_verts.shape)
    save_ply(os.path.join(path2target_ts, save_name[:-4] +'_subd.ply'), vertices=subd_verts, faces=subd_faces, vertex_normals=subd_vertex_normals, vertex_colors=subd_vertex_colors, only_points=False)

    # subd2
    subd_verts, subd_faces, subd_dict = trimesh.remesh.subdivide(vertices=subd_verts, faces = subd_faces, face_index=None, vertex_attributes={"normal": subd_vertex_normals, "color":subd_vertex_colors}, return_index=True)
    subd_vertex_normals = subd_dict["normal"]
    subd_vertex_colors = subd_dict["color"]
    print(save_name[:-4]  +'_subd2.ply')
    print("subd verts: ",subd_verts.shape)
    save_ply(os.path.join(path2target_ts, save_name[:-4]  +'_subd2.ply'), vertices=subd_verts, faces=subd_faces, vertex_normals=subd_vertex_normals, vertex_colors=subd_vertex_colors, only_points=False)

    # subd3
    subd_verts, subd_faces, subd_dict = trimesh.remesh.subdivide(vertices=subd_verts, faces = subd_faces, face_index=None, vertex_attributes={"normal": subd_vertex_normals, "color":subd_vertex_colors}, return_index=True)
    subd_vertex_normals = subd_dict["normal"]
    subd_vertex_colors = subd_dict["color"]
    print(save_name[:-4]  +'_subd3.ply')
    print("subd verts: ",subd_verts.shape)
    save_ply(os.path.join(path2target_ts, save_name[:-4]  +'_subd3.ply'), vertices=subd_verts, faces=subd_faces, vertex_normals=subd_vertex_normals, vertex_colors=subd_vertex_colors, only_points=False)

    path2json = os.path.join(path2tracking, "transforms.json")
    path2cam_json = os.path.join(path2cal, "camera_params.json")

    camjson = {}
    camjson.update({"world_2_cam": {}})
    camjson["world_2_cam"].update({"222200037": None})
    camjson.update({"intrinsics": None})

    print("cam json: \n", camjson)

    with open(path2json) as f:
        d = json.load(f)

    cx = d["cx"]
    cy = d["cy"]
    fl_x = d["fl_x"]
    fl_y = d["fl_y"]
    RT = d["frames"][timestep]["transform_matrix"]

    cam_intrinsic = np.eye(3)
    cam_intrinsic[0][0] = fl_x
    cam_intrinsic[1][1] = fl_y
    cam_intrinsic[0][2] = cx
    cam_intrinsic[1][2] = cy

    print("cam intrinsic: \n", cam_intrinsic)

    cam_extrinsic = np.eye(4)
    cam_extrinsic = np.array(RT) #c2w
    print("cam extrisic: \n", cam_extrinsic)

    #convert to w2c
    cam_extrinsic = np.linalg.inv(cam_extrinsic)
    print("inversed cam ext: \n", cam_extrinsic)


    camjson["world_2_cam"]["222200037"] = cam_extrinsic.tolist()
    camjson["intrinsics"] = cam_intrinsic.tolist()

    with open(path2cam_json, 'w') as f:
        json.dump(camjson, f)

    cano2world = np.zeros((3, 4))
    cano2world[:3, :3] = np.eye(3)
    print("cano2world: ", cano2world)
    world2cano_txt_path = os.path.join(path2target_ts, "flame_cano2world.txt") 
    with open(world2cano_txt_path, "w") as f:
        for row in cano2world:
            f.write(" ".join([str(x) for x in row])+"\n")


if __name__=="__main__":
    parser = ArgumentParser(description="insta2nersemble")
    parser.add_argument("--path2dir", type = str, default="/mnt/hdd/dataset/INSTA")
    parser.add_argument("--path2save_dir", type = str, default="/mnt/hdd/dataset/269-single-timestep-EXP-1-head")
    args = parser.parse_args(sys.argv[1:])

    list_folder_names, list_folder_paths = Filehandler.dirwalker_InFolder(path_to_folder=args.path2dir, prefix="")

    for folder_name, folder_path in zip(list_folder_names, list_folder_paths):
        path2save = os.path.join(args.path2save_dir, folder_name)
        print(path2save)
        if folder_name == "dataset":
            continue
        insta2nersemble(path2tracking=folder_path, path2save=path2save)


