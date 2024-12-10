
import numpy as np
import torch
import trimesh
import os,sys
from .ply_helper import save_ply
from .dataset_handler import Filehandler
from .pkl_handler import load_binary_pickle


class FLAME:
    def __init__(self, path2ply, path2mask_pkl=os.path.join(os.getcwd(), "data", "FLAME_masks.pkl")):
        self.mask_info = load_binary_pickle(path2pkl=path2mask_pkl)
        self.mesh = trimesh.load(path2ply)
        self.vertices = np.asarray(self.mesh.vertices)
        self.vertex_normals = np.asarray(self.mesh.vertex_normals)
        self.faces = np.asarray(self.mesh.faces)
        self.vertex_colors = np.asarray(self.mesh.visual.vertex_colors)[:, :3]
        self.new_mouth_faces = np.array([
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
        self.new_mouth_vert_idx, _ = np.unique(self.new_mouth_faces, return_counts=True)
        print(self.new_mouth_vert_idx)
        print(self.new_mouth_vert_idx.shape)
        self.new_mouth_vert_coords = self.vertices[self.new_mouth_vert_idx]
        # add new mounth region to the lips region
        self.mask_info["new_mouth"] = self.new_mouth_vert_idx



def npz2torch_cuda(path2npz, id_timestep:int = 0):
    s, exp, p, n, eyes, t, rot_mat, scale = npz2params(path2npz=path2npz, id_timestep=id_timestep)

    shape_params =  torch.tensor(s, dtype=torch.float32).cuda()
    expression_params =  torch.tensor(exp, dtype=torch.float32).cuda()
    pose_params =  torch.tensor(p, dtype=torch.float32).cuda()
    neck_pose =  torch.tensor(n, dtype=torch.float32).cuda()
    eye_pose =  torch.tensor(eyes, dtype=torch.float32).cuda()
    transl =  torch.tensor(t, dtype=torch.float32).cuda()
    rot_mat = torch.tensor(rot_mat, dtype = torch.float32).cuda()
    scale = torch.tensor(scale, dtype = torch.float32).cuda()

    return shape_params, expression_params, pose_params, neck_pose, eye_pose, transl, rot_mat, scale

def npz2torch(path2npz, id_timestep:int = 0):
    s, exp, p, n, eyes, t, rot_mat, scale = npz2params(path2npz=path2npz, id_timestep=id_timestep)

    shape_params =  torch.tensor(s, dtype=torch.float32)
    expression_params =  torch.tensor(exp, dtype=torch.float32)
    pose_params =  torch.tensor(p, dtype=torch.float32)
    neck_pose =  torch.tensor(n, dtype=torch.float32)
    eye_pose =  torch.tensor(eyes, dtype=torch.float32)
    transl =  torch.tensor(t, dtype=torch.float32)
    rot_mat = torch.tensor(rot_mat, dtype = torch.float32)
    scale = torch.tensor(scale, dtype = torch.float32)

    return shape_params, expression_params, pose_params, neck_pose, eye_pose, transl, rot_mat, scale


def npz2params(path2npz, id_timestep:int = 0):
    # loading parameters from flame fitting
    array_from_npz = np.load(path2npz)

    # print(f"type: {type(array_from_npz)}")
    # print(f"keys: {len(array_from_npz.keys())}")

    shape_params=None
    expression_params=None
    pose_params=None #:3-> global rotation(rotation), 3:-> jaw rotation
    neck_pose=None
    eye_pose=None
    transl=None
    global_rotation = None
    jaw_rotation = None

    its = np.argwhere(array_from_npz["frames"]==id_timestep)
    if len(its) > 1:
        its = its[0]
    print(f"index of timestep of interest: {its}")
    
    
    for key in array_from_npz.keys():
        # print(f"key: {key}")
        # print(f"shape: {array_from_npz[key].shape}")

        if key == "shape":
            shape_params = array_from_npz[key].reshape(1, -1)
            # print(f"shape: {shape_params.shape}")
        elif key == "expression":
            expression_params = array_from_npz[key][its].reshape(1, -1)
            # print(f"expression: {expression_params.shape}")
        elif key == "rotation":
            global_rotation = np.zeros_like(array_from_npz[key][its].reshape(1, -1))
            # print(f"global_rot: {global_rotation.shape}")
        elif key == "translation":
            transl = array_from_npz[key][its].reshape(1, -1)
            # print(f"translation: {transl.shape}")
        elif key == "jaw":
            jaw_rotation = array_from_npz[key][its].reshape(1, -1)
            # print(f"jaw_rot: {jaw_rotation.shape}")
        elif key == "neck":
            neck_pose = array_from_npz[key][its].reshape(1, -1)
            # print(f"neck_pose: {neck_pose.shape}")
        elif key == "eyes":
            eye_pose = array_from_npz[key][its].reshape(1, -1)
            # print(f"eye_pose: {eye_pose.shape}")
        elif key == "rotation_matrices":
            rotation_mat = array_from_npz[key][its].reshape(3, 3)
        elif key == "scale":
            scale = array_from_npz[key][its].reshape(-1)

    # pose[:, :3] = global_rot
    # pose[:, 3:] = jaw_rot
    pose_params = np.hstack((global_rotation, jaw_rotation)).reshape(1,-1)
    assert (global_rotation == pose_params[:, :3]).all() and (jaw_rotation == pose_params[:, 3:]).all()
    # print(f"pose: {pose_params.shape}")
    
    return shape_params, expression_params, pose_params, neck_pose, eye_pose, transl, rotation_mat, scale



def facemask_assignment(path2pkl='../data/FLAME_masks.pkl', mask_regions = ['new_mouth','face','left_eyeball','right_eyeball','lips','nose','left_eye_region','right_eye_region'], path2ply='../data/flame_2023.obj'):
    # 14 regions in total
    
    total_regions = ['new_mouth','eye_region', 'neck', 'left_eyeball', 'right_eyeball', 'right_ear', 'right_eye_region', 'forehead', 'lips', 'nose', 'scalp', 'boundary', 'face', 'left_ear', 'left_eye_region']
    add_regions = ['boundary']
    excluded_regions = []

    for region in total_regions:
        if region not in mask_regions:
            excluded_regions.append(region)

    print(excluded_regions)
    
    # face mask region -> 3 regions (left_eyeball, face, right_eyeball)
    # assign green color (0, 255, 0) to the corresponding vertices

    flame_mesh = FLAME(path2ply=path2ply)
    mask_info = flame_mesh.mask_info
    flame_mesh.mesh = flame_mesh.mesh
    vertices = flame_mesh.vertices
    vertex_normals = flame_mesh.vertex_normals
    faces = flame_mesh.faces
    print(f"The number of faces: {faces.shape}")

    vertex_colors = np.zeros_like(flame_mesh.vertex_colors)
    excluded_vertex_colors = np.zeros_like(vertices)
    assign_color = np.asarray([0, 255, 0], dtype = np.uint8)
    flipped_assign_color = np.asarray([0, 0, 0], dtype = np.uint8)

    facemask_region_vertex_id = []
    print("-"*10+"mask regions"+"-"*10)
    for key in mask_info:
        if key in mask_regions:
            print(key)
            print(f"The number of corresponding vertex: {len(mask_info[key])}")
            for vertex_id in mask_info[key]:
                vertex_colors[vertex_id] = assign_color
                facemask_region_vertex_id.append(vertex_id)

    for key in mask_info:
        # print(key)
        # print(f"The number of corresponding vertex: {len(mask_info[key])}")
        for vertex_id in mask_info[key]:
            excluded_vertex_colors[vertex_id] = assign_color

    for key in mask_info:
        if key in mask_regions:
            # print(key)
            # print(f"The number of corresponding vertex: {len(mask_info[key])}")
            for vertex_id in mask_info[key]:
                excluded_vertex_colors[vertex_id] = flipped_assign_color
        elif key in add_regions:
            for vertex_id in mask_info[key]:
                excluded_vertex_colors[vertex_id] = flipped_assign_color

    # path2npz: /exp_folder/annotations/tracking/FLAME2023_v2/.npz
    # path2output: /exp_folder/timesteps/frame_00000/00000.ply
    path2output = os.path.dirname(path2ply)
    print(f"path2output: {path2output}")
    save_name =  path2output[-5:]+"_colored_facemask.ply"
    print(f"save name: {save_name}")
    save_ply(os.path.join(path2output, save_name), vertices = vertices, faces = faces, vertex_normals = vertex_normals, vertex_colors = vertex_colors, only_points=False)

    save_name_exc =  path2output[-5:]+"_colored_excmask.ply"
    print(f"save name excluded: {save_name_exc}")
    save_ply(os.path.join(path2output, save_name_exc), vertices = vertices, faces = faces, vertex_normals = vertex_normals, vertex_colors = excluded_vertex_colors, only_points=False)

    noneck_exc_vertex_colors = np.array([[0, 255, 0] for i in range(vertex_colors.shape[0])], dtype=np.uint8)
    print(noneck_exc_vertex_colors.shape)

    for key in mask_info:
        if key in add_regions:
            for vertex_id in mask_info[key]:
                noneck_exc_vertex_colors[vertex_id] = flipped_assign_color

    save_name_noneck_exc =  path2output[-5:]+"_colored_noneck.ply"
    print(f"save name excluded: {save_name_noneck_exc}")
    save_ply(os.path.join(path2output, save_name_noneck_exc), vertices = vertices, faces = faces, vertex_normals = vertex_normals, vertex_colors = noneck_exc_vertex_colors, only_points=False)

    colored_facemask_mesh = trimesh.load(os.path.join(path2output, save_name))
    face_mask = (colored_facemask_mesh.visual.face_colors == [0,255,0,255]).all(axis=1)
    # print(face_mask.shape)
    vert_mask = (colored_facemask_mesh.visual.vertex_colors == [0,255,0,255]).all(axis=1)
    print(vert_mask.shape)

    facemask_vertID_list = np.where(vert_mask==True)[0]
    print(facemask_vertID_list.shape)

    if not os.path.exists(os.path.join(os.path.dirname(path2pkl), "FLAME_mask_vertexID.npy")):
        np.save(file = os.path.join(os.path.dirname(path2pkl), "FLAME_mask_vertexID.npy"), arr=facemask_vertID_list)


    colored_facemask_mesh.update_vertices(vert_mask)
    facemask_vertices = colored_facemask_mesh.vertices
    # print(facemask_vertices.shape)
    colored_facemask_mesh.update_faces(face_mask)
    facemask_faces = colored_facemask_mesh.faces
    # print(facemask_faces.shape)

    facemask_trimesh = trimesh.Trimesh(vertices=facemask_vertices, faces = facemask_faces)

    facemask_normals = facemask_trimesh.vertex_normals
    # print(facemask_normals.shape)
    facemask_vertex_colors = np.uint8(np.ones_like(facemask_trimesh.visual.vertex_colors)*255)
    # print(facemask_vertex_colors.shape)

    save_facemask_fname = os.path.join(path2output, path2output[-5:]+'_facemask.ply')
    print(f"save_facemask_fname: {save_facemask_fname}")
    save_ply(save_facemask_fname, vertices=facemask_vertices, faces = facemask_faces, vertex_normals=facemask_normals, vertex_colors=facemask_vertex_colors, only_points=False)

    colored_excmask_mesh = trimesh.load(os.path.join(path2output, save_name_exc))
    face_excmask = (colored_excmask_mesh.visual.face_colors == [0,255,0,255]).all(axis=1)
    # print(face_mask.shape)
    vert_excmask = (colored_excmask_mesh.visual.vertex_colors == [0,255,0,255]).all(axis=1)
    # print(vert_mask.shape)

    colored_excmask_mesh.update_vertices(vert_excmask)
    excmask_vertices = colored_excmask_mesh.vertices
    # print(facemask_vertices.shape)
    colored_excmask_mesh.update_faces(face_excmask)
    excmask_faces = colored_excmask_mesh.faces
    # print(facemask_faces.shape)

    excmask_trimesh = trimesh.Trimesh(vertices=excmask_vertices, faces = excmask_faces)

    excmask_normals = excmask_trimesh.vertex_normals
    # print(facemask_normals.shape)
    excmask_vertex_colors = np.uint8(np.ones_like(excmask_trimesh.visual.vertex_colors)*255)
    # print(facemask_vertex_colors.shape)

    save_excmask_fname = os.path.join(path2output, path2output[-5:]+'_excmask.ply')
    print(f"save_excmask_fname: {save_excmask_fname}")
    save_ply(save_excmask_fname, vertices=excmask_vertices, faces = excmask_faces, vertex_normals=excmask_normals, vertex_colors=excmask_vertex_colors, only_points=False)

    colored_noneck_mesh = trimesh.load(os.path.join(path2output, save_name_noneck_exc))
    face_noneck = (colored_noneck_mesh.visual.face_colors == [0,255,0,255]).all(axis=1)
    # print(face_mask.shape)
    vert_noneck = (colored_noneck_mesh.visual.vertex_colors == [0,255,0,255]).all(axis=1)
    # print(vert_mask.shape)

    colored_noneck_mesh.update_vertices(vert_noneck)
    noneck_vertices = colored_noneck_mesh.vertices
    # print(facemask_vertices.shape)
    colored_noneck_mesh.update_faces(face_noneck)
    noneck_faces = colored_noneck_mesh.faces
    # print(facemask_faces.shape)

    noneck_trimesh = trimesh.Trimesh(vertices=noneck_vertices, faces = noneck_faces)

    noneck_normals = noneck_trimesh.vertex_normals
    # print(facemask_normals.shape)
    noneck_vertex_colors = np.uint8(np.ones_like(noneck_trimesh.visual.vertex_colors)*255)
    # print(facemask_vertex_colors.shape)

    save_noneck_fname = os.path.join(path2output, path2output[-5:]+'_noneck.ply')
    print(f"save_noneck_fname: {save_noneck_fname}")
    save_ply(save_noneck_fname, vertices=noneck_vertices, faces = noneck_faces, vertex_normals=noneck_normals, vertex_colors=noneck_vertex_colors, only_points=False)


    facemask_mesh = trimesh.load(save_facemask_fname) 
    # # save two subdivision versions
    subd_verts, subd_faces, subd_dict = trimesh.remesh.subdivide(vertices=facemask_mesh.vertices, faces = facemask_mesh.faces, face_index=None, vertex_attributes={"normal": facemask_mesh.vertex_normals, "color":facemask_mesh.visual.vertex_colors}, return_index=True)
    subd_vertex_normals = subd_dict["normal"]
    subd_vertex_colors = subd_dict["color"]

    save_facemasksubd_fname = os.path.join(path2output, path2output[-5:]+'_facemask_subd.ply')
    print(f"save_facemasksubd_fname: {save_facemasksubd_fname}")
    save_ply(os.path.join(path2output, path2output[-5:]+'_facemask_subd.ply'), vertices=subd_verts, faces = subd_faces, vertex_normals=subd_vertex_normals, vertex_colors=subd_vertex_colors, only_points=False)

    subd_verts, subd_faces, subd_dict = trimesh.remesh.subdivide(vertices=subd_verts, faces = subd_faces, face_index=None, vertex_attributes={"normal": subd_vertex_normals, "color":subd_vertex_colors}, return_index=True)
    subd_vertex_normals = subd_dict["normal"]
    subd_vertex_colors = np.uint8(np.ones_like(subd_dict["color"])*255)

    save_facemasksubd2_fname = os.path.join(path2output, path2output[-5:]+'_facemask_subd2.ply')
    print(f"save_facemasksubd2_fname: {save_facemasksubd2_fname}")
    save_ply(os.path.join(path2output, path2output[-5:]+'_facemask_subd2.ply'), vertices=subd_verts, faces = subd_faces, vertex_normals=subd_vertex_normals, vertex_colors=subd_vertex_colors, only_points=False)

    return face_mask

if __name__ == "__main__":
    path2pkl = '../data/FLAME_masks.pkl'
    path2ply = '/mnt/hdd/dataset/Ryoto-single-timestep/017/sequences/SEN-10-port_strong_smokey/timesteps/frame_00000/00000.ply'
    mask_info = facemask_assignment(path2pkl=path2pkl, path2ply=path2ply)

    # print(mask_info)