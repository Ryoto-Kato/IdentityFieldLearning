    

import numpy as np
import torch

def get_verts_from_codes(flamelayer, config, shape_params, expression_params, other_codes, facemask=True):

    [pose_params, neck_pose, eye_pose, transl, rot_mat, scale] = other_codes
    pose_params = pose_params.cuda()
    neck_pose = neck_pose.cuda()
    eye_pose = eye_pose.cuda()
    transl = transl.cuda()
    rot_mat = rot_mat.cuda()
    scale = scale.cuda()
    
    scale_mat = torch.eye(3).cuda()
    scale_mat *=scale

    # Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework
    flame_vertice, flame_landmark = flamelayer(shape_params = shape_params, expression_params = expression_params, pose_params = pose_params, neck_pose = neck_pose, eye_pose= eye_pose, transl = None if config.separate_transform else transl)
    print(flame_vertice.size(), flame_landmark.size())

    # Visualize Landmarks
    # This visualises the static landmarks and the pose dependent dynamic landmarks used for RingNet project
    vertices = flame_vertice
    scale_mat = scale_mat

    # print("vertices:", vertices.shape)
    # apply scale to the flame mesh in canonical space
    final_vertices = torch.matmul(vertices, scale_mat.T).float().reshape(-1, 3)
    # print("final_vertices:", final_vertices.shape)
    
    if facemask:
        path2vertID = "/home/kato/Photorealistic-3DMM/IdentityLearning_3DGS/samples/flame_meshes/FLAME_mask_vertexID.npy"
        facemask_vertmask = list(np.load(path2vertID))
        final_vertices = final_vertices[facemask_vertmask]
    # print("final_vertices:", final_vertices.shape)

    # print("Initialize vertices: ", final_vertices.shape)

    return final_vertices





        
    