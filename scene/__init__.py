#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import random
import json
import h5py
import trimesh
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel, BlendshapeProp, GaussianBlenshapeProp
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from nersemble_dataset_readers import readNersembleSceneInfo
# from multiface_dataset_readers import readMultifaceSceneInfo
from utils.quaterion_slerp import transform_interpolation
from scene.cameras import Camera
from scene.dataset_readers import CameraInfo
from utils.general_utils import PILtoTorch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, load_iteration=None, shuffle=True, resolution_scales=[1.0], expName = "E001_Neutral_Eyes_Open", ALLcam = False, frame_counter = 0, render_blendshape = False, path_to_hdf5 = None, num_Blendshape_compos = 3, dc_type = "pca", sp_interp_cams=None, subject_id=0, scale = 1.0, Center=False, Render_Interp_CAM=False, subd=-1, render_numGauss = None, full_head = False, gt_type = "flame", parts = None, Two=False, Triple=False):
        """b
        :param path: Path to colmap scene main folder.
        
            parts: [None, "face", "hair_torso"]
            None: full-head
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.neutral_gaussians = None

        self.num_Blendshape_compos = num_Blendshape_compos
        self.bp = None
        print(path_to_hdf5)
        self.blendshape_type = "ALL"

        self.dc_type = dc_type

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.interps_cameras = {}
        print(args.source_path)
        
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            # scene_info = readMultifaceSceneInfo(expName=expName, frame_counter=frame_counter, ALLcam=ALLcam)
            scene_info, flame_tracking_pcd, cano2world = readNersembleSceneInfo(expName=expName, frame_counter=frame_counter, subject_id=subject_id, ALLcam=ALLcam, scale = scale, Center = Center, render_numGauss=render_numGauss, subd=subd, full_head=full_head, gt_type = gt_type, parts=parts, Two = Two, Triple = Triple)
        
        self.scene_info = scene_info
        self.flame_tracking_pcd = flame_tracking_pcd
        
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        self.interps_cameras[resolution_scale] = None
        if Render_Interp_CAM:
            if Center:
                target_cameras = self.getTestCameras()
                target_camera_infos = scene_info.test_cameras
            else:
                target_cameras = self.getTrainCameras() #center camera
                target_camera_infos = scene_info.train_cameras

            num_steps = 30
            T = np.linspace(0, 1, num_steps)

            interp_camera_ids = ["cam_222200047", "cam_222200036"]
            # interp_camera_ids = [400013, 400016]
            # interp_camera_ids = sorted(interp_camera_ids, reverse=True)
            
            selected_target_cams = [None for i in range(len(interp_camera_ids))]
            selected_target_cam_infos = [None for i in range(len(interp_camera_ids))]
            
            print(interp_camera_ids)
            counter = 0
            init_cam=None
            init_cam_info=None
            loop = True

            for id, (target_camera, target_camera_info) in enumerate(zip(target_cameras, target_camera_infos)):
                if target_camera.image_name in interp_camera_ids:
                    index_of_caminfo = interp_camera_ids.index(target_camera.image_name)
                    
                    if index_of_caminfo==0:
                        init_cam=target_camera
                        init_cam_info=target_camera_info

                    selected_target_cams[index_of_caminfo] = target_camera
                    selected_target_cam_infos[index_of_caminfo] = target_camera_info
                    counter +=1
            
            # if loop:
            #     for cam, info in zip(selected_target_cams, selected_target_cam_infos):
            selected_target_cams.append(selected_target_cams[0])
            selected_target_cam_infos.append(selected_target_cam_infos[0])
                
            print(selected_target_cams)
            # selected_target_cams_infos = sorted(selected_target_cam_infos.copy(), key=lambda x: x.uid, reverse=True)
            # selected_target_cams = sorted(selected_target_cams.copy(), key = lambda x: x.colmap_id, reverse=True)
            
            interp_common_info = selected_target_cam_infos[0]

            interp_camera_id = 0
            interp_cam_infos = []

            for i, (camera, scene_cam_info) in enumerate(zip(selected_target_cams, selected_target_cam_infos)):
                # print("cameara id:", camera.colmap_id)
                # print("scene_cam_id:", scene_cam_info.uid)
                if i < len(selected_target_cams)-1:
                    next_cam = selected_target_cams[i+1]
                else:
                    next_cam = selected_target_cams[-1]
                # print("next_cam id:", next_cam.colmap_id)
                start_R = camera.R
                start_T = camera.T
                end_R = next_cam.R
                end_T = next_cam.T

                for id_step, t in enumerate(T):
                    interp_camera_id = interp_camera_id+1

                    # if id_step == 0:
                    #     # print(interp_camera_id)
                    #     interp_R, interp_T = camera.R, camera.T
                    #     interp_cam_info = CameraInfo(uid = interp_camera_id, R = interp_R, T = interp_T, K = interp_common_info.K, FovX=interp_common_info.FovX, FovY=interp_common_info.FovY, image = interp_common_info.image, image_path=interp_common_info.image_path, image_name=interp_common_info.image_name, width=interp_common_info.width, height=interp_common_info.height)
                    #     interp_cam_infos.append(interp_cam_info)
                    # elif t == T[-1]:
                    #     # print(interp_camera_id)
                    #     interp_R, interp_T = next_cam.R, next_cam.T
                    #     interp_cam_info = CameraInfo(uid = interp_camera_id, R = interp_R, T = interp_T, K = interp_common_info.K, FovX=interp_common_info.FovX, FovY=interp_common_info.FovY, image = interp_common_info.image, image_path=interp_common_info.image_path, image_name=interp_common_info.image_name, width=interp_common_info.width, height=interp_common_info.height)
                    #     interp_cam_infos.append(interp_cam_info)
                    # else:
                        # print(interp_camera_id)
                    interp_R, interp_T = transform_interpolation(start_R=start_R, end_R=end_R, start_t=start_T, end_t=end_T, time_step=t)
                    interp_cam_info = CameraInfo(uid = interp_camera_id, R = interp_R, T = interp_T, K = interp_common_info.K, FovX=interp_common_info.FovX, FovY=interp_common_info.FovY, image = interp_common_info.image, image_path=interp_common_info.image_path, image_name=interp_common_info.image_name, width=interp_common_info.width, height=interp_common_info.height)
                    interp_cam_infos.append(interp_cam_info)
        
            self.interps_cameras[resolution_scale] = cameraList_from_camInfos(interp_cam_infos, resolution_scale, args)


    def set_gaussians(self, gaussians, index_in_batch=0, xyz=None, fdc=None, frest=None, scale=None, rot=None, opac=None, input_format='learnable_blenshape', UVs=None, uv_map=None):
        """
        input_format: 
            - 'uv': uv_mapping given uv_coordinate
            - 'pcd': point clouds
            - 'learnable_uv': learnable uv

            - xyz: [3007, 3] (squeeze the first dimention)
        """
        self.gaussians = gaussians

        if input_format=='pcd':
            self.gaussians.create_from_pcd(self.scene_info.point_cloud, self.cameras_extent)
        elif input_format == 'learnable_blenshape':
            self.gaussians.create_from_learnable_blenshape(xyz, fdc, frest, scale, rot, opac)
        else:
            if UVs==None and uv_map == None:
                print(f"you need to feed UVs and uvmap for construct 3D Gaussians")
            else:
                self.gaussians.create_from_uv(UVs = UVs, uv_map=uv_map, index_in_batch=index_in_batch, learnable_uv=False)

    def get_gaussians(self):
        return self.gaussians
    
    def update_blendshape(self, coeffs):
        assert coeffs.shape[0] == self.num_Blendshape_compos
        gbp = self.gaussians.blenshape_computation(bp = self.bp, coeffs=coeffs, Ncomps=self.num_Blendshape_compos, dc_type=self.dc_type)
        self.gaussians.update_from_GaussianBlendshapeProp(gbp=gbp)
    
    def update_xyz_blendshape(self, coeffs):
        assert coeffs.shape[0] == self.num_Blendshape_compos
        # neutral_gbp = self.gaussians.blenshape_reset(bp = self.bp)
        # self.gaussians.creat_from_GaussianBlendshapeProp(gbp=neutral_gbp)
        gbp = self.gaussians.blenshape_xyz_computation(bp = self.bp, coeffs=coeffs, Ncomps=self.num_Blendshape_compos, dc_type=self.dc_type)
        self.gaussians.update_from_GaussianBlendshapeProp(gbp=gbp)
    
    def save(self, iteration, subject_id=None, test=False):
        save_ref = False
        if subject_id != None:
            point_cloud_path = os.path.join(self.model_path, "point_cloud", str(subject_id), "iteration_{}".format(iteration))
            save_ref = True
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        
        if save_ref:
            ref_mesh = trimesh.load(self.scene_info.ply_path)
            ref_mesh.export('ref_mesh.ply', encoding='ascii')

        # self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_GaussianProp(os.path.join(point_cloud_path))
        # if test:
        #     print("Load pickel")
        #     gaussian_prop=load_from_memory(path_to_memory=point_cloud_path, pickle_fname="gaussian_prop.pkl")
        #     print(gaussian_prop)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getInterpCameras(self, scale=1.0):
        return self.interps_cameras[scale]
