import os,sys
import numpy as np
from typing import NamedTuple
from PIL import Image

# utils of 3DGS
from utils.graphics_utils import focal2fov
from utils.dataset_handler import Filehandler
from scene.dataset_readers import CameraInfo, getNerfppNorm, SceneInfo, fetchPly
from utils.metadata_loader import load_KRT_from_json
from utils.common_utils import uptoscale_transform

# adaption to the NeRSemble dataset
# .
# ├── calibration
# │   ├── calibration_result.json
# │   ├── camera_params.json
# │   ├── config.json
# │   └── timestep_order.json
# └── sequences
#     └── expression-name
#         ├── annotations
#         │   └── tracking
#         │       └── FLAME2023_v2
#         │           └── tracked_flame_params.npz
#         └── timesteps
#             └── frame_00000
#                 ├── 00000_facemask.ply
#                 ├── 00000.ply
#                 ├── alpha_map
#                 │   ├── cam_220700191.png
#                 ├── b_background
#                 │   ├── cam_220700191_background.png
#                 ├── colmap
#                 │   └── depth_maps_geometric
#                 │       └── 16
#                 │           ├── cam_220700191.png
#                 ├── facer_segmentation_masks
#                 │   ├── color_segmentation_cam_220700191.png
#                 │   ├── segmentation_cam_220700191.png
#                 ├── flamemask_proj
#                 │   ├── cam_220700191_b.pngrender_numGauss
#                 ├── images-2x
#                 │   ├── cam_220700191.jpg
#                 └── proj
#                     ├── cam_220700191.png


def readNersembleSceneInfo(subject_id="017", expName="SEN-10-port_strong_smokey", frame_counter = 0, ALLcam=False, scale = 1.0, Center=False, render_numGauss = None, subd = -1, full_head=False, gt_type = "flame", parts = None, Two = False, Triple=False):
    ID = subject_id
    first_time_stamp = "00000"
    
    # [TODO] set the path to re-structured mutl-face dataset
    # path_to_dataset = os.path.join(os.getcwd(), os.pardir, "dataset")
    path_to_dataset = os.path.join("/mnt/hdd", "dataset", "269-single-timestep-EXP-1-head")
    path_to_subject = os.path.join(path_to_dataset, subject_id)
    path_to_metadata = os.path.join(path_to_subject, "calibration")

    # Get the directory containing images at each time stamp in the expression folder 
    list_TimeStampDirNames, list_TimeStampDir_Paths = Filehandler.dirwalker_InFolder(path_to_folder=os.path.join(path_to_subject, "sequences", expName, "timesteps"), prefix='frame_')

    # Get the name of first frame time stamp
    time_stamp = list_TimeStampDirNames[frame_counter][:6]
    path_to_tsdir = list_TimeStampDir_Paths[frame_counter]
    # print("Time stamp: ", time_stamp)

    # path to original tracked mesh 
    # ply_path = os.path.join(path_to_tsdir, time_stamp+'.ply')
    
    # path to subdivided ply instead of originall tracked mesh
    # ply_path = os.path.join(path_to_tsdir, f"{frame_counter:05}"+'_subd.ply')
    # ply_path = os.path.join(path_to_tsdir, f"{frame_counter:05}"+'_subd2.ply')
    # facemask_path = os.path.join(path_to_tsdir, f"{frame_counter:05}"+'_facemask.ply')
    # facemask_path = os.path.join(path_to_tsdir, f"{frame_counter:05}"+'_facemask_subd.ply')

    
    if subd < 0:
        facemask_path = os.path.join(path_to_tsdir, f"{frame_counter:05}"+'_facemask_subd3.ply')
    else:
        if subd == 0:
            facemask_path = os.path.join(path_to_tsdir, f"{frame_counter:05}"+'_facemask.ply')
        elif subd == 1:
            facemask_path = os.path.join(path_to_tsdir, f"{frame_counter:05}"+'_facemask_subd.ply')
        elif subd == 2:
            facemask_path = os.path.join(path_to_tsdir, f"{frame_counter:05}"+'_facemask_subd2.ply')
        

    ave_path = "/home/kato/Photorealistic-3DMM/DeformationLearning_3DGS/samples/flame_meshes/flame_facemask_subd.ply"
    meta_cameras = load_KRT_from_json(os.path.join(path_to_metadata, "camera_params.json"))
    # canonical 2 camera
    path2cano2w = os.path.join(path_to_tsdir, "flame_cano2world.txt")
    cano2world = uptoscale_transform(path2txt=path2cano2w)


    # print(world2cano)

    # eval
    # eval = False
    llffhold = 16 # where i%8==0: test_cam_infos = cam_infos[i]  

    # get all camera configuration as train cameras
    Train_SelectedCAM = meta_cameras.keys()

    cam_infos = []

    sys.stdout.write("loading camera\n")
    index_of_center_cam = -1
    index_of_left_cam = -1
    index_of_right_cam = -1
    
    for i, cam_name in enumerate(meta_cameras.keys()):
        camera_id = cam_name
        if "222200037" in camera_id: 
            index_of_center_cam = i

        if "222200036" in camera_id: #222200042, 222200046, 222200036
            index_of_right_cam = i
        
        if "222200049" in camera_id: #221501007, 222200049
            index_of_left_cam = i

        cam_name = "cam_"+str(camera_id)

        # image scale compared to original capture
        meta_cameras[camera_id]["intrin"] = meta_cameras[camera_id]["intrin"]*scale        
        # sys.stdout.write("loading camera {}/{}".format(i+1, len(meta_cameras.keys())))
        # sys.stdout.write("_Name: {}".format(cam_name))
        # sys.stdout.flush()
        camera_extrinsic= meta_cameras[camera_id]["extrin"]
        
        # cano 2 world 2 camera
        camera_extrinsic = np.matmul(camera_extrinsic, cano2world)
        U, S, Vh = np.linalg.svd(camera_extrinsic[:3, :3])
        camera_extrinsic[:3, :3] = np.matmul(U, Vh)

        # load extrinsic
        trans_R = np.transpose(camera_extrinsic[:3, :3]) #W2C to C2W #3x3 mat
        T = np.array(camera_extrinsic[:3, 3:4]).ravel() #vector with 3 dims

        # load intrinsic
        focal_x, focal_y = meta_cameras[camera_id]["intrin"][0, 0], meta_cameras[camera_id]["intrin"][1, 1]
        cx, cy = meta_cameras[camera_id]["intrin"][0, 2], meta_cameras[camera_id]["intrin"][1, 2]

        # load image
        # image_name = cam_name+'.jpg'
        # image_path = os.path.join(path_to_tsdir, "images-2x", image_name)
        image_name = cam_name
        image_path = None
        if not full_head:
            if gt_type == "torso":
                image_path = os.path.join(path_to_tsdir, "filled_masked_images", "facer", image_name+'_w.png')
            else:
                image_path = os.path.join(path_to_tsdir, "filled_masked_images", image_name+'.png')
        else:
            if gt_type == "torso":
                if parts == None:
                    if "flame" not in subject_id:
                        image_path = os.path.join(path_to_tsdir, "filled_torso_mask_by_gt", image_name+'_w.png')
                    else:
                        image_path = os.path.join(path_to_tsdir, "images", image_name+'.jpg')
                else:
                    if parts == "face":
                        image_path = os.path.join(path_to_tsdir, "filled_masked_images", "facer", image_name+'_w.png')                
                    elif parts == "hair_torso":
                        image_path = os.path.join(path_to_tsdir, "filled_torso_face_mask_by_gt", image_name+'_w.png')
                    elif parts == "face_neck":
                        image_path = os.path.join(path_to_tsdir, "filled_torso_hair_mask_by_gt", image_name+'_w.png')
            elif gt_type == "flame":
                image_path = os.path.join(path_to_tsdir, "filled_flamemesh_images", image_name+'_w.png')
            elif gt_type == "bg_mat":
                image_path = os.path.join(path_to_tsdir, "fullhead_masked_images", image_name+"_w.png")
            elif gt_type == "original":
                image_path = os.path.join(path_to_tsdir, "images-2x", image_name+'.jpg')

        _image = Image.open(image_path)
        
        # width, height
        width, height = _image.size
        # print(f"- image size (w, h): {width, height}")
        
        # FOV
        FovY = focal2fov(focal_y, height)
        FovX = focal2fov(focal_x, width)

        # Camera Infos
        cam_info = CameraInfo(uid=camera_id, R = trans_R, T = T, K = meta_cameras[camera_id]["intrin"], FovY = FovY, FovX = FovX, image = _image, image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
        # sys.stdout.write('\n')

    # Sort camera infos according to the name of picture
    cam_infos = sorted(cam_infos.copy(), key=lambda x : x.image_name)

    if not Center:
        if not ALLcam:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 == 0]
    else:
        if Two:
            train_cam_infos = [cam_infos[index_of_left_cam], cam_infos[index_of_right_cam]]
            test_cam_infos = cam_infos
        elif Triple:
            train_cam_infos = [cam_infos[index_of_left_cam], cam_infos[index_of_right_cam], cam_infos[index_of_center_cam]]
            test_cam_infos = cam_infos
        else:
            train_cam_infos = [cam_infos[index_of_center_cam]]
            test_cam_infos = cam_infos

    print("Num of train_cams:", len(train_cam_infos))
    # print("Train cams: ", train_cam_infos)
    print("Num of test_cams:", len(test_cam_infos))
    # print("Test cams: ", test_cam_infos)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if render_numGauss == None:
        if subd < 0: # tracking
            pcd = None
        else:   # pcd init
            pcd = fetchPly(facemask_path)
    else:
        if subd < 0: # render_numGauss init
            pcd = fetchPly(facemask_path, render_numGauss=render_numGauss)
        else: # pcd init *should be specified by either render_numGauss or subd
            pcd = fetchPly(facemask_path)

    scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=facemask_path)

    return scene_info, pcd, cano2world

if __name__=="__main__":
    scene_info = readNersembleSceneInfo(ALLcam = True)
    
