
import pyvista as pv
from dreifus.pyvista import render_from_camera
from dreifus.matrix import Pose, Intrinsics, CameraCoordinateConvention, PoseType
import sys, os
import numpy as np
import vtk
import trimesh
from PIL import Image, ImageFilter
from argparse import ArgumentParser
import PIL.ImageOps

# append src directory to the sys.path
from utils.metadata_loader import load_KRT_from_json
from utils.dataset_handler import Filehandler
from utils.common_utils import uptoscale_transform

def input_error_print():
    print("-"*10)
    print("[ERROR] run with 'python *.py --path2data <path2data> --subject_id <subject_id> --exp_id <exp_id> --timestep_id <timestep_id> --server'")
    print("\tWithout specification, based on following .pkl")
    print("-"*10)


def project_ply(args):
    # read args
    try:
        path2data = args.path2data
        subject_id = args.subject_id
        exp_id = args.exp_id
        timestep_id = args.timestep_id
    except:
        input_error_print

    print("-"*10+subject_id+"-"*10)
    # image dimension
    img_dim = (args.img_width, args.img_height) #(1100, 1604)

    path2camjson = os.path.join(path2data, subject_id, "calibration", "camera_params.json")
    # loading corresponding camera callibration json
    cameras = load_KRT_from_json(path2camjson=path2camjson)

    # directory which contains each timestep image directory
    parent_directory = os.path.join(os.path.dirname(path2camjson), os.pardir, "sequences", exp_id, "timesteps")

    print(parent_directory)
    assert os.path.exists(parent_directory)==True

    # paths and names for each timestamp directory
    list_ts_dirNames, list_ts_dirPaths = Filehandler.dirwalker_InFolder(path_to_folder=parent_directory, prefix='frame_')
    assert len(list_ts_dirNames) != 0 and len(list_ts_dirPaths) !=0

    path_to_ts_directory = list_ts_dirPaths[int(timestep_id)]

    path2output = path_to_ts_directory
    if args.facemask == False:
        if args.mesh_alpha:
            if args.mesh_type == "noneck":
                path2output = os.path.join(path2output, "flamemesh_noneck_proj")
                print(path2output)
            elif args.mesh_type == "original":
                path2output = os.path.join(path2output, "flamemesh_proj")
                print(path2output)
        elif args.excluded_mask_alpha:
            path2output = os.path.join(path2output, "excmask_alpha")
            print(path2output)
        else:
            path2output = os.path.join(path2output, 'proj')
    else:
        if args.facemask_background_gt:
            path2output = os.path.join(path2output, 'check_flamemask_proj')
        else:
            path2output = os.path.join(path2output, 'flamemask_proj')
    os.makedirs(path2output, exist_ok=True)
    # paths and names for image directory
    if args.facemask:
        if not args.facemask_background_gt:
            list_im_dirNames, list_im_dirPaths = Filehandler.dirwalker_InFolder(path_to_folder=list_ts_dirPaths[int(timestep_id)], prefix='image')
        else:
            list_im_dirNames, list_im_dirPaths = Filehandler.dirwalker_InFolder(path_to_folder=list_ts_dirPaths[int(timestep_id)], prefix='image')
    else:
        list_im_dirNames, list_im_dirPaths = Filehandler.dirwalker_InFolder(path_to_folder=list_ts_dirPaths[int(timestep_id)], prefix='image')
    assert len(list_im_dirNames) != 0 and len(list_im_dirPaths) !=0

    # get list of image from each timestep image directory
    path_to_image_directory = list_im_dirPaths[0] # assume there is only one folder (named "images") for image at the timestep
    print(f"path to image directory: {path_to_image_directory}")
    if args.facemask:
        if not args.facemask_background_gt:
            list_fNames, list_fPaths = Filehandler.fileWalker_InDirectory(path_to_directory=path_to_image_directory, ext='.jpg')
        else:
            list_fNames, list_fPaths = Filehandler.fileWalker_InDirectory(path_to_directory=path_to_image_directory, ext='.jpg')
    else:
        list_fNames, list_fPaths = Filehandler.fileWalker_InDirectory(path_to_directory=path_to_image_directory, ext='.jpg')
    assert len(list_fNames) != 0 and len(list_fPaths) !=0
        
    print(f"Show the images at {timestep_id}")

    print("Loading camera callibration...")
    print("Number of Cameras: ", len(cameras))

    # virtual frame buffer for offscreen rendering
    pv.start_xvfb()
    # pyvista scene setting
    p = pv.Plotter(window_size = [img_dim[0], img_dim[1]], off_screen = True)

    # background
    if args.facemask:
        facemask_tracked_mesh = os.path.join(path_to_ts_directory, timestep_id+'_colored_facemask.ply')
        facemask_tri_mesh = trimesh.load(facemask_tracked_mesh)
        facemask_mesh = pv.wrap(facemask_tri_mesh)
        p.add_mesh(facemask_mesh, lighting=False, scalars = np.asarray(facemask_tri_mesh.visual.vertex_colors), culling="back", cmap ='binary_r')
        p.remove_scalar_bar()
    elif args.mesh_alpha:
        if args.mesh_type == "noneck":
            tracked_mesh = os.path.join(path_to_ts_directory, timestep_id+'_noneck.ply')
        elif args.mesh_type == "original":
            tracked_mesh = os.path.join(path_to_ts_directory, timestep_id+'.ply')

        tri_mesh = trimesh.load(tracked_mesh)
        mesh = pv.wrap(tri_mesh)
        p.add_mesh(mesh, lighting=False, color="white", culling="none")
    elif args.excluded_mask_alpha:
        tracked_mesh = os.path.join(path_to_ts_directory, timestep_id+'_colored_excmask.ply')
        tri_mesh = trimesh.load(tracked_mesh)
        mesh = pv.wrap(tri_mesh)
        p.add_mesh(mesh, lighting=False, scalars=np.asarray(tri_mesh.visual.vertex_colors), cmap = 'binary_r', culling = "back")
        p.remove_scalar_bar()
    else:
        tracked_mesh = os.path.join(path_to_ts_directory, timestep_id+'.ply')
        tri_mesh = trimesh.load(tracked_mesh)
        mesh = pv.wrap(tri_mesh)
        p.add_mesh(mesh)

    # canonical 2 camera
    path2cano2w = os.path.join(path_to_ts_directory, "flame_cano2world.txt")
    cano2world = uptoscale_transform(path2txt=path2cano2w)

    counter = 0
    for key in cameras.keys():
        camera_extrinsic = cameras[key]["extrin"]
        if not (camera_extrinsic.shape[0] == 4 and camera_extrinsic.shape[1] == 4):
            # homogeneous coordinate
            camera_extrinsic = np.vstack((camera_extrinsic, [0,0,0,1]))
        
        # cano 2 world 2 camera
        camera_extrinsic = np.matmul(camera_extrinsic, cano2world)
        # U, S, Vh = np.linalg.svd(camera_extrinsic[:3, :3])
        # camera_extrinsic[:3, :3] = np.matmul(U, Vh)

        camera_intrinsic = cameras[key]["intrin"]

        pose = Pose(camera_extrinsic, pose_type = PoseType.WORLD_2_CAM, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV)
        intrinsics = Intrinsics(camera_intrinsic/args.img_scale)

        # print("--------------------------------------")
        print(f"camear id: {key}")
        # print(f"camera_extrinsic\n {camera_extrinsic}")
        # print(f"camera_intrinsic\n {camera_intrinsic}")

        # background
        if args.facemask or args.mesh_alpha or args.excluded_mask_alpha:
            if args.facemask_background_gt:
                background_img_name = list_fPaths[list_fNames.index("cam_"+str(key)+'.jpg')]
                p.add_background_image(background_img_name)
                path2save = os.path.join(path2output, "cam_"+key+"_wgt.png")
            else:                
                if not args.overlayed_with_color_segumentation:
                    path2save = os.path.join(path2output, "cam_"+key+".png")
                    # path2background_save = os.path.join(path_to_ts_directory, 'b_background')
                    # os.makedirs(path2background_save, exist_ok= True)
                    # background_img_name = os.path.join(path2background_save,"cam_"+str(key)+"_background.png")
                    # if not os.path.exists(background_img_name):
                    #     recreate_black_background = Image.fromarray(np.zeros_like(np.asarray(Image.open(path2color_mask))))
                    #     recreate_black_background.save(background_img_name)
                    # p.add_background_image(background_img_name)
                    p.set_background("black", top="black")
                else:
                    color_mask_fnames, color_mask_paths = Filehandler.fileWalker_InDirectory(path_to_directory=path_to_image_directory, ext="_"+str(key)+'.png')
                    print(color_mask_paths)

                    if len(color_mask_paths)>1:
                        path2color_mask = color_mask_paths[0]
                    path2save = os.path.join(path2output, "cam_"+key+"_w_colormask.png")
                    p.add_background_image(path2color_mask)
        
            rendered_img = None    
            rendered_img = render_from_camera(p, pose, intrinsics)
            rendered_img = np.uint8(np.asarray(rendered_img))
            # print(rendered_img.shape)
        
            # if args.facemask:
                # one_hot = np.ones_like(rendered_img)
                # one_hot[...,:3] = rendered_img[...,:3]>0.0
                # rendered_img = rendered_img * (one_hot)

            print(f"rendered img shape\n {np.asarray(rendered_img).shape}")
            #plt
            # plt.subplot(1, 2, 1)
            # plt.axis('off')
            # plt.imshow(rendered_img)
            # plt.title("Rendered image")
            # plt.subplot(1, 2, 2)
            # plt.axis('off')
            # plt.imshow(gt_img)
            # plt.title("GT")
            # plt.savefig(path2save, dpi=128)

            #PIL
            im_PIL = Image.fromarray(rendered_img)
            if args.facemask and args.blur:
                im_PIL = im_PIL.filter(ImageFilter.GaussianBlur(radius = 7)) 

            im_PIL.save(path2save)
            counter = counter +1

            # p.remove_background_image()
        elif args.mask_images:
            if args.mask_type=="facemask":
                path2masked_images = os.path.join(path_to_ts_directory, "filled_masked_images")
            elif args.mask_type == "mesh_noneck":
                path2masked_images = os.path.join(path_to_ts_directory, "filled_noneck_flamemesh_images")
            elif args.mask_type == "original":
                path2masked_images = os.path.join(path_to_ts_directory, "filled_flamemesh_images")

            os.makedirs(path2masked_images, exist_ok=True)
            gt_img = Image.open(os.path.join(path_to_image_directory, "cam_"+str(key)+".jpg"))
            gt_img = gt_img.convert("RGBA")
            # gt_img = np.asarray(gt_img)
            # mask_img = Image.open(os.path.join(path_to_ts_directory, "filled_flamemask_proj_v2", "cam_"+str(key)+'_b.png'))
            if args.mask_type=="facemask":
                mask_img = Image.open(os.path.join(path_to_ts_directory, "flamemask_proj", "cam_"+str(key)+'_b.png'))
            elif args.mask_type == "mesh_noneck":
                mask_img = Image.open(os.path.join(path_to_ts_directory, "flamemesh_noneck_proj", "cam_"+str(key)+'_b.png'))
            elif args.mask_type == "original":
                mask_img = Image.open(os.path.join(path_to_ts_directory, "flamemesh_proj", "cam_"+str(key)+'_b.png'))

            mask_img = mask_img.convert("L")
            # mask_img = np.asarray(mask_img).astype(bool)
            # masked_image = Image.fromarray(generate_masked_image(mask = mask_img, image = gt_img))
            gt_img.putalpha(mask_img)

            # np_background=np.zeros((1604, 1100, 3)).astype(np.uint8)
            np_background=np.zeros((img_dim[1], img_dim[0], 3)).astype(np.uint8)
            print(np_background.shape)

            if args.bg_color == "b":
                background = Image.fromarray(np_background)
                background = background.convert("RGBA")
                background.paste(gt_img, mask = gt_img)
                background = background.convert("RGB")
                background.save(os.path.join(path2masked_images, "cam_"+key+"_b.png"))
            elif args.bg_color == "w":
                invers_mask = PIL.ImageOps.invert(mask_img).convert("L")
                wwb_img = Image.new("RGBA", img_dim)
                wwb_img.paste(gt_img, (0, 0), mask_img)
                _bg = Image.new("RGBA", wwb_img.size, "WHITE")
                _bg.paste(wwb_img, (0, 0), wwb_img)
                _bg.save(os.path.join(path2masked_images, "cam_"+key+"_w.png"))
        else:
            path2save = os.path.join(path2output, "cam_"+key+".png")
            p.add_background_image(list_fPaths[list_fNames.index("cam_"+str(key)+'.jpg')])
            
            rendered_img = None    
            rendered_img = render_from_camera(p, pose, intrinsics)
            rendered_img = np.uint8(np.asarray(rendered_img))
        
            # if args.facemask:
            #     one_hot = np.ones_like(rendered_img)
            #     one_hot[...,:3] = rendered_img[...,:3]>0
            #     rendered_img = rendered_img * (one_hot)

            print(f"rendered img shape\n {np.asarray(rendered_img).shape}")
            #plt
            # plt.subplot(1, 2, 1)
            # plt.axis('off')
            # plt.imshow(rendered_img)
            # plt.title("Rendered image")
            # plt.subplot(1, 2, 2)
            # plt.axis('off')
            # plt.imshow(gt_img)
            # plt.title("GT")
            # plt.savefig(path2save, dpi=128)

            #PIL
            im_PIL = Image.fromarray(rendered_img)
            im_PIL.save(path2save)
            counter = counter +1

            p.remove_background_image()


if __name__ == "__main__":
    parser = ArgumentParser(description="projection of ply")
    parser.add_argument('--path2data', type = str, default='/mnt/hdd/dataset/269-single-timestep-EXP-1-head')
    parser.add_argument('--subject_id', type = str, default='test')
    parser.add_argument('--exp_id', type = str, default = 'EXP-1-head')
    parser.add_argument('--timestep_id', type = str, default = '00000')
    parser.add_argument('--path2output', type = str, default=  os.path.join(os.getcwd(),'outputs'))
    parser.add_argument('--overlayed_with_color_segumentation', action='store_true', default=False)
    parser.add_argument('--facemask', action='store_true', default=False)
    parser.add_argument('--facemask_background_gt', action='store_true', default=False)
    parser.add_argument('--mask_images', action='store_true', default=False)
    parser.add_argument('--mesh_alpha', action='store_true', default=False)
    parser.add_argument('--mesh_type', type = str, default="noneck")
    parser.add_argument('--mask_type', type=str, default="facemask")
    parser.add_argument('--bg_color', type = str, default="b")
    parser.add_argument('--excluded_mask_alpha', action='store_true', default=False)
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--blur', action="store_true", default=False)
    parser.add_argument('--img_width', type=int, default=1100)
    parser.add_argument('--img_height', type=int, default=1604)
    parser.add_argument('--img_scale', type = float, help="check the scalar from (cx, cy) = s*(w, h)", default=2.0)
    args = parser.parse_args(sys.argv[1:])

    if args.overlayed_with_color_segumentation == True:
        args.facemask = True

    if args.all:
        list_subject_ids, list_subject_paths=Filehandler.dirwalker_InFolder(path_to_folder=args.path2data, prefix='')
        print(f"the number of subjects: {len(list_subject_ids)}")
        for id in list_subject_ids:
            args.subject_id = id
            args.path2output =  os.path.join(os.getcwd(),'outputs')
            args.path2output = os.path.join(args.path2output, id)
            project_ply(args)
    else:
        args.path2output = os.path.join(args.path2output, args.subject_id)
        project_ply(args)





