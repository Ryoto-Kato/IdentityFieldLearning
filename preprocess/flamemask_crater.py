
import sys
import os
import numpy as np
from argparse import ArgumentParser

# append src directory to the sys.path
from utils.dataset_handler import Filehandler
from utils.FLAME_helper import facemask_assignment


def flamefacemask_generator(args):
    # read args
    path2pkl = os.path.join(os.getcwd(), "data", "FLAME_masks.pkl")
    path2data = args.path2data
    subject_id = args.subject_id
    exp_id = args.exp_id
    timestep_id = args.timestep_id

    # image dimension
    path2subj_dir= os.path.join(path2data, subject_id)

    # directory which contains each timestep image directory
    parent_directory = os.path.join(path2subj_dir, "sequences", exp_id, "timesteps")

    print(parent_directory)
    assert os.path.exists(parent_directory)==True

    # paths and names for each timestamp directory
    list_ts_dirNames, list_ts_dirPaths = Filehandler.dirwalker_InFolder(path_to_folder=parent_directory, prefix='frame_')
    assert len(list_ts_dirNames) != 0 and len(list_ts_dirPaths) !=0

    path_to_ts_directory = list_ts_dirPaths[int(timestep_id)]

    path2flame_ply = os.path.join(path_to_ts_directory, f'{timestep_id}.ply')
    # print(path2flame_ply)

    # total region
    # total_regions = ['new_mouth','eye_region', 'neck', 'left_eyeball', 'right_eyeball', 'right_ear', 'right_eye_region', 'forehead', 'lips', 'nose', 'scalp', 'boundary', 'face', 'left_ear', 'left_eye_region']
    
    # facemask region    
    # mask_regions = ['new_mouth','face','left_eyeball','right_eyeball','lips','nose','left_eye_region','right_eye_region']
    
    # head region *facemask+hair region
    # mask_regions = ['new_mouth','face','left_eyeball','right_eyeball','lips','nose','left_eye_region','right_eye_region']

    _ = facemask_assignment(path2pkl = path2pkl, path2ply = path2flame_ply)
    assert os.path.exists(os.path.join(os.path.dirname(path2flame_ply), f"{timestep_id}_facemask.ply"))
    

if __name__ == "__main__":
    parser = ArgumentParser(description="projection of ply")
    parser.add_argument('--path2data', type = str, default='/mnt/hdd/dataset/269-single-timestep-EXP-1-head')
    parser.add_argument('--subject_id', type = str, default='test')
    parser.add_argument('--exp_id', type = str, default = 'EXP-1-head')
    parser.add_argument('--timestep_id', type = str, default = '00000')
    parser.add_argument('--all', action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])

    if args.all:
        list_subject_ids, list_subject_paths=Filehandler.dirwalker_InFolder(path_to_folder=args.path2data, prefix='')
        print(f"the number of subjects: {len(list_subject_ids)}")
        for id in list_subject_ids:
            args.subject_id = id
            flamefacemask_generator(args)

        ## average flame mesh
        path2pkl = os.path.join(os.getcwd(), "data", "FLAME_masks.pkl")
        path2flame_ply = os.path.join(os.path.dirname(path2pkl), "flame", f'flame.ply')
        # print(path2flame_ply)
        _ = facemask_assignment(path2pkl = path2pkl, path2ply = path2flame_ply)
    else:
        flamefacemask_generator(args)

    ## average flame mesh
    # path2pkl = os.path.join(os.getcwd(), "data", "FLAME_masks.pkl")
    # path2flame_ply = os.path.join(os.path.dirname(path2pkl), "flame", f'flame.ply')
    # print(path2flame_ply)
    # _ = facemask_assignment(path2pkl = path2pkl, path2ply = path2flame_ply)






