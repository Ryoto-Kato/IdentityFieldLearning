import numpy as np
import sys, os
import trimesh
import pyrender
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

# backend of off-screen rendering
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

from argparse import ArgumentParser, Namespace

path_to_3WI = os.path.join(os.getcwd(), os.pardir, os.pardir, "3DSSL-WS23_IntuitiveAnimation", "")
sys.path.append(os.path.join(path_to_3WI, 'src'))
from utils.Dataset_handler import Filehandler
from utils.metadata_loader import load_KRT, load_RT
from utils.PLY_helper import tex2vertsColor, gammaCorrect

parser = ArgumentParser(description="Classic rendering")
parser.add_argument('--ID', type=str, default = "6795937")
parser.add_argument('--expName', type = str, default = "E001_Neutral_Eyes_Open")
parser.add_argument('--timestamp', type=str, default = None)
parser.add_argument('--savefig', action='store_true', default=False)
args = parser.parse_args(sys.argv[1:])

ID = args.ID
expName = args.expName
timestamp = args.timestamp
savefig = args.savefig

# python classic_render.py --expName=E001_Neutral_Eyes_Open  --savefig

path_to_dataset = os.path.join(path_to_3WI, "dataset")
path_to_multiface = os.path.join(path_to_dataset, "multiface")
path_to_metadata = os.path.join(path_to_multiface, "meta_data")

# Get the directory containing images at each time stamp in the expression folder
list_TimeStampDirNames, list_TimeStampDir_Paths = Filehandler.dirwalker_InFolder(path_to_folder = os.path.join(path_to_dataset, "COLMAP", ID, expName), prefix='0')

if timestamp == None:
    timestamp = list_TimeStampDirNames[0][:6]
print(f"Expression:{expName}, Timestamp:{timestamp}")

path_to_tsdir = list_TimeStampDir_Paths[0]

assert os.path.exists(path_to_tsdir) == True

ply_path = os.path.join(path_to_tsdir, timestamp+'.obj')
tracked_mesh = trimesh.load(file_obj = ply_path)

# for i, point in enumerate(tracked_mesh.vertices):
#     # print("before:",point)
#     tracked_mesh.vertices[i][1] *= -1
#     tracked_mesh.vertices[i][2] *= -1
#     # print("after:" ,tracked_mesh.vertices[i])

headpose = load_RT(os.path.join(path_to_tsdir, timestamp+"_transform.txt"))
f_KRT = "KRT"
meta_cameras = load_KRT(os.path.join(path_to_metadata, f_KRT))

# upscale factor (rendering_res = image_res * upscale factor)
uf = 1
img_dim = (1334*uf, 2048*uf) # width and height
for i, cam_name in enumerate(meta_cameras.keys()):
    if i == 0:
        camera_id = cam_name

        sys.stdout.write("Reading camera {}/{}\n".format(i+1, len(meta_cameras.keys())))
        sys.stdout.write("Name: {}\n".format(cam_name))
        sys.stdout.flush()

        #load extrinsic
        camera_pose = np.vstack((meta_cameras[camera_id]["extrin"], [0,0,0,1]))
        
        #convert camera extrinsic convention (openCV) to OPENGL
        print("before:", camera_pose)
        
        camera_pose[:3, 1:3] *= -1.0

        print("after:", camera_pose)
        
        # load intrinsicshape
        focal_x, focal_y = meta_cameras[camera_id]["intrin"][0, 0], meta_cameras[camera_id]["intrin"][1, 1]
        cx, cy = meta_cameras[camera_id]["intrin"][0, 2], meta_cameras[camera_id]["intrin"][1, 2]

        #load image
        image_name = str(camera_id)+'.png'
        image_path = os.path.join(path_to_tsdir, str(camera_id)+'.png')
        _image = np.asarray(Image.open(image_path).resize(img_dim, resample=Image.BOX))

        height, width, _ = _image.shape
        print(height, width)
        camera = pyrender.IntrinsicsCamera(fx=focal_x*uf, fy=focal_y*uf, cx = (cx)*uf, cy = (cy)*uf, zfar=1e8*uf, znear = 0.01)
        # render the tracked mesh onto the camera
        mesh = pyrender.Mesh.from_trimesh(tracked_mesh)
        scene = pyrender.Scene(ambient_light = np.zeros(3), bg_color=[0.0, 0.0, 0.0])
        scene.add(mesh)
        scene.add(camera, pose=camera_pose)
        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=camera_pose)

        r = pyrender.OffscreenRenderer(viewport_width = img_dim[0], viewport_height = img_dim[1])
        rendered_img, depth = r.render(scene, flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.FLAT)

        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(rendered_img)
        plt.title("Rendered image")
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow(_image)
        plt.title("GT")
        plt.suptitle(f"Camera ID: {camera_id}, {expName} at {timestamp}")
        plt.savefig("test.png", bbox_inches='tight', pad_inches=0.0)
        r.delete()
    