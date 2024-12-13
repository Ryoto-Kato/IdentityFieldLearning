
# Semi-Low-End Identity Field Learning towards Photorealitic 3D Parametric Face Models

<p align="center">
Ryoto Kato, Tobias Kirschstein, and Matthias Nießner 
</p>

<p align="center">
Technical University of Munich
</p>

<p align="center">
    <img src = https://github.com/user-attachments/assets/62a0d47b-f04e-4a64-a955-85ac01ec8e29 width="5000" alt="Results"/>
</p>

### [Full paper](https://arxiv.org) 

### Bibtex
```bib
@inproceedings{??,
    title = {Identity Field Learning towards Photorealistic 3D Parametric Face Models},
    author = {Ryoto Kato, Tobias Kirschstein, Matthias Nießner},
    booktitle = {???},
    year = {2024},
   }
```

## Background
3D face modelling has a significant gap between high-end and low-end automated approaches. While high-end approaches enable us to reconstruct a photorealistic 3D face, these methods are not often used in practice since they are still expensive regarding time/space complexity and are missing low-end properties such as cheap, fast, high compatibility and robust performance in a tracking/reconstruction task. On the other hand, a low-end approach, such as FLAME, is often used in practice due to its properties and is beneficial as a baseline in an application.  

## Abstruct
We present the identity field learning towards 3D Parametric Face Models, which is interleaving the significant gap in 3D face modelling, and it leads to practically beneficial above low-end properties with high-end properties, such as photorealistic appearance. Given the limited number of training samples (128 identities), our generative model can achieve high generality and expressiveness. The training takes **c.a. 9 hours on NVIDIA RTX 3080**. Our model architecture enables us to create 3D consistent photorealistic face representations without constraints in semi-real-time. In a test time, one can reconstruct a 3D photorealistic face given a single view in **30 seconds**. We hope our model can contribute to one's photorealistic face reconstruction task as a baseline to make the gap smaller in 3D face modelling. 

### Acknowledgement
This project is done in a Guided Research Module at TUM: [Visual Computing and AI Group](https://www.niessnerlab.org/index.html). I want to thank [MSc. Tobias Kirschstein](https://tobias-kirschstein.github.io/) and Prof. Dr Matthias Nießner, thank you for such a great opportunity and for providing resources and support. Additionally, I would like to thank all contributors/authors of related works: [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [StyleGAN2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch), [FLAME](https://flame.is.tue.mpg.de/), and [HeadNeRF](https://hy1995.top/HeadNeRF-Project/).

#### Reference: an adaption of 3DGS to non-zero principle point offsets: [issue #144](https://github.com/graphdeco-inria/gaussian-splatting/issues/144#issuecomment-1938504456), officially apply to the official 3DGS implementation

# Pipeline overview
![model-architecture](https://github.com/user-attachments/assets/c47a842a-ba63-4225-b824-f6970d6e7590)

# Requirements
- Linux x86-64 (test on Ubuntu 22.04.4)
- python 3.7.13 (conda)
- GPU 10GB VRAM (at least 8GB)
    - must fulfil requirements for 3D Gaussian splatting pipelines

# Setup
## set up conda env
```sh
conda create --name 3dgs+3dsrf+styleGAN+flame --file 3dgs+3dsrf+styleGAN+flame.yml
conda activate 3dgs+3dsrf+styleGAN+flame
```
- when you have a problem with creating conda-env with above .yml, you can create by step-by-step following respective official set-up instructions

```sh
conda create --name 3dgs+3dsrf+styleGAN+flame --file 3dsrf.yml
conda activate 3dgs+3dsrf+styleGAN+flame
"""
follow official repo instruction, 
1. 3dgs
2. StyleGAN
3. FLAME
"""
```


## Dataset setup and pre-processing

### FLAME
- Download the followings from the official FLAME website and save them into ./data
    - `flame2023.pkl`
    - `flame_dynamic_embedding.npy`
    - `flame_static_embedding.pkl`

- we want to obtain a FLAME Vertex Mask for a face-mask region
    - We have already provided vertex-mask `FLAME_mask_vertexID.npy` in ./data
    - This is obtained by using official FLAME Vertex Masks. Check out the FLAME official side

### NeRSemble
#### Download NeRsemble
- Follow the guide to download them from [official-repo](https://tobias-kirschstein.github.io/nersemble/).
- If you want to apply this method on your own dataset, which has been pre-processed with FLAME, refer (insta2nersemble.py) and apply the following preprocessing

#### Pre-process
- we want to pre-process the dataset to obtain
    - individual FLAME mesh (*.ply)
    - individual FLAME face-mask (*.ply)
    - individual FLAME face-mask back-projection mask on 16 cameras(*.png, binary)

- Steps
    - use codes under utils/preprocess
        1. Obtain FLAME mesh (*.ply)
        ```sh
            python FLAME2PLY.py
        ```
        2. Obtain FLAME face-mask (*.ply)
        ```sh
            # nersemble
            python flamemask_creater.py
            # insta
            python flamemask_creater.py --subject_id="flame1"
        ```
        3. Obtain projection of face-mask on 16 cameras (*.png, binary)
        ```sh
            # nersemble
            python projection_ply.py --facemask --blur
            # insta
            python projection_ply.py --subject_id="flame1" --facemask --blur --img_width=512 --img_height=512 --img_scale=1.0
        ```
        4. Obtain masked GT by using a mask from 3rd step (*.png, RGB)
        ```sh
            # nersemble
            python projection_ply.py --mask_images --mask_type="facemask" --bg_color= b
            # insta
            python projection_ply.py --mask_images --subject_id="flame1" --mask_type="facemask" --bg_color=b --img_width=512 --img_height=512 --img_scale=1.0
        ```

#### Final data structure
```sh
/subject_name/
├── calibration
│   ...
└── sequences
    └── EXP-1-head
        ├── annotations
        │   ├── color_correction
        │   │   ...
        │   └── tracking
        │       └── FLAME2023_v2
        │           ├── tracked_flame_params.npz
        │           └── video.mp4
        └── timesteps
            └── frame_00000
                ├── 00000_colored_excmask.ply
                ├── 00000_colored_facemask.ply
                ├── 00000_colored_noneck.ply
                ├── 00000_excmask.ply
                ├── 00000_facemask.ply
                ├── 00000_facemask_subd2.ply
                ├── 00000_facemask_subd3.ply
                ├── 00000_facemask_subd4.ply
                ├── 00000_facemask_subd.ply
                ├── 00000_noneck.ply
                ├── 00000.ply
                ├── 00000_subd2.ply
                ├── 00000_subd3.ply
                ├── 00000_subd.ply
                ├── alpha_map
                ├── colmap
                ├── facer_segmentation_masks
                ├── flamemask_proj #projected FLAME facemask
                │   ├── cam_220700191_b.png
                │   ...
                ├── filled_masked_images #facemasked GT
                │   ├── cam_220700191.png
                    ...
                ├── flame_cano2world.txt #transformation mat 
                ├── images-2x
                │   ├── cam_220700191.jpg
                │   ...

``` 

## Training with 128 identities
```sh
bash run_training.sh
```
- If you might face the problem with gradient-explosion w.r.t Gaussians's scale. For stable training after c.a. 2200 epochs, we would recommend you use Gaussian's scale regularization by using `cs_reg_weight=1e-3` from 2000 epochs
- 2200 epochs is the best #epochs so far, and longer training worsens the tracking performance due to over-fitting. 

## Tracking/Reconstruction given single image)
```sh
bash run_tracking.sh
```

## Identity-Interpolation 
```sh
bash run_interpolation.sh
```
