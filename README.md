
# Semi Low-End Identity Field Learning towards Photorealitic 3D Parametric Face Models

<p align="center">
Ryoto Kato, Tobias Kirschstein, and Matthias Nießner 
</p>

<p align="center">
Technical University of Munich
</p>

### [Full paper](https://arxiv.org) 



## Bibtex
```
@inproceedings{??,
    title = {Identity Field Learning towards Photorealitic 3D Parametric Face Models},
    author = {Ryoto Kato, Tobias Kirschstein, Matthias Nießner},
    booktitle = {???},
    year = {2024},
   }
```

## Background
The field of 3D face modeling has a significant gap between high-end and low-end automated approaches. While high-end approaches enables us to reconstruct a photorealitic 3D face, these methods are not often used in practice since it still costs expensive in terms of time/space complexity and is missing low-end property such as cheap, fast, high compatibility, and robust performance in a tracking/reconstruction task. On the other hand, a low-end approach, such as FLAME, is often used in practice due to thoese properties and benefitial to be used as a baseline in an application.  

## Abstruct
We present the identity field learning towards 3D Parametric Face Models, which is interleaving the significant gap in 3D face modelling, and it leads to practically beneficial above low-end properties with high-end properties, such as photorealistic appearance. Our generative model can achieve both high generality and experssiveness given the limited number of training samples (128 identities). The training takes c.a. 9 hours on NVIDIA RTX 3080. Our model architecture enables us to create 3D consistent photorealistic face representations without constraints in semi real-time. In a test time, one can reconstruct a 3D photorealistic face given a single view in **30 seconds**. We hope our model can contribute to one's photorealistic face reconstruction task as a baseline to make the gap smaller in 3D face modelling. 

### Acknowledgement
This project is done in a Guided Research Module at TUM: [Visual Computing and AI Group](https://www.niessnerlab.org/index.html). I would like to thank [MSc. Tobias Kirschstein](https://tobias-kirschstein.github.io/) and Prof. Dr Matthias Nießner for such a great opportunity and for providing resources and support.

# Pipeline overview


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
follow officitial repo instruction, 
1. 3dgs
2. styleGAN
3. FLAME
"""
```


## Dataset setup and pre-processing

### FLAME
- Download followings from official FLAME website and save into ./data
    - `flame2023.pkl`
    - `flame_dynamic_embedding.npy`
    - `flame_static_embedding.pkl`

- we want to obtain FLAME Vertex Mask for a face-mask region
    - We have already provided vertex-mask `FLAME_mask_vertexID.npy` in ./data
    - This is obtained by using official FLAME Vertex Masks, check out the FLAME official side

### NeRSemble
#### Download NeRsemble
- Follow the guide to download them from [official-repo](https://tobias-kirschstein.github.io/nersemble/).
- If you want to apply this methods on your own dataset, which are pre-processed with FLAME, refer (insta2nersemble.py) and apply the following preprocessings

#### Pre-process
- we want to pre-process dataset to obtain
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
- If you might be facing the problem with graident-explosion w.r.t Gaussians's scale. For stable training with over 2000 epochs, we would recomment you to use Gaussian's scale regularization by using `cs_reg_weight=1e-3` from 2000 epochs
- 2200 epochs is the best #epochs so far and longer training makes the tracking performance worse due to the over-fitting. 

## Tracking/Reconstruction given single image)
```sh
bash run_tracking.sh
```

## Identity-Interpolation 
```sh
bash run_interpolation.sh
```
