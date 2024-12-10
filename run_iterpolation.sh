# Ours
iter=2200
IL_folder_name="" #replace with the name of trained folder
python identity_interpolating_styleGAN.py --path2blendshape=${IL_folder_name}/EXP-1-head/0/blendshape_{iter}.pth  --set_id=-3 --IL_name="test" --render_camInterpolation=True
