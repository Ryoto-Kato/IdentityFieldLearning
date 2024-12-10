
# Ours
iter=2200
IL_folder_name="128subjects_styleGAN_noisemode_const_common_plane_K12tex64_anchor_subd0_xyzmethod_default_learnableMEAN1430_1109" #replace with the name of trained folder
IL_name="After_GR_RELEASE_iter${iter}" #replace this by a text which will be put at the beginning of the saving folder
python identity_tracking_styleGAN.py --IL_name=${IL_name} --set_id=0 --path2blendshape=/mnt/hdd/GRFinal/${IL_folder_name}/EXP-1-head/0/blendshape_${iter}.pth
python identity_tracking_styleGAN.py --IL_name=${IL_name} --set_id=1 --path2blendshape=/mnt/hdd/GRFinal/${IL_folder_name}/EXP-1-head/0/blendshape_${iter}.pth
python identity_tracking_styleGAN.py --IL_name=${IL_name} --set_id=2 --path2blendshape=/mnt/hdd/GRFinal/${IL_folder_name}/EXP-1-head/0/blendshape_${iter}.pth
