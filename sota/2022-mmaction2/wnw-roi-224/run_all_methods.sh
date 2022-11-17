# Activate mmaction2 library
source ~/Software/mma2_venv/bin/activate

# go to mmaction2 directory
cd ~/Software/mmaction2/

# i3d
CUDA_VISIBLE_DEVICES=1 python tools/train.py /home/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/i3d/trn_i3d_r50_video_32x2x1_100e_kinetics400_rgb_vj.py --validate --deterministic

#slowfast
CUDA_VISIBLE_DEVICES=1 python tools/train.py /home/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/slowfast/trn_slowfast_r50_video_4x16x1_256e_kinetics400_rgb_vj.py --validate --deterministic

# TSM
CUDA_VISIBLE_DEVICES=1 python tools/train.py /home/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/tsm/trn_tsm_r50_video_1x1x8_50e_kinetics400_rgb_vj.py --validate --deterministic

# TSN
CUDA_VISIBLE_DEVICES=1 python tools/train.py /home/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/tsn/trn_tsn_r50_video_1x1x8_100e_kinetics400_rgb_vj.py --validate --deterministic
