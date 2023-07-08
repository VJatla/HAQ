# NOTE:
#    (1) Run the following comands from ~/Software/mmaction2/ after activating mma2_venv using
#     source ~/Software/mma2_venv/bin/activate
#    (2) To use cuda device 1 use CUDA_VISIBLE_DEVICES=1 before "python ****" command


#
#          C1L1P-E, 20170302  (Worked)
#
# # I3D
# time python tools/test_vj.py \
# 		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/i3d/testing_C1L1P-E_20170302_i3d_r50_video_32x2x1_100e_kinetics400_rgb_vj.py \
# 		    /mnt/twelvetb/vj/mmaction2_2022/workdir/wnw_table_roi/resized_224_30fps/i3d/run0_Dec26_2022/best_top1_acc_epoch_40.pth \
#                     --out ~/Dropbox/writing-nowriting/C1L1P-E/20170302/wnw-roi-sota-i3d.pkl
# # TSM 
# time python tools/test_vj.py \
# 		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/tsm/testing_C1L1P-E_20170302_tsm_r50_video_1x1x8_50e_kinetics400_rgb_vj.py \
# 		    /mnt/twelvetb/vj/mmaction2_2022/workdir/wnw_table_roi/resized_224_30fps/tsm/run0_Jan03_2023/best_top1_acc_epoch_15.pth \
#                     --out ~/Dropbox/writing-nowriting/C1L1P-E/20170302/wnw-roi-sota-tsm.pkl
# # TSN
# time python tools/test_vj.py \
# 		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/tsn/testing_C1L1P-E_20170302_tsn_r50_video_1x1x8_100e_kinetics400_rgb_vj.py \
# 		    /mnt/twelvetb/vj/mmaction2_2022/workdir/wnw_table_roi/resized_224_30fps/tsn/run0_Jan03_2023/best_top1_acc_epoch_95.pth \
#                     --out ~/Dropbox/writing-nowriting/C1L1P-E/20170302/wnw-roi-sota-tsn.pkl
# # Slowfast
# time python tools/test_vj.py \
# 		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/slowfast/testing_C1L1P-E_20170302_slowfast_r50_video_4x16x1_256e_kinetics400_rgb_vj.py \
# 		    /mnt/twelvetb/vj/mmaction2_2022/workdir/wnw_table_roi/resized_224_30fps/slowfast/run0_Jan03_2023/best_top1_acc_epoch_180.pth \
#                     --out ~/Dropbox/writing-nowriting/C1L1P-E/20170302/wnw-roi-sota-slowfast.pkl



#
#          C2L1P-B, 20180223  (Worked after removing one instance)
#
# cd ~/Software/mmaction2/
# # I3D
# time python tools/test_vj.py \
# 		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/i3d/testing_C2L1P-B_20180223_i3d_r50_video_32x2x1_100e_kinetics400_rgb_vj.py \
# 		    /mnt/twelvetb/vj/mmaction2_2022/workdir/wnw_table_roi/resized_224_30fps/i3d/run0_Dec26_2022/best_top1_acc_epoch_40.pth \
#                     --out ~/Dropbox/writing-nowriting/C2L1P-B/20180223/wnw-roi-sota-i3d.pkl
# # TSM 
# time python tools/test_vj.py \
# 		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/tsm/testing_C2L1P-B_20180223_tsm_r50_video_1x1x8_50e_kinetics400_rgb_vj.py \
# 		    /mnt/twelvetb/vj/mmaction2_2022/workdir/wnw_table_roi/resized_224_30fps/tsm/run0_Jan03_2023/best_top1_acc_epoch_15.pth \
#                     --out ~/Dropbox/writing-nowriting/C2L1P-B/20180223/wnw-roi-sota-tsm.pkl
# # TSN
# time python tools/test_vj.py \
# 		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/tsn/testing_C2L1P-B_20180223_tsn_r50_video_1x1x8_100e_kinetics400_rgb_vj.py \
# 		    /mnt/twelvetb/vj/mmaction2_2022/workdir/wnw_table_roi/resized_224_30fps/tsn/run0_Jan03_2023/best_top1_acc_epoch_95.pth \
#                     --out ~/Dropbox/writing-nowriting/C2L1P-B/20180223/wnw-roi-sota-tsn.pkl
# # Slowfast
# time python tools/test_vj.py \
# 		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/slowfast/testing_C2L1P-B_20180223_slowfast_r50_video_4x16x1_256e_kinetics400_rgb_vj.py \
# 		    /mnt/twelvetb/vj/mmaction2_2022/workdir/wnw_table_roi/resized_224_30fps/slowfast/run0_Jan03_2023/best_top1_acc_epoch_180.pth \
#                     --out ~/Dropbox/writing-nowriting/C2L1P-B/20180223/wnw-roi-sota-slowfast.pkl



#
#          C3L1P-C, 20180411
#
cd ~/Software/mmaction2/
# I3D
time python tools/test_vj.py \
		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/i3d/testing_C3L1P-C_20190411_i3d_r50_video_32x2x1_100e_kinetics400_rgb_vj.py \
		    /mnt/twelvetb/vj/mmaction2_2022/workdir/wnw_table_roi/resized_224_30fps/i3d/run0_Dec26_2022/best_top1_acc_epoch_40.pth \
                    --out ~/Dropbox/writing-nowriting/C3L1P-C/20190411/wnw-roi-sota-i3d.pkl
# TSM 
time python tools/test_vj.py \
		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/tsm/testing_C3L1P-C_20190411_tsm_r50_video_1x1x8_50e_kinetics400_rgb_vj.py \
		    /mnt/twelvetb/vj/mmaction2_2022/workdir/wnw_table_roi/resized_224_30fps/tsm/run0_Jan03_2023/best_top1_acc_epoch_15.pth \
                    --out ~/Dropbox/writing-nowriting/C3L1P-C/20190411/wnw-roi-sota-tsm.pkl
# TSN
time python tools/test_vj.py \
		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/tsn/testing_C3L1P-C_20190411_tsn_r50_video_1x1x8_100e_kinetics400_rgb_vj.py \
		    /mnt/twelvetb/vj/mmaction2_2022/workdir/wnw_table_roi/resized_224_30fps/tsn/run0_Jan03_2023/best_top1_acc_epoch_95.pth \
                    --out ~/Dropbox/writing-nowriting/C3L1P-C/20190411/wnw-roi-sota-tsn.pkl
# Slowfast
time python tools/test_vj.py \
		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/slowfast/testing_C3L1P-C_20190411_slowfast_r50_video_4x16x1_256e_kinetics400_rgb_vj.py \
		    /mnt/twelvetb/vj/mmaction2_2022/workdir/wnw_table_roi/resized_224_30fps/slowfast/run0_Jan03_2023/best_top1_acc_epoch_180.pth \
                    --out ~/Dropbox/writing-nowriting/C3L1P-C/20190411/wnw-roi-sota-slowfast.pkl
