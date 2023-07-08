# NOTE:
#     Run the following comands from ~/Software/mmaction2/ after activating mma2_venv using
#     source ~/Software/mma2_venv/bin/activate 



#
#          C3L1P-D, 20190221 (WORKED)
#

# I3D
time CUDA_VISIBLE_DEVICES=1 python tools/test_vj.py \
		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/i3d/testing_C3L1P-D_20190221_i3d_r50_video_32x2x1_100e_kinetics400_rgb_vj.py \
		    /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/i3d/run1_Dec27_2022/best_top1_acc_epoch_35.pth \
                    --out ~/Dropbox/typing-notyping/C3L1P-D/20190221/tynty-roi-sota-i3d.pkl

# TSM
time CUDA_VISIBLE_DEVICES=1 python tools/test_vj.py \
		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/tsm/testing_C3L1P-D_20190221_tsm_r50_video_1x1x8_50e_kinetics400_rgb_vj.py \
		    /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/tsm/run0_Jan03/best_top1_acc_epoch_30.pth \
                    --out ~/Dropbox/typing-notyping/C3L1P-D/20190221/tynty-roi-sota-tsm.pkl
# TSN
time CUDA_VISIBLE_DEVICES=1 python tools/test_vj.py \
		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/tsn/testing_C3L1P-D_20190221_tsn_r50_video_1x1x8_100e_kinetics400_rgb_vj.py \
		    /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/tsn/run0_Jan03/best_top1_acc_epoch_35.pth \
                    --out ~/Dropbox/typing-notyping/C3L1P-D/20190221/tynty-roi-sota-tsn.pkl

# Slowfast
time CUDA_VISIBLE_DEVICES=1 python tools/test_vj.py \
		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/slowfast/testing_C3L1P-D_20190221_slowfast_r50_video_4x16x1_256e_kinetics400_rgb_vj.py \
		    /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/slowfast/run0_Jan03/best_top1_acc_epoch_85.pth \
                    --out ~/Dropbox/typing-notyping/C3L1P-D/20190221/tynty-roi-sota-slowfast.pkl


#
#          C1L1P-E, 20170302 
#

#  i3d
time python tools/test_vj.py \
		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/i3d/testing_C1L1P-E_20170302_i3d_r50_video_32x2x1_100e_kinetics400_rgb_vj.py \
		    /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/i3d/run1_Dec27_2022/best_top1_acc_epoch_35.pth \
                    --out ~/Dropbox/typing-notyping/C1L1P-E/20170302/tynty-roi-sota-i3d.pkl

#  tsm
time  python tools/test_vj.py \
		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/tsm/testing_C1L1P-E_20170302_tsm_r50_video_1x1x8_50e_kinetics400_rgb_vj.py \
		    /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/tsm/run0_Jan03/best_top1_acc_epoch_30.pth \
                    --out ~/Dropbox/typing-notyping/C1L1P-E/20170302/tynty-roi-sota-tsm.pkl

#  tsn
time  python tools/test_vj.py \
		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/tsn/testing_C1L1P-E_20170302_tsn_r50_video_1x1x8_100e_kinetics400_rgb_vj.py \
		    /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/tsn/run0_Jan03/best_top1_acc_epoch_35.pth \
                    --out ~/Dropbox/typing-notyping/C1L1P-E/20170302/tynty-roi-sota-tsn.pkl

#  slowfast
time  python tools/test_vj.py \
		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/slowfast/testing_C1L1P-E_20160302_slowfast_r50_video_4x16x1_256e_kinetics400_rgb_vj.py \
		    /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/slowfast/run0_Jan03/best_top1_acc_epoch_85.pth \
                    --out ~/Dropbox/typing-notyping/C1L1P-E/20170302/tynty-roi-sota-slowfast.pkl





#
#          C2L1P-B, 20180223 (DID NOT WORK)
#

# C2L1P-B, 20180223 i3d
time CUDA_VISIBLE_DEVICES=1 python tools/test_vj.py \
		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/i3d/testing_C2L1P-B_20180223_i3d_r50_video_32x2x1_100e_kinetics400_rgb_vj.py \
		    /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/i3d/run1_Dec27_2022/best_top1_acc_epoch_35.pth \
                    --out ~/Dropbox/typing-notyping/C2L1P-B/20190221/tynty-roi-sota-i3d.pkl

# C2L1P-B, 20180223 tsm
time CUDA_VISIBLE_DEVICES=1 python tools/test_vj.py \
		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/tsm/testing_C2L1P-B_20180223_tsm_r50_video_1x1x8_50e_kinetics400_rgb_vj.py \
		    /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/tsm/run0_Jan03/best_top1_acc_epoch_30.pth \
                    --out ~/Dropbox/typing-notyping/C2L1P-B/20190221/tynty-roi-sota-tsm.pkl

# C2L1P-B, 20180223 tsn
time CUDA_VISIBLE_DEVICES=1 python tools/test_vj.py \
		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/tsn/testing_C2L1P-B_20180223_tsn_r50_video_1x1x8_100e_kinetics400_rgb_vj.py \
		    /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/tsn/run0_Jan03/best_top1_acc_epoch_35.pth \
                    --out ~/Dropbox/typing-notyping/C2L1P-B/20190221/tynty-roi-sota-tsn.pkl

# C2L1P-B, 20180223 slowfast
time CUDA_VISIBLE_DEVICES=1 python tools/test_vj.py \
		    /mnt/twelvetb/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/slowfast/testing_C2L1P-B_20180223_slowfast_r50_video_4x16x1_256e_kinetics400_rgb_vj.py \
		    /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/slowfast/run0_Jan03/best_top1_acc_epoch_85.pth \
                    --out ~/Dropbox/typing-notyping/C2L1P-B/20190221/tynty-roi-sota-slowfast.pkl
