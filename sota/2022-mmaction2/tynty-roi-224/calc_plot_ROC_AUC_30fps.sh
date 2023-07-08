# Activate mmaction2 library
source ~/Software/mma2_venv/bin/activate

# I3D
# time python calc_plot_ROC_AUC_30fps.py \
#     ~/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/i3d/trn_i3d_r50_video_32x2x1_100e_kinetics400_rgb_vj.py \
#     /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/i3d/run1_Dec27_2022/best_top1_acc_epoch_35.pth \
#     /mnt/twotb/aolme_datasets/tynty_table_roi/resized_224_30fps/tst_videos_all.txt

# Slowfast
# time python calc_plot_ROC_AUC_30fps.py \
#     ~/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/slowfast/trn_slowfast_r50_video_4x16x1_256e_kinetics400_rgb_vj.py \
#     /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/slowfast/run0_Jan03/best_top1_acc_epoch_85.pth \
#     /mnt/twotb/aolme_datasets/tynty_table_roi/resized_224_30fps/tst_videos_all.txt

# TSM
time python calc_plot_ROC_AUC_30fps.py \
    ~/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/tsm/trn_tsm_r50_video_1x1x8_50e_kinetics400_rgb_vj.py \
    /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/tsm/run0_Jan03/best_top1_acc_epoch_30.pth \
    /mnt/twotb/aolme_datasets/tynty_table_roi/resized_224_30fps/tst_videos_all.txt
time python calc_plot_ROC_AUC_30fps.py \
    ~/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/tsm/trn_tsm_r50_video_1x1x8_50e_kinetics400_rgb_vj.py \
    /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/tsm/run0_Jan03/best_top1_acc_epoch_30.pth \
    /mnt/twotb/aolme_datasets/tynty_table_roi/resized_224_30fps/tst_videos_all.txt
# TSN
time python calc_plot_ROC_AUC_30fps.py \
    ~/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/tynty-roi-224/tsn/trn_tsn_r50_video_1x1x8_100e_kinetics400_rgb_vj.py \
    /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224_30fps/tsn/run0_Jan03/best_top1_acc_epoch_35.pth \
    /mnt/twotb/aolme_datasets/tynty_table_roi/resized_224_30fps/tst_videos_all.txt



