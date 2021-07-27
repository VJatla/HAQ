cd /home/vj/mmaction2
for i in 1
do
    # i3d
    CUDA_VISIBLE_DEVICES=0 python tools/train.py \
			configs/tynty_group_loocv_one_trim_per_instance_3sec_224/configs/C1L1P-B/i3d_r50_video_32x2x1_100e_kinetics400_rgb.py \
			--work-dir work_dirs/tynty_group_loocv_one_trim_per_instance_3sec_224/C1L1P-B/i3d/run$i \
			--validate \
			--seed 0 \
			--deterministic


    # slowfast
    CUDA_VISIBLE_DEVICES=0 python tools/train.py \
			configs/tynty_group_loocv_one_trim_per_instance_3sec_224/configs/C1L1P-B/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py \
			--work-dir work_dirs/tynty_group_loocv_one_trim_per_instance_3sec_224/C1L1P-B/slowfast/run$i \
			--validate \
			--seed 0 \
			--deterministic
    # slowonly
    CUDA_VISIBLE_DEVICES=0 python tools/train.py \
			configs/tynty_group_loocv_one_trim_per_instance_3sec_224/configs/C1L1P-B/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py \
			--work-dir work_dirs/tynty_group_loocv_one_trim_per_instance_3sec_224/C1L1P-B/slowonly/run$i \
			--validate \
			--seed 0 \
			--deterministic
    # tsm
    CUDA_VISIBLE_DEVICES=0 python tools/train.py \
			configs/tynty_group_loocv_one_trim_per_instance_3sec_224/configs/C1L1P-B/tsm_r50_video_1x1x8_50e_kinetics400_rgb.py \
			--work-dir work_dirs/tynty_group_loocv_one_trim_per_instance_3sec_224/C1L1P-B/tsm/run$i \
			--validate \
			--seed 0 \
			--deterministic
    # tsn
    CUDA_VISIBLE_DEVICES=0 python tools/train.py \
			configs/tynty_group_loocv_one_trim_per_instance_3sec_224/configs/C1L1P-B/tsn_r50_video_1x1x8_6e_kinetics400_rgb.py \
			--work-dir work_dirs/tynty_group_loocv_one_trim_per_instance_3sec_224/C1L1P-B/tsn/run$i \
			--validate \
			--seed 0 \
			--deterministic
done
