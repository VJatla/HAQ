cd /home/vj/mmaction2

for i in 1
do
    CUDA_VISIBLE_DEVICES=0 python tools/train.py \
			configs/tynty_100-25-50_one_trim_per_instance_3sec_224/configs/i3d_r50_video_32x2x1_100e_kinetics400_rgb.py \
			--work-dir work_dirs/tynty_100-25-50_one_trim_per_instance_3sec_224/i3d/run$i \
			--validate \
			--seed 0 \
			--deterministic
    # slowfast
    CUDA_VISIBLE_DEVICES=0 python tools/train.py \
			configs/tynty_100-25-50_one_trim_per_instance_3sec_224/configs/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py \
			--work-dir work_dirs/tynty_100-25-50_one_trim_per_instance_3sec_224/slowfast/run$i \
			--validate \
			--seed 0 \
			--deterministic
# slowonly
    CUDA_VISIBLE_DEVICES=0 python tools/train.py \
			configs/tynty_100-25-50_one_trim_per_instance_3sec_224/configs/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py \
			--work-dir work_dirs/tynty_100-25-50_one_trim_per_instance_3sec_224/slowonly/run$i \
			--validate \
			--seed 0 \
			--deterministic
# tsm
    CUDA_VISIBLE_DEVICES=0 python tools/train.py \
			configs/tynty_100-25-50_one_trim_per_instance_3sec_224/configs/tsm_r50_video_1x1x8_50e_kinetics400_rgb.py \
			--work-dir work_dirs/tynty_100-25-50_one_trim_per_instance_3sec_224/tsm/run$i \
			--validate \
			--seed 0 \
			--deterministic
# tsn
    CUDA_VISIBLE_DEVICES=0 python tools/train.py \
			configs/tynty_100-25-50_one_trim_per_instance_3sec_224/configs/tsn_r50_video_1x1x8_6e_kinetics400_rgb.py \
			--work-dir work_dirs/tynty_100-25-50_one_trim_per_instance_3sec_224/tsn/run$i \
			--validate \
			--seed 0 \
			--deterministic
done
