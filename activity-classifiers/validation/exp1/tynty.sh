# 50 samples
CUDA_VISIBLE_DEVICES=0 python tynty.py 1 \
       /mnt/twotb/dyadic_nn/workdir/tynty/one_trim_per_instance_3sec_224/trn_videos_142per_act/dyad_1/run0 \
       /mnt/twotb/aolme_datasets/tynty/trimmed_videos/one_trim_per_instance_3sec_224/val_videos_all.txt \
       val_videos_all_log.json
CUDA_VISIBLE_DEVICES=0 python tynty.py 2 \
       /mnt/twotb/dyadic_nn/workdir/tynty/one_trim_per_instance_3sec_224/trn_videos_142per_act/dyad_2/run0 \
       /mnt/twotb/aolme_datasets/tynty/trimmed_videos/one_trim_per_instance_3sec_224/val_videos_all.txt \
       val_videos_all_log.json
CUDA_VISIBLE_DEVICES=0 python tynty.py 3 \
       /mnt/twotb/dyadic_nn/workdir/tynty/one_trim_per_instance_3sec_224/trn_videos_142per_act/dyad_3/run0 \
       /mnt/twotb/aolme_datasets/tynty/trimmed_videos/one_trim_per_instance_3sec_224/val_videos_all.txt \
       val_videos_all_log.json
CUDA_VISIBLE_DEVICES=0 python tynty.py 4 \
       /mnt/twotb/dyadic_nn/workdir/tynty/one_trim_per_instance_3sec_224/trn_videos_142per_act/dyad_4/run0 \
       /mnt/twotb/aolme_datasets/tynty/trimmed_videos/one_trim_per_instance_3sec_224/val_videos_all.txt \
       val_videos_all_log.json
