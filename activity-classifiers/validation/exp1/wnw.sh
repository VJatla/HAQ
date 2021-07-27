# 50 samples
CUDA_VISIBLE_DEVICES=1 python wnw.py 1 \
       /mnt/twotb/dyadic_nn/workdir/wnw/one_trim_per_instance_3sec_224/trn_videos_150per_act/dyad_1/run0 \
       /mnt/twotb/aolme_datasets/wnw/trimmed_videos/one_trim_per_instance_3sec_224/val_videos_all.txt \
       val_videos_all_log.json
CUDA_VISIBLE_DEVICES=1 python wnw.py 2 \
       /mnt/twotb/dyadic_nn/workdir/wnw/one_trim_per_instance_3sec_224/trn_videos_150per_act/dyad_2/run0 \
       /mnt/twotb/aolme_datasets/wnw/trimmed_videos/one_trim_per_instance_3sec_224/val_videos_all.txt \
       val_videos_all_log.json
CUDA_VISIBLE_DEVICES=1 python wnw.py 3 \
       /mnt/twotb/dyadic_nn/workdir/wnw/one_trim_per_instance_3sec_224/trn_videos_150per_act/dyad_3/run0 \
       /mnt/twotb/aolme_datasets/wnw/trimmed_videos/one_trim_per_instance_3sec_224/val_videos_all.txt \
       val_videos_all_log.json
CUDA_VISIBLE_DEVICES=1 python wnw.py 4 \
       /mnt/twotb/dyadic_nn/workdir/wnw/one_trim_per_instance_3sec_224/trn_videos_150per_act/dyad_4/run0 \
       /mnt/twotb/aolme_datasets/wnw/trimmed_videos/one_trim_per_instance_3sec_224/val_videos_all.txt \
       val_videos_all_log.json
