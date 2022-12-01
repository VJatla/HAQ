for i in {0..0}
do
    python train_and_validate_dyadic_cnn3d.py \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_20fps \
	   4 \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_20fps/trn_videos_all.txt \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_20fps/val_videos_all.txt \
	   /mnt/twelvetb/vj/dyadic_nn/wnw_table_roi/resized_224_20fps/dev/run"$i"
done
