for i in {0..21}
do
    python train_and_validate_dyadic_cnn3d.py \
	   /mnt/twotb/aolme_datasets/tynty_table_roi/resized_224 \
	   4 \
	   /mnt/twotb/aolme_datasets/tynty_table_roi/resized_224/trn_videos_all.txt \
	   /mnt/twotb/aolme_datasets/tynty_table_roi/resized_224/val_videos_all.txt \
	   /mnt/twelvetb/vj/dyadic_nn/tynty_table_roi/resized_224/run"$i"_Oct04_2022
done
