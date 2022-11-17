for i in {0..10}
do
    python train_and_validate_dyadic_cnn3d.py \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224/ \
	   4 \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224/trn_videos_all.txt \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224/val_videos_all.txt \
	   /mnt/twelvetb/vj/dyadic_nn/wnw_table_roi/resized_224/Nov10/run"$i"_Nov10_2022
done
