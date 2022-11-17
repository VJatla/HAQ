for i in {0..21}
do
    CUDA_VISIBLE_DEVICES=1 python train_and_validate_dyadic_cnn3d.py \
	   /mnt/twotb/aolme_datasets/wnw_gt/resized_224/ \
	   4 \
	   /mnt/twotb/aolme_datasets/wnw_gt/resized_224/trn_videos_all.txt \
	   /mnt/twotb/aolme_datasets/wnw_gt/resized_224/val_videos_all.txt \
	   /mnt/twelvetb/vj/dyadic_nn/wnw_gt/resized_224/run"$i"_Sep05_2022
done
