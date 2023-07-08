# #
# #                                     Dyads = 4
# #
# # 10 FPS
# for i in {0..10}
# do
#     python train_and_validate_10fps_dyadic_cnn3d.py \
# 	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_10fps \
# 	   4 \
# 	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_10fps/trn_videos_all.txt \
# 	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_10fps/val_videos_all.txt \
# 	   /mnt/twelvetb/vj/dyadic_nn/wnw_table_roi/resized_224_10fps/Dec26/run"$i"
# done

# # 20 FPS
# for i in {0..10}
# do
#     python train_and_validate_20fps_dyadic_cnn3d.py \
# 	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_20fps \
# 	   4 \
# 	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_20fps/trn_videos_all.txt \
# 	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_20fps/val_videos_all.txt \
# 	   /mnt/twelvetb/vj/dyadic_nn/wnw_table_roi/resized_224_20fps/Dec26/run"$i"
# done

# # 30 FPS
# for i in {0..10}
# do
#     python train_and_validate_30fps_dyadic_cnn3d.py \
# 	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps \
# 	   4 \
# 	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps/trn_videos_all.txt \
# 	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps/val_videos_all.txt \
# 	   /mnt/twelvetb/vj/dyadic_nn/wnw_table_roi/resized_224_30fps/Dec26/run"$i"
# done




#
#                                     Dyads = 3
#
# 10 FPS
for i in {0..0}
do
    python train_and_validate_10fps_dyadic_cnn3d.py \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_10fps \
	   3 \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_10fps/trn_videos_all.txt \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_10fps/val_videos_all.txt \
	   /mnt/twelvetb/vj/dyadic_nn/wnw_table_roi/resized_224_10fps/Feb19_3dyads/run"$i"
done

# 20 FPS
for i in {0..0}
do
    python train_and_validate_20fps_dyadic_cnn3d.py \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_20fps \
	   3 \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_20fps/trn_videos_all.txt \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_20fps/val_videos_all.txt \
	   /mnt/twelvetb/vj/dyadic_nn/wnw_table_roi/resized_224_20fps/Feb19_3dyads/run"$i"
done

# 30 FPS
for i in {0..0}
do
    python train_and_validate_30fps_dyadic_cnn3d.py \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps \
	   3 \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps/trn_videos_all.txt \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps/val_videos_all.txt \
	   /mnt/twelvetb/vj/dyadic_nn/wnw_table_roi/resized_224_30fps/Feb19_3dyads/run"$i"
done


#
#                                     Dyads = 2
#
# 10 FPS
for i in {0..0}
do
    python train_and_validate_10fps_dyadic_cnn3d.py \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_10fps \
	   2 \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_10fps/trn_videos_all.txt \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_10fps/val_videos_all.txt \
	   /mnt/twelvetb/vj/dyadic_nn/wnw_table_roi/resized_224_10fps/Feb19_2dyads/run"$i"
done

# 20 FPS
for i in {0..0}
do
    python train_and_validate_20fps_dyadic_cnn3d.py \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_20fps \
	   2 \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_20fps/trn_videos_all.txt \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_20fps/val_videos_all.txt \
	   /mnt/twelvetb/vj/dyadic_nn/wnw_table_roi/resized_224_20fps/Feb19_2dyads/run"$i"
done

# 30 FPS
for i in {0..0}
do
    python train_and_validate_30fps_dyadic_cnn3d.py \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps \
	   2 \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps/trn_videos_all.txt \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps/val_videos_all.txt \
	   /mnt/twelvetb/vj/dyadic_nn/wnw_table_roi/resized_224_30fps/Feb19_2dyads/run"$i"
done



#
#                                     Dyads = 1
#
# 10 FPS
for i in {0..0}
do
    python train_and_validate_10fps_dyadic_cnn3d.py \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_10fps \
	   1 \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_10fps/trn_videos_all.txt \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_10fps/val_videos_all.txt \
	   /mnt/twelvetb/vj/dyadic_nn/wnw_table_roi/resized_224_10fps/Feb19_1dyads/run"$i"
done

# 20 FPS
for i in {0..0}
do
    python train_and_validate_20fps_dyadic_cnn3d.py \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_20fps \
	   1 \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_20fps/trn_videos_all.txt \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_20fps/val_videos_all.txt \
	   /mnt/twelvetb/vj/dyadic_nn/wnw_table_roi/resized_224_20fps/Feb19_1dyads/run"$i"
done

# 30 FPS
for i in {0..0}
do
    python train_and_validate_30fps_dyadic_cnn3d.py \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps \
	   1 \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps/trn_videos_all.txt \
	   /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps/val_videos_all.txt \
	   /mnt/twelvetb/vj/dyadic_nn/wnw_table_roi/resized_224_30fps/Feb19_1dyads/run"$i"
done
