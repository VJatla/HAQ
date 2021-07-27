cd /home/vj/mmaction2
echo "Using checkpoints from having best validation accuracy"
time python tools/test.py \
       configs/wnw_100-25-50_one_trim_per_instance_3sec_224/configs/testing/using_testing_data/i3d_r50_video_32x2x1_100e_kinetics400_rgb.py \
       work_dirs/wnw_one_trim_per_instance_3sec_224/i3d/run1/epoch_39.pth \
       --eval top_k_accuracy
