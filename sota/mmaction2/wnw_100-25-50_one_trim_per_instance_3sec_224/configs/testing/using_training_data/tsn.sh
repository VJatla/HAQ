cd /home/vj/mmaction2
echo "Using checkpoints from having best validation accuracy"
time python tools/test.py \
              configs/tynty_100-25-50_one_trim_per_instance_3sec_224/configs/testing/using_training_data/tsn_r50_video_1x1x8_6e_kinetics400_rgb.py \
       work_dirs/tynty_one_trim_per_instance_3sec_224/tsn/run1/epoch_49.pth \
       --eval top_k_accuracy
