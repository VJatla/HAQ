cd /home/vj/mmaction2
echo "Using checkpoints from having best validation accuracy"
time python tools/test.py \
              configs/tynty_100-25-50_one_trim_per_instance_3sec_224/configs/testing/using_training_data/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py \
       work_dirs/tynty_one_trim_per_instance_3sec_224/slowonly/run1/epoch_149.pth \
       --eval top_k_accuracy
