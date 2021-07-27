cd /home/vj/mmaction2
for i in 1
do
    # i3d
    echo "-------------------I3D ----------------"
    time python tools/test.py \
	 configs/tynty_group_loocv_one_trim_per_instance_3sec_224/configs/C3L1W-D/testing/val_data/i3d_r50_video_32x2x1_100e_kinetics400_rgb.py \
	 work_dirs/tynty_group_loocv_one_trim_per_instance_3sec_224/C3L1W-D/i3d/run1/epoch_6.pth \
	 --eval top_k_accuracy

    echo "-------------------Slow fast ----------------"
        time python tools/test.py \
	     configs/tynty_group_loocv_one_trim_per_instance_3sec_224/configs/C3L1W-D/testing/val_data/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py \
	     work_dirs/tynty_group_loocv_one_trim_per_instance_3sec_224/C3L1W-D/slowfast/run1/epoch_8.pth \
	     --eval top_k_accuracy

    echo "-------------------Slow only ----------------"
        time python tools/test.py \
       configs/tynty_group_loocv_one_trim_per_instance_3sec_224/configs/C3L1W-D/testing/val_data/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py \
       work_dirs/tynty_group_loocv_one_trim_per_instance_3sec_224/C3L1W-D/slowonly/run1/epoch_55.pth \
       --eval top_k_accuracy

	    echo "-------------------TSN ----------------"
        time python tools/test.py \
	     configs/tynty_group_loocv_one_trim_per_instance_3sec_224/configs/C3L1W-D/testing/val_data/tsn_r50_video_1x1x8_6e_kinetics400_rgb.py \
	     work_dirs/tynty_group_loocv_one_trim_per_instance_3sec_224/C3L1W-D/tsn/run1/epoch_7.pth \
	     --eval top_k_accuracy

done
