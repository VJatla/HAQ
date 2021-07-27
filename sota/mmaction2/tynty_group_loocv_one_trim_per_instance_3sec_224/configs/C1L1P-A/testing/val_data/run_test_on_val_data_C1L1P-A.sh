cd /home/vj/mmaction2
for i in 1
do
    # i3d
    echo "-------------------I3D ----------------"
    time python tools/test.py \
	 configs/tynty_group_loocv_one_trim_per_instance_3sec_224/configs/C1L1P-A/testing/val_data/i3d_r50_video_32x2x1_100e_kinetics400_rgb.py \
	 work_dirs/tynty_group_loocv_one_trim_per_instance_3sec_224/C1L1P-A/i3d/run1/epoch_5.pth \
	 --eval top_k_accuracy

    echo "-------------------Slow fast ----------------"
        time python tools/test.py \
	     configs/tynty_group_loocv_one_trim_per_instance_3sec_224/configs/C1L1P-A/testing/val_data/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py \
	     work_dirs/tynty_group_loocv_one_trim_per_instance_3sec_224/C1L1P-A/slowfast/run1/epoch_5.pth \
	     --eval top_k_accuracy

         echo "-------------------Slow only ----------------"
        time python tools/test.py \
	     configs/tynty_group_loocv_one_trim_per_instance_3sec_224/configs/C1L1P-A/testing/val_data/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py \
	     work_dirs/tynty_group_loocv_one_trim_per_instance_3sec_224/C1L1P-A/slowonly/run1/epoch_11.pth \
	     --eval top_k_accuracy


	    echo "-------------------TSN ----------------"
        time python tools/test.py \
	     configs/tynty_group_loocv_one_trim_per_instance_3sec_224/configs/C1L1P-A/testing/val_data/tsn_r50_video_1x1x8_6e_kinetics400_rgb.py \
	     work_dirs/tynty_group_loocv_one_trim_per_instance_3sec_224/C1L1P-A/tsn/run1/epoch_12.pth \
	     --eval top_k_accuracy

done
