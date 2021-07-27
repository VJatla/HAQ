"""
Exp2: Group level leave one out for typing/notyping
"""
import os
import pdb

dyads = [2, 3, 4]
groups = ['C1L1P-A', 'C1L1P-B', 'C1L1P-C', 'C1L1W-A', 'C2L1W-B', 'C3L1W-D']
vdir = "/mnt/twotb/aolme_datasets/tynty/trimmed_videos/one_trim_per_instance_3sec_224"
trn_list_dir = "/mnt/twotb/aolme_datasets/tynty/trimmed_videos/one_trim_per_instance_3sec_224/group_leave_one_out/exp2"
workdir_root = "/mnt/twotb/dyadic_nn/workdir/tynty/one_trim_per_instance_3sec_224/exp2_group_leave_one_out"
cuda_device = 0

cmd_head = f"CUDA_VISIBLE_DEVICES={cuda_device} python train_dyadic_cnn3d.py {vdir}"

for grp in groups:
    
    # Text file having training list
    trn_list_file = f"{trn_list_dir}/{grp}/trn.txt"
    
    for dyad in dyads:

        # Working directory
        wdir = f"{workdir_root}/{grp}/dyad_{dyad}"

        # Command to execute
        cmd = f"{cmd_head} {dyad} {trn_list_file} {wdir}"

        # Execute
        os.system(cmd)
        
        
    



