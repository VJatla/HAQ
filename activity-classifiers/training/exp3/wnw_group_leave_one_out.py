"""
Exp2: Group level leave one out for writing/nowriting
"""
import os
import pdb

dyads = [2, 3, 4]
groups = ['C1L1P-B', 'C1L1P-C', 'C2L1P-B', 'C2L1P-C']
vdir = "/mnt/twotb/aolme_datasets/wnw/trimmed_videos/one_trim_per_instance_3sec_224"
trn_list_dir = "/mnt/twotb/aolme_datasets/wnw/trimmed_videos/one_trim_per_instance_3sec_224/group_leave_one_out/exp2"
workdir_root = "/mnt/twotb/dyadic_nn/workdir/wnw/one_trim_per_instance_3sec_224/exp2_group_leave_one_out"
cuda_device = 1

cmd_head = f"CUDA_VISIBLE_DEVICES={cuda_device} python train_dyadic_cnn3d.py {vdir}"

for grp in groups:

    # Training list file
    trn_list_file = f"{trn_list_dir}/{grp}/trn.txt"
    
    for dyad in dyads:

        # Working directory
        wdir = f"{workdir_root}/{grp}/dyad_{dyad}"

        # Command to execute
        cmd = f"{cmd_head} {dyad} {trn_list_file} {wdir}"
        pdb.set_trace()

        # Execute
        os.system(cmd)
