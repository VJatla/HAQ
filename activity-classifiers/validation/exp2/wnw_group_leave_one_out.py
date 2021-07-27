"""
Exp2: Group level leave one out for typing/notyping
    args_inst.add_argument("vdir", type=str, help=("Directory to find trimmed videos"))
    args_inst.add_argument("ndyads", type=int, help=("Number of dyads"))
    args_inst.add_argument("workdir",
                           type=str,
                           help=("Training directory having checkpoints"))
    args_inst.add_argument("vlist",
                           type=str,
                           help=("Text file having validation list"))
    args_inst.add_argument(
        "log_name",
        type=str,
        help=("Name of validation log file. It is saved in working directory"))
"""
import os
import pdb

dyads = [2, 3, 4]
groups = ['C1L1P-B']
vdir = "/mnt/twotb/aolme_datasets/wnw/trimmed_videos/one_trim_per_instance_3sec_224"
list_dir = "/mnt/twotb/aolme_datasets/wnw/trimmed_videos/one_trim_per_instance_3sec_224/group_leave_one_out/exp2"
workdir_root = "/mnt/twotb/dyadic_nn/workdir/wnw/one_trim_per_instance_3sec_224/exp2_group_leave_one_out"
cuda_device = 0

cmd_head = f"CUDA_VISIBLE_DEVICES={cuda_device} python validate_dyadic_cnn3d.py {vdir}"

for grp in groups:
    
    # Text file having training list
    val_list_file = f"{list_dir}/{grp}/val.txt"
    
    for dyad in dyads:

        # Working directory
        wdir = f"{workdir_root}/{grp}/dyad_{dyad}"

        # Command to execute
        cmd = f"{cmd_head} {dyad} {wdir} {val_list_file} val_log.json"

        # Execute
        os.system(cmd)
