"""
The following script creates `trn_videos.txt`, `val_videos.txt` and
`tst_videos.txt` for aolme trimmed videos dataset in compliance with
TSN standards. This script is written to run in `mmaction` docker container.
"""
import pdb
from aqua.data_tools import AOLMETrimmedVideos
import argparse


def main():
    """ Execution starts here
    """
    args = {
        'rdir':
        "/mnt/twotb/aolme_datasets/wnw/trimmed_videos/one_trim_per_instance_3sec_224",
        'split_info_path':
        "/home/vj/Dropbox/writing-nowriting-GT/trn-val-tst-splits.csv",
        'labels': {
            'nowriting': 0,
            'writing': 1
        },
        # If groups := [] then all the groups are considered
        'groups':['C1L1P-B', 'C1L1P-C', 'C2L1P-B', 'C2L1P-C']
    }
    odir = args['rdir']+"/group_leave_one_out/exp2"

    # Video and rawframe directories
    video_dir = args['rdir']

    # Creating a trimmed video instance
    trmvids = AOLMETrimmedVideos(video_dir, ".mp4")

    # Create lists
    # trmvids.create_video_tvt_lists(args['rdir'], args['labels'],
    #                                args['split_info_path'])

    # Create subsample list --> Maintaining Diversity'])
    trmvids.create_leave_onegroup_out_lists(args['groups'], odir, args['labels'], args['split_info_path'])

if __name__ == "__main__":
    main()
