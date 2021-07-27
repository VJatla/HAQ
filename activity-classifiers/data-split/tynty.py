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
        "/mnt/twotb/aolme_datasets/tynty/trimmed_videos/one_trim_per_instance_3sec_224",
        'split_info_path':
        "/home/vj/Dropbox/typing-notyping/trn-val-tst-splits.csv",
        'labels': {
            'notyping': 0,
            'typing': 1
        },
        # 'num_samples': (-1, -1, -1) # <-- To create lists for all the trimmed videos
        'num_samples': (0, 50, 200)  # <-- Samples per activity
    }

    # Video and rawframe directories
    video_dir = args['rdir']

    # Creating a trimmed video instance
    trmvids = AOLMETrimmedVideos(video_dir, ".mp4")

    # Create lists
    # trmvids.create_video_tvt_lists(args['rdir'], args['labels'],
    #                                args['split_info_path'])

    # Create subsample list --> Maintaining Diversity
    trmvids.create_subsampled_tvt_lists(args['rdir'], args['labels'],
                                        args['split_info_path'],
                                        args['num_samples'])


if __name__ == "__main__":
    main()
