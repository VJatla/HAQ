"""
DESCRIPTION
-----------
Creating session video taking 1 frame every 1 second.

USAGE
-----
python create_session_videos.py <roi directory location> <comma separated video indexes> or "all"

EXAMPLE
-------
python create_session_videos.py /home/vj/Dropbox/table_roi_annotation/C1L1P-A/20170216 8
"""
import argparse
from tqdm import tqdm
import pdb
import pandas as pd
import numpy as np
from matplotlib import cm
import cv2

# User defined
import pytkit as pk
from rois import ROI


def _arguments():
    """Parse input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=("Creating session video taking 1 frame every 1 second."), formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args_inst.add_argument("roi_path", type=str, help="Table roi directory location")
    args_inst.add_argument(
        "vidxs",
        type=str,
        help=("Comma separated values of video indexes for which we"
              "have activity ground truth. If we have to use all the videos we pass the string all.")
    )
    args = args_inst.parse_args()

    args_dict = {"roi_path": args.roi_path, "vidxs": args.vidxs}
    return args_dict


# Execution starts from here
if __name__ == "__main__":
    
    # Arguments
    args_dict = _arguments()
    p = args_dict['roi_path']
    vidxs = args_dict['vidxs']

    # Initializing ROI instance
    roi = ROI(p, 30)

    # Create session level video taking one frame
    roi.create_session_video(vidxs)
