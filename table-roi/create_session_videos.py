"""
DESCRIPTION
-----------
The following script creates once csv file for a session called,
`session_table_rois.csv`. The csv file will have coordinates of bounding
boxes associated with each person in the group.

USAGE
-----
python create_session_videos.py

EXAMPLE
-------
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

# Calling as script
if __name__ == "__main__":
    
    # Session paths in Windows
    # paths = [
    #     "/home/vj/Dropbox/table_roi_annotation/C1L1P-C/20170330",
    #     "/home/vj/Dropbox/table_roi_annotation/C1L1P-C/20170413",
    #     "/home/vj/Dropbox/table_roi_annotation/C1L1P-E/20170302",
    #     "/home/vj/Dropbox/table_roi_annotation/C2L1P-B/20180223",
    #     "/home/vj/Dropbox/table_roi_annotation/C2L1P-D/20180308",
    #     "/home/vj/Dropbox/table_roi_annotation/C3L1P-C/20190411",
    #     "/home/vj/Dropbox/table_roi_annotation/C3L1P-D/20190221"
    # ]
    paths = [
        "/home/vj/Dropbox/table_roi_annotation/C1L1P-C/20170330",
    ]
    
    for p in paths:
        
        # Create an ROI instance
        roi = ROI(p, 30)

        # Create session level video taking one frame
        roi.create_session_video()
