"""
DESCRIPTION
-----------
Creating session level video taking `n` frames every second.

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
    
    # Session paths
    paths = [
        "/home/vj/Dropbox/table_roi_annotation/C1L1P-A/20170216",
    ]
    
    for p in paths:
        
        # Create an ROI instance
        roi = ROI(p, 30)

        # Create session level video taking one frame
        roi.create_session_video()
