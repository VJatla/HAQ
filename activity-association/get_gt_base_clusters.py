"""
DESCRIPTION
-----------
The followng script clusters activities using hand 
distances.

If two activities are `factor*hand_size` apart then 
we classify the activities to belong to different 
clusters.

The output of this script is a csv file with each activity associated
to a cluster.

USAGE
-----
python get_basic_association.py <hands csv path> <activity csv path>

EXAMPLE
-------
# Linux

# Windows

"""
import os
import argparse
import pdb
import json
import math
import pandas as pd

# User defined libraries
import pytkit as pk
from hand_size import HandSize
from base_clusters import BaseClusters

def _arguments():
    """Parses input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=(
            """
            DESCRIPTION
            -----------
            The followng script clusters activities using hand 
            distances.

            If two activities are `factor*hand_size` apart then 
            we classify the activities to belong to different 
            clusters.
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Adding arguments
    args_inst.add_argument(
        "cfg",
        type=str,
        help=("""JSON configuration file"""))
    
    
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {
        'cfg'    : args.cfg,
    }

    # Return arguments as dictionary
    return args_dict


# Calling as script
if __name__ == "__main__":
    # Arguments
    cfg_path = _arguments()['cfg']
    with open(cfg_path) as cfg_file:
        cfg = json.load(cfg_file)

    # Loop through each video in ground truth
    gtdf = pd.read_csv(cfg['gt_csv'])
    gtdf = gtdf[gtdf['activity'] == cfg['activity']].copy()
    vnames = list(gtdf['name'].unique())

    # Open log file to write
    logf = open(cfg['out_log'], 'w')

    # Loop through each video in ground truth
    for vname in vnames:

        # Get hand size for current video
        vname_no_ext = os.path.splitext(vname)[0]
        hands_csv = (
            f"{cfg['hands_csv_loc']}/{vname_no_ext}_one_det_per_sec.csv"
        )
        HS = HandSize(hands_csv)
        hs = HS.get_hand_size(10)

        # Get activity dataframe for current video
        actdf = gtdf[gtdf['name'] == vname].copy()

        # Cluster activities using base algorithm
        BC = BaseClusters(hs, actdf)
        BC.cluster_activities()

        # Save to CSV file
        BC.to_csv(
            os.path.dirname(cfg['gt_csv']),
            vname,
            cfg['labeled_csv_postfix']
        )

        # Calculate the performance using adjusted Rand index.
        perf = BC.evaluate_clusters("person", ['RI'])
        logf.write(f"{vname} : (RI, {perf['RI']})\n")

        # Write video for perf < 0.8
        if perf['RI'] < 0.8:
            # Asssuming the video file is located in the same
            # directory as the csv file containing activities
            vpth = f"{os.path.dirname(cfg['gt_csv'])}/{vname}"
            BC.to_video(vpth, cfg['out_vid_postfix'])

    logf.close()
