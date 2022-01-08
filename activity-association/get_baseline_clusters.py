"""
DESCRIPTION
-----------
The followng script clusters activities using hand 
distances for an entire session. 

If two activities are `factor*hand_size` apart then 
we classify the activities to belong to different 
clusters.

OUTPUT
------
Updates excel file that contains activity instances with "Base Clusters" sheet.

ASSUMPTIONS
-----------
1. The kids do not move during a session

USAGE
-----
python get_gt_basic_clusters_excel.py  <activity config file>

EXAMPLE
-------
# Linux
python get_gt_base_clusters_excel.py ./json_configs/gt/writing/C1L1P_Mar02_E.json

# Windows

"""
import os
import argparse
import pdb
import json
import math
import pandas as pd
from openpyxl import load_workbook

# User defined libraries
import pytkit as pk
from hand_size import HandSize
from base_line_clusters import BaseLineClusters

def _arguments():

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=(
            '''
            DESCRIPTION
            -----------
            The followng script clusters activities using hand 
            distances.

            If two activities are `factor*hand_size` apart then 
            we classify the activities to belong to different 
            clusters.
            '''
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Adding arguments
    args_inst.add_argument(
        'cfg',
        type=str,
        help=("JSON configuration file"))
    
    
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

    # Open log file to write
    logf = open(cfg['session_level_log'], 'w')

    # Read the Machine readable sheet
    gtdf = pd.read_excel(
        cfg['gt_excel'], sheet_name="Machine readable"
    )
    hr_gtdf = pd.read_excel(
        cfg['gt_excel'], sheet_name="Human readable"
    )
    mr_gtdf = pd.read_excel(
        cfg['gt_excel'], sheet_name="Machine readable"
    )

    # Get hand size using first 60 seconds of first video
    vname_first = cfg['vname1']
    vname_no_ext = os.path.splitext(vname_first)[0]
    hands_csv = (
        f"{cfg['hands_csv_loc']}/{vname_no_ext}_one_det_per_sec.csv"
    )
    HS = HandSize(hands_csv)
    hs = HS.get_hand_size(60)
    th = cfg['hand_dist_th']

    # Cluster activities using base algorithm
    BLC = BaseLineClusters(hs, th, gtdf)
    BLC.cluster_activities_base_line()
    
    # Calculate the performance using adjusted Rand index.
    perf = BLC.evaluate_clusters("person", ['RI'])

    # Calculate if the cluster is stable
    stability_str = BLC.get_cluster_stability_using_meta_data(
        cfg['num_persons'], cfg['session_csv']
    )

    # Write to log file
    log_text = (
    f"{stability_str} RI={perf['RI']}"
    )
    logf.write(log_text)
    
    # Add cluster column to Human/Machine readable sheets
    hr_gtdf['baseline_labels'] = BLC.baseline_df['baseline_labels']
    mr_gtdf['baseline_labels'] = BLC.baseline_df['baseline_labels']

    # Add sheet with clusters
    book = load_workbook(cfg['gt_excel'])
    writer = pd.ExcelWriter(cfg['gt_excel'], engine = 'openpyxl')
    writer.book = book
    hr_gtdf.to_excel(writer, sheet_name = 'BLC-HR', index=False)
    mr_gtdf.to_excel(writer, sheet_name = 'BLC-MR', index=False)
    writer.save()
    writer.close()

    # Create videos with clusters
    BLC.to_video(cfg['vdir'], "BLC")

    # Print information
    print(f"{cfg['vname1']}: {log_text}")

    # Close log
    logf.close()
