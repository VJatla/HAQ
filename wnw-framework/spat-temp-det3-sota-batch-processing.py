"""Writing and NoWriting classification using SOTA batch processing.

Note
----
1. This script expects the predictions as `pkl` files we get after
using mmaction2's tools/test.py script

2. The propsal csv file having writing region proposals
"""


import argparse
import json
import pandas as pd
import os
import sys
import pytkit as pk
from tqdm import tqdm
import pickle

def _arguments():
    """Parse input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=(
            """Writing and NoWriting classification using SOTA batch processing."""),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args_inst.add_argument("cfg_path", type=str, help="Configuration JSON file.")
    args = args_inst.parse_args()

    return args.cfg_path


# Ececution starts here.
if __name__ == "__main__":

    # Load configuration JSON file as a dictionary
    cfg_path = _arguments()
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Loading input files
    pred_file_handle = open(cfg['pred_pkl'], 'rb')
    preds = pickle.load(pred_file_handle)

    # Reading the proposal csv file
    proposal_df = pd.read_csv(cfg['rp_roi_only_file'])

    # Number of propoosals and prediction check
    num_porposals = len(proposal_df)
    num_preds = len(preds)
    if num_porposals != num_preds:
        raise Exception(f"Number of proposals ({num_porposals}) != number of predictions ({num_preds})")

    # Add default colums to the proposal dataframe
    wdf = proposal_df.copy()
    activity_lst = ["nowriting"]*len(wdf)
    class_idx_lst = [0]*len(wdf)
    class_prob_lst = [0]*len(wdf)

    # Going through each prediction
    for i, pred in enumerate(preds):
        pred_prob_writing = round(pred[1], 2)
        class_prob_lst[i] = pred_prob_writing

        if pred_prob_writing >= 0.5:
            activity_lst[i] = "writing"
            class_idx_lst[i] = 1

    # Add to writing dataframe
    wdf['activity'] = activity_lst
    wdf['class_idx'] = class_idx_lst
    wdf['class_prob'] = class_prob_lst

    # Write writing dataframe
    wdf.to_csv(cfg['output_csv'], index=False)
    
