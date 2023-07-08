"""Typing and NoTyping classification using SOTA batch processing.

Note
----
1. This script expects the predictions as `pkl` files we get after
using mmaction2's tools/test.py script

2. The propsal csv file having typing region proposals
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
            """Typing and NoTyping classification using SOTA batch processing."""),
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
    tydf = proposal_df.copy()
    activity_lst = ["notyping"]*len(tydf)
    class_idx_lst = [0]*len(tydf)
    class_prob_lst = [0]*len(tydf)

    # Going through each prediction
    for i, pred in enumerate(preds):
        pred_prob_typing = round(pred[1], 2)
        class_prob_lst[i] = pred_prob_typing

        if pred_prob_typing >= 0.5:
            activity_lst[i] = "typing"
            class_idx_lst[i] = 1

    # Add to typing dataframe
    tydf['activity'] = activity_lst
    tydf['class_idx'] = class_idx_lst
    tydf['class_prob'] = class_prob_lst

    # Write typing dataframe
    tydf.to_csv(cfg['output_csv'], index=False)
    
