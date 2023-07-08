"""Typing and NoTyping classification using SOTA methods.

WARNING
-------
The following script processes one sample video at a time. This is
not optimal and is DEPRECATED in favour of batch processing.

NOTE:
----
    1. This script requires that we use python environment that has
    MMACTION2 installed.

    2. It also requires region proposal csv file.
"""


import argparse
import json
import pandas as pd
import os
import sys
import pytkit as pk
from tqdm import tqdm
from mmaction.apis import inference_recognizer, init_recognizer

def _arguments():
    """Parse input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=(
            """Typing detection framework using table ROI and keboard
            detections."""),
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

    # Load required parameters from the configuration
    rp_roi_only_file = cfg['rp_roi_only_file']
    out_file = cfg['ofile']
    mma2_cfg = cfg['mma2_cfg']
    mma2_ckpt = cfg['mma2_ckpt']

    # Load the network into GPU memory
    net = init_recognizer(mma2_cfg, mma2_ckpt, device='cuda:0')

    # Open a text file to write labels and probabilities
    out_file_txt = os.path.splitext(os.path.basename(out_file))[0]
    f = open(f"{cfg['oloc']}/{out_file_txt}.txt", "w")

    # Loop through each video
    tyrp_roi_only = pd.read_csv(rp_roi_only_file)
    video_names = tyrp_roi_only['name'].unique().tolist()
    for i, video_name in enumerate(video_names):

        # Typing proposals for current dataframe
        print(f"Classifying typing in {video_name}")
        tyrp_video = tyrp_roi_only[tyrp_roi_only['name'] == video_name].copy()
        tyrp_video['activity'] = ""
        tyrp_video['class_idx'] = -1
        tyrp_video['class_prob'] = "-1"

        # Read the video
        ivid = pk.Vid(f"{cfg['vdir']}/{video_name}", "read")

        # Loop through each instance in the video
        for ii, row in tqdm(tyrp_video.iterrows(), total=tyrp_video.shape[0], desc="INFO: Classifying"):

            # Spatio temporal trim coordinates
            bbox = [row['w0'],row['h0'], row['w'], row['h']]
            sfrm = row['f0']
            efrm = row['f1']
            vopth = (f"{cfg['oloc']}/temp_tsn.mp4")

            # Spatio temporal trim
            ivid.save_spatiotemporal_trim(sfrm, efrm, bbox, vopth)

            # Classifying
            ipred = inference_recognizer(net, vopth)
            ipred_class_idx = ipred[0][0]

            # Updating the columns
            if ipred_class_idx == 1:
                ipred_class = "typing"
                ipred_class_prob = ipred[0][1]
            else:
                ipred_class = "notyping"
                ipred_class_prob = 1 - ipred[0][1]

            # Updating the dataframe
            tyrp_video.at[ii, 'activity'] = ipred_class
            tyrp_video.at[ii, 'class_idx'] = ipred_class_idx
            tyrp_video.at[ii, 'class_prob'] = round(ipred_class_prob, 2)

            # This is because for 0.5 I am having problems in ROC curve
            if tyrp_video.at[ii, 'class_prob'] == 0.5:
                if ipred_class_idx == 1:
                    tyrp_video.at[ii, 'class_prob'] = 0.51
                else:
                    tyrp_video.at[ii, 'class_prob'] = 0.49
                    

                
            f.write(f"{ipred_class_idx}, {ipred_class}, {str(round(ipred_class_prob, 2))}\n")
            f.close()
            f = open(f"{cfg['oloc']}/{out_file_txt}.txt", "a")


        # Close the vide
        ivid.close()

        # If this is the first time, copy the proposal dataframe to typing dataframe
        # else concatinate
        if i == 0:
            tydf = tyrp_video
        else:
            tydf = pd.concat([tydf, tyrp_video])

    # Updating the csv file once every video
    tydf.to_csv(out_file, index=False)
    f.close()
    
