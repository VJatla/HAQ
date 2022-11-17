"""Writing and NoWriting classification using SOTA methods.

NOTE:
----
    1. This script requires that we use python environment that has
    MMACTION2 installed.
    2. To make this work we should already create writing proposal regions.
    These are created in `spat-temp-det3-ours-roi.py`.
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
            """Writing detection framework using table ROI and keboard
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
    wrp_roi_only = pd.read_csv(rp_roi_only_file)
    video_names = wrp_roi_only['name'].unique().tolist()
    for i, video_name in enumerate(video_names):

        # Writing proposals for current dataframe
        print(f"Classifying writing in {video_name}")
        wrp_video = wrp_roi_only[wrp_roi_only['name'] == video_name].copy()
        wrp_video['activity'] = ""
        wrp_video['class_idx'] = -1
        wrp_video['class_prob'] = "-1"

        # Read the video
        ivid = pk.Vid(f"{cfg['vdir']}/{video_name}", "read")

        # Loop through each instance in the video
        for ii, row in tqdm(wrp_video.iterrows(), total=wrp_video.shape[0], desc="INFO: Classifying"):

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
                ipred_class = "writing"
                ipred_class_prob = ipred[0][1]
            else:
                ipred_class = "nowriting"
                ipred_class_prob = 1 - ipred[0][1]

            # Updating the dataframe
            wrp_video.at[ii, 'activity'] = ipred_class
            wrp_video.at[ii, 'class_idx'] = ipred_class_idx
            wrp_video.at[ii, 'class_prob'] = round(ipred_class_prob, 2)

            # This is because for 0.5 I am having problems in ROC curve
            if wrp_video.at[ii, 'class_prob'] == 0.5:
                if ipred_class_idx == 1:
                    wrp_video.at[ii, 'class_prob'] = 0.51
                else:
                    wrp_video.at[ii, 'class_prob'] = 0.49
                    

                
            f.write(f"{ipred_class_idx}, {ipred_class}, {str(round(ipred_class_prob, 2))}\n")
            f.close()
            f = open(f"{cfg['oloc']}/{out_file_txt}.txt", "a")


        # Close the vide
        ivid.close()

        # If this is the first time, copy the proposal dataframe to writing dataframe
        # else concatinate
        if i == 0:
            wdf = wrp_video
        else:
            wdf = pd.concat([wdf, wrp_video])

    # Updating the csv file once every video
    wdf.to_csv(out_file, index=False)
    f.close()
    
