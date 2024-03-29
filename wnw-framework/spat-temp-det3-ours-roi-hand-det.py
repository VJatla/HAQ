"""Writing detection framework using table ROI and keboard
detections."""


import argparse
import json
from aqua.frameworks.writing.framework3_roi_hands import Writing3
import sys

def _arguments():
    """Parse input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=("Writing detection framework using table ROI and keboard detections."),
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

    # Initializing typing instance
    wr = Writing3(cfg)

    # 1. Table region proposals using table ROI
    wr.generate_writing_proposals_using_roi(dur=3, overwrite=False)

    # 2. Uses hand detection to improve writing classification by
    #    removing writing region proposals.
    wr.classify_proposals_using_hand_det(overwrite=True)

    # 2. Classifying without using hand detections
    # wr.classify_writing_proposals_roi(overwrite=False)
    


