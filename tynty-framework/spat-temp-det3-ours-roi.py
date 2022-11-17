"""Typing detection framework using table ROI and keboard
detections."""


import argparse
import json
from aqua.frameworks.typing.framework3_roi_kb import Typing3

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

    # Initializing typing instance
    ty = Typing3(cfg)

    # Create a spatio-temporal typing proposals using
    #    1. table region of interest
    ty.generate_typing_proposals_using_roi(dur=3, overwrite=False)
    ty.classify_typing_proposals_roi(overwrite=False)
    
    # Create a spatio-temporal typing proposals and classify using
    #     1. keyboad detection
    #     2. table region of interest
    # ty.generate_typing_proposals_using_roi_kbdet(dur=3, overwrite=False)
    # ty.classify_typing_proposals_roi_and_kdet(overwrite=False)

    # Add a column comparing against ground truth
    # ty.add_gt_column()
    
    
