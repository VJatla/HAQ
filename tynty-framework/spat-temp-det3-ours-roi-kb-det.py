"""Typing detection framework using table ROI and keboard
detections."""


import argparse
import json
from aqua.frameworks.typing.framework3_roi_kb import Typing3

def _arguments():
    """Parse input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=("Typing detection framework using table ROI and keboard detections."),
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

    # 1. Table region proposals using table ROI
    ty.generate_typing_proposals_using_roi(dur=3, overwrite=False)

    # 2. Uses keyboard and hand detection to improve typing classification
    ty.classify_proposals_using_kb_det(overwrite=False)
    
    
