"""Typing detection framework using table ROI and keboard
detections.

"""


import argparse
import json
from aqua.frameworks.typing.framework3_roi_kb import Typing3
import time

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
    st = time.time()
    ty.generate_typing_proposals_using_roi(dur=3, overwrite=False)
    et = time.time()
    print(f"Time taken for generating proposal csv file: {int(et-st)} sec.")

    # 2. Extract proposal regions into `proposals` directory
    st = time.time()
    ty.extract_typing_proposals_using_roi_kbdet(overwrite=False, model_fps=cfg['model_fps'])
    et = time.time()
    print(f"Time taken in extracting proposal videos: {int(et-st)} sec.")

    # 3. Classify proposals
    st = time.time()
    ty.classify_typing_proposals_roi_fast_approach(overwrite=False, batch_size=16, kb_det=True)
    et = time.time()
    print(f"Time taken in classifying the proposals: {int(et-st)} sec.")
    
    
