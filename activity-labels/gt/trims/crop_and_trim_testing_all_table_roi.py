"""
DESCRIPTION
-----------
Crop and trim activity instances form videos using ground truth and
object detections.

USAGE
-----
python crop_and_trim.py ???

EXAMPLE
-------

"""


import sys
import argparse
import keyboard
from aqua.data_tools import TabROIGTTrims



def get_json_file_path():
    """ Jason file input """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description="""
        Description
        -----------
        <description>

        Example usage:
        --------------
        <example usage>
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args_inst.add_argument("cfg", type=str, help="Path to configuration file")

    if len(sys.argv) == 1:
        return ""

    args = args_inst.parse_args()
    return args.cfg


# Execution starts from here
if __name__ == "__main__":

    # Default configuration
    DEF_CFG = "/home/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/activity-labels/gt/trims/cfgs/typing/C1L1P-A/20170216/C1L1P-A_20170216.json"

    # Getting configuration path
    cfg_path = get_json_file_path()

    # Use default configuration if it is not provided
    if cfg_path == "":
        key_press = input(f"Enter c to continue using configuration\n{DEF_CFG}\n")
        if key_press == 'c':
            cfg_path = DEF_CFG
        else:
            sys.exit()

    # Initialize ground truth trimming instance
    gt_trims = TabROIGTTrims(cfg_path)

    # Creates a DataFrame with 3 second activity instances.
    gt_trims.get_3sec_activity_instance()

    # Updating the csv file with roi coordinates
    gt_trims.get_roi_bounding_boxes()

    # Trim instances uisng object detection and store in proper directory
    gt_trims.trim_bboxes()
