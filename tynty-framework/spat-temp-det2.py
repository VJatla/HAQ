"""
DESCRIPTION
-----------
Detects typing in a video using keyboard detections.

USAGE
-----
python spat-temp-det2.py <configuration json file>


EXAMPLE
-------
python spat-temp-det2.py ~/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/tynty-framework/cfg/C1L1P-C/20170413/cfg_03-07.json
"""


import sys
import argparse
import keyboard
import pdb
import pandas as pd
import argparse
from aqua.frameworks.typing.framework2 import Typing
from aqua.nn.models import DyadicCNN3D
from torchvision import models
from torchsummary import summary
import json



def get_json_file_path():
    """ Jason file input """

    # Initialize argument instance
    args_inst = argparse.ArgumentParser(
        description="""
        DESCRIPTION
        -----------
        Detects typing in a video using keyboard region proposals.

        USAGE
        -----
        python spat-temp-det.py <configuration json file>


        EXAMPLE
        -------
        python spat-temp-det2.py ~/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/tynty-framework/cfg/C1L1P-C/20170413/cfg_03-07.json
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
    DEF_CFG = "/home/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/tynty-framework/cfg/C1L1P-C/20170413/cfg_03-07.json"

    # Getting configuration path
    cfg_path = get_json_file_path()

    # Use default configuration if it is not provided
    if cfg_path == "":
        # key_press = input(f"Enter c to continue using configuration\n{DEF_CFG}\n")
        key_press = 'c'
        if key_press == 'c':
            cfg_path = DEF_CFG
        else:
            sys.exit()

    # Parsing the configuration file
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Creating spatio temporal typing instance
    ty = Typing(cfg)
    
    # Calculating typing region proposal based on keyboard detectons and
    # Table ROI
    tyrp = ty.get_typing_proposals()

    # Classifying each typing proposal region
    tydf = ty.classify_typing_proposals(tyrp, debug=True)
    tydf.to_csv("alg-wnw.csv", index=False)

    # Execution starts here
    if __name__ == "__main__":
        main()
