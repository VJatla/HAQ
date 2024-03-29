"""
Description
-----------
    The following code creates labelled video (labelled with activity
    labels). This is done to provide easy visualization.

Output
------
    The output of this script is a video with activity labelels bounding boxes.

Example
-------
    python create_labeled_video.py ~/Dropbox/typing-notyping/C1L1P-E/20170302/gTruth-tynty_30fps.csv ~/Dropbox/typing-notyping/kid-pseudonym-mapping.csv typing
"""


import argparse
import os
from aqua.data_tools.aolme import AOLMEActivityLabels

def _arguments():
    """Parses input arguments."""

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=(
            """
            The following code creates labelled video (labelled with activity
            labels). This is done to provide easy visualization for researchers.
            """))

    # Adding arguments
    args_inst.add_argument("acty_csv", type=str, help=("CSV file having activity instance"))
    args_inst.add_argument("names_csv", type=str, help=("CSV file having names."))
    args_inst.add_argument("activity", type=str, help=("Activity that is being processed."))
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {'acty_csv': args.acty_csv, 'names_csv': args.names_csv,
                 'activity':args.activity}

    # Return arguments as dictionary
    # Hello world how are you doing
    return args_dict


def main():
    """Main function."""
    argd = _arguments()
    acty_csv = argd['acty_csv']
    names_csv = argd['names_csv']
    activity = argd['activity']

    # Extract root directory and labels file name from csv file name
    rdir = os.path.dirname(acty_csv)
    acty_csv_name = os.path.basename(acty_csv)

    act_labels = AOLMEActivityLabels(rdir, acty_csv_name)
    act_labels.create_labeled_videos(names_csv, activity)

# Execution starts here
if __name__ == "__main__":
    main()
