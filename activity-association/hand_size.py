"""
DESCRIPTION
-----------
This file contains methods that can estimate hand size from the
detections.

USAGE
-----
python hand_size.py <CSV file having hand detections> <duration in seconds>


EXAMPLE
-------
# As a script, in Linux
python hand_size.py /home/vj/Dropbox/objectdetection-aolme/hand-tensorflow/C1L1P-E/20170302/G-C1L1P-Mar02-E-Irma_q2_02-08_30fps_detection.csv 10
"""
import argparse
import pdb
import pandas as pd

# User defined libraries
import pytkit as pk

class HandSize:
    """ 
    Provides methods to estimate hand size from hand detections. The
    detections are provided as csv file.
    """

    # Class variables
    hand_df = pd.DataFrame()
    """ Dataframe containing hand bounding boxes """

    # Constructor/Class initialization
    def __init__(self, csv_path):

        # Load hand detections to dataframe
        if pk.check_file(csv_path):
            self.hand_df = pd.read_csv(csv_path)
        else:
            raise Exception(f"USER_ERROR: \n\t{csv_path} not found.")

    def get_hand_size(self, dur):
        """
        Returns median hand size in pixel^2 within `dur` seconds.

        Parameters
        ----------
        dur : int
            Duration in seconds

        Returns
        -------
        Hand size in piexl^2.
        """

        # Extracting hand instances within `dur` seconds
        FPS = self.hand_df['FPS'].unique().item()
        df = self.hand_df[self.hand_df['f0'] <= FPS*dur].copy()

        # Creating size column
        df['size'] = df['h']*df['w']

        # Taking median
        hand_size = df['size'].median()

        # Returning
        return hand_size
        




def _arguments():
    """Parses input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=(
            """
            DESCRIPTION
            -----------
            This script contains modules to get hand size from the csv
            files containing hand bounding boxes.
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Adding arguments
    args_inst.add_argument(
        "hands_csv",
        type=str,
        help=(
            """CSV file having hand detections. Note the script works
            with hands before the post processing proposed by Sravani
            Teeparthi in her Thesis.""")
    )
    args_inst.add_argument(
        "dur",
        type=int,
        help=("""Duration in seconds to cosnider in estimating hand 
        size."""))
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {
        'hands_csv' : args.hands_csv,
        'dur' : args.dur
    }

    # Return arguments as dictionary
    # Hello world how are you doing
    return args_dict


# Calling as script
if __name__ == "__main__":
    # Input argumetns
    args = _arguments()
    hands_csv = args["hands_csv"]
    dur_to_consider = args["dur"]
    
    HS = HandSize(hands_csv)

    hand_size = HS.get_hand_size(dur_to_consider)
    print(hand_size)
    
