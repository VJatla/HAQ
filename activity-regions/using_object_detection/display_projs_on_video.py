"""
Display projectiosn on video. It can do projections every t seconds.

Example:
$ python display_projs_on_video.py <projections csv> <video> <proj interval>
"""
import pdb
import argparse
from aqua.objdets import ObjDetProjs


def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(description=("""
    Projects hand detections for an entire video.
        """))

    # Adding arguments
    args_inst.add_argument(
        "csv",
        type=str,
        help=("CSV file having object detection bounding boxes."))
    args_inst.add_argument(
        "vpth",
        type=str,
        help=("Video path"))
    args_inst.add_argument(
        "t",
        type=int,
        help=("Projections are calculated for every 't' seconds"))
    
    # Parse arguments
    args = args_inst.parse_args()

    # Create a dictionary having arguments and their values
    args_dict = {
        'csv': args.csv,
        'vpth':args.vpth,
        't':args.t
    }

    # Return arguments as dictionary
    return args_dict


def main():
    """ Main function """
    argd = _arguments()

    # Creting projection instance
    proj = ObjDetProjs(argd['csv'], argd['t'])

    # Creting a video with projection map for
    # every t seconds
    proj.display_on_video(argd['vpth'], ws=True)

# Execution starts here
if __name__ == "__main__":
    main()