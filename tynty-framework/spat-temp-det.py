"""
Spatio temporal typing detection. The output of this
script is a csv file with relevant information.
"""
import pdb
import pandas as pd
import argparse
from aqua.frameworks.typing import Typing
from aqua.nn.models import DyadicCNN3D
from torchvision import models
from torchsummary import summary

def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(description=("""
    Spatio temporal typing detection. The output of this
    script is a csv file with relevant information.
        """))

    # Adding arguments
    args_inst.add_argument("video", type=str, help=("Video path"))
    args_inst.add_argument("bboxes", type=str,
                           help=("Path to CSV file having keyboard bounding "
                                 " boxes."))
    args_inst.add_argument("ndyads", type=int, help=("Number of dyads"))
    args_inst.add_argument("ckpt", type=str,
                           help=("Path to check point file that is trained on "
                                 "typing/notyping"))
    # Parse arguments
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {'video': args.video,
                 'ndyads':args.ndyads,
                 'bboxes': args.bboxes,
                 'ckpt':args.ckpt}

    # Return arguments as dictionary
    return args_dict


def main():
    """ Main function """
    argd = _arguments()

    # Creating a network instance
    cnn3d = DyadicCNN3D(argd['ndyads'], [3, 90, 224, 224])
    summary(cnn3d,(3,90,224,224))
    ty = Typing(argd['video'], argd['bboxes'], cnn3d, argd['ckpt'], tydur=3)
    ty.write_to_csv()

# Execution starts here
if __name__ == "__main__":
    main()
