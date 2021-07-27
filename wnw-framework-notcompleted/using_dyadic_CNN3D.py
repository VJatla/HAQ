"""
Spatio temporal writing detection. The output of this
script is a csv file with relevant information.

Example:
```bash
python using_dyadic_CNN3D.py ~/Dropbox/writing-nowriting-GT/C1L1P-C/20170413/G-C1L1P-Apr13-C-Windy_q2_04-07_30fps.mp4 ~/Dropbox/objectdetection-aolme/hand/C1L1P-C/20170413/G-C1L1P-Apr13-C-Windy_q2_04-07_30fps_detection.csv 4 /mnt/twotb/dyadic_nn/workdir/wnw/one_trim_per_instance_3sec_224/exp2_group_leave_one_out/C1L1P-C/dyad_4/epoch_19.pth
```
"""
import pdb
import pandas as pd
import argparse
from aqua.frameworks import Writing
from aqua.nn.models import DyadicCNN3D

def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(description=("""
    Spatio temporal writing detection. The output of this
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
                                 "writing/nowriting"))
    # Parse arguments
    args = args_inst.parse_args()

    # Create a dictionary having arguments and their values
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
    w = Writing(argd['video'], argd['bboxes'], cnn3d, argd['ckpt'], wdur=3)
    w.write_to_csv()

# Execution starts here
if __name__ == "__main__":
    main()







