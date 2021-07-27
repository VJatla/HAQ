"""
Plots object detection projecitons for a video.
"""
import pdb
import argparse
import numpy as np
from aqua.objdets import ObjDetProjs
import matplotlib.pyplot as plt

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
    
    # Parse arguments
    args = args_inst.parse_args()

    # Create a dictionary having arguments and their values
    args_dict = {'csv': args.csv}

    # Return arguments as dictionary
    return args_dict


def main():
    """ Main function """
    argd = _arguments()

    # Creting heat map instance
    proj = ObjDetProjs(argd['csv'], -1)
    proj_img = proj.get_proj_map_for_full_video()

    # Threshold projeciton map at 90 percentile
    percentile_th = 90
    proj_flat = proj_img.flatten()
    proj_flat_nonzero = proj_flat[np.nonzero(proj_flat)]
    qth_val = np.percentile(proj_flat_nonzero, percentile_th)
    proj_th = 1*(proj_img > qth_val)

    fig, ax = plt.subplots()
    ax.imshow(proj_img, cmap="gray")
    ax.set_title("Projections in Gray scale")
    
    fig, ax = plt.subplots()
    ax.imshow(proj_th, cmap="gray")
    ax.set_title("Projections threshold at 90 percentile")
    plt.show()
    

# Execution starts here
if __name__ == "__main__":
    main()

