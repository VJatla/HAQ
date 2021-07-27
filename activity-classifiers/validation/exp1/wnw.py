import sys
import pdb
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from aqua.nn.models import DyadicCNN3D
from torch.utils.data import DataLoader
from aqua.nn.dloaders import AOLMETrmsDLoader
from aqua.nn.validator import SGPU_Val


def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(description=("""
    Trains (does not validate) dyadic neural network.
        """))

    # Adding arguments
    args_inst.add_argument("ndyads", type=int, help=("Number of dyads"))
    args_inst.add_argument("workdir",
                           type=str,
                           help=("Training directory having checkpoints"))
    args_inst.add_argument("vlist",
                           type=str,
                           help=("Text file having validation list"))
    args_inst.add_argument(
        "log_name",
        type=str,
        help=("Name of validation log file. It is saved in working directory"))

    # Parse arguments
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {
        'ndyads': args.ndyads,
        'workdir': args.workdir,
        'vlist': args.vlist,
        'log_name': args.log_name,
    }

    # Return arguments as dictionary
    return args_dict


def main():
    # Arguments
    argd = _arguments()

    # Dataset locations and properties
    max_epochs = 50
    ndyads = argd['ndyads']
    workdir = argd['workdir']
    vdir = "/mnt/twotb/aolme_datasets/wnw/trimmed_videos/one_trim_per_instance_3sec_224"
    vlist = f"{argd['vlist']}"

    # Validation data
    vdata = AOLMETrmsDLoader(vdir, vlist, oshape=(224, 224))
    vloader = DataLoader(vdata, batch_size=16, shuffle=True, num_workers=16)

    # Build model
    cnn3d = DyadicCNN3D(ndyads, [3, 90, 224, 224])

    # Loss and optimizer
    criterion = nn.BCELoss()  # Using Binary Cross Entropy loss

    # Validator
    validator = SGPU_Val(workdir,
                         cnn3d,
                         vloader,
                         max_epochs,
                         criterion,
                         eskip=1)

    # Validate
    log_path = f"{workdir}/{argd['log_name']}"
    validator.validate(log_path)


# Execution starts here
if __name__ == "__main__":
    main()
