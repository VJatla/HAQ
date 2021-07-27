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
from aqua.nn.trainer import SGPU_TrnOnly


def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(description=("""
    Trains (does not validate) dyadic neural network.
        """))

    # Adding arguments
    args_inst.add_argument("vdir", type=str, help=("Directory to find trimmed videos"))
    args_inst.add_argument("ndyads", type=int, help=("Number of dyads"))
    args_inst.add_argument("trnlst", type=str, help=("Training list text file"))
    args_inst.add_argument("workdir", type=str, help=("Working directory"))

    # Parse arguments
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {'ndyads': args.ndyads, 'vdir':args.vdir,
                 'trnlst':args.trnlst, 'workdir': args.workdir}

    # Return arguments as dictionary
    return args_dict


def main():
    # Arguments
    argd = _arguments()
    ndyads = argd['ndyads']

    # Dataset locations and properties
    vdir = argd['vdir']
    trn_list = argd['trnlst']
    input_shape = [3, 90, 224, 224]
    max_epochs = 50
    cur_cuda_device_id = 0

    # Initialize single gpu training instance
    training_params = {
        "input_shape": tuple(input_shape),
        "max_epochs": max_epochs,
        "work_dir": argd['workdir'],
        "ckpt_save_interval": 1,
        "log_pth": f"{argd['workdir']}/trn_log.json"
    }

    # Training data loader
    train_data = AOLMETrmsDLoader(vdir, trn_list, oshape=(224, 224))
    trainloader = DataLoader(train_data,
                             batch_size=16,
                             shuffle=True,
                             num_workers=16)

    # Build model
    cnn3d = DyadicCNN3D(ndyads, [3, 90, 224, 224])

    # Loss and optimizer
    criterion = nn.BCELoss()  # Using Binary Cross Entropy loss
    optimizer = optim.SGD(cnn3d.parameters(), lr=0.001, momentum=0.9)

    trainer = SGPU_TrnOnly(training_params,
                           cnn3d,
                           optimizer,
                           criterion,
                           trainloader,
                           cuda_device_id=cur_cuda_device_id)

    # Train
    training_time = trainer.train()


# Execution starts here
if __name__ == "__main__":
    main()
