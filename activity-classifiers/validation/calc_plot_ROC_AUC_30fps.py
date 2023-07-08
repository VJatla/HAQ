"""
Description
-----------
Plots ROC and calculates AUC curve for validation data.

Usage
-----
#                 Typing/notyping validation
time CUDA_VISIBLE_DEVICES=0 python calc_plot_ROC_AUC_30fps.py \
/mnt/twotb/aolme_datasets/tynty_table_roi/resized_224_30fps \
4 \
/mnt/twelvetb/vj/dyadic_nn/tynty_table_roi/resized_224_30fps/Dec26/run0/best_epoch50.pth \
/mnt/twotb/aolme_datasets/tynty_table_roi/resized_224_30fps/val_videos_all.txt

#                 Typing/notyping testing
time CUDA_VISIBLE_DEVICES=0 python calc_plot_ROC_AUC_30fps.py \
/mnt/twotb/aolme_datasets/tynty_table_roi/resized_224_30fps \
4 \
/mnt/twelvetb/vj/dyadic_nn/tynty_table_roi/resized_224_30fps/Dec26/run0/best_epoch50.pth \
/mnt/twotb/aolme_datasets/tynty_table_roi/resized_224_30fps/tst_videos_all.txt

#                 Writing/nowriting validation
time CUDA_VISIBLE_DEVICES=0 python calc_plot_ROC_AUC_30fps.py \
/mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps \
4 \
/mnt/twelvetb/vj/dyadic_nn/wnw_table_roi/resized_224_30fps/Dec26/run4/best_epoch50.pth \
/mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps/val_videos_all.txt

#                 Writing/nowriting testing
time CUDA_VISIBLE_DEVICES=0 python calc_plot_ROC_AUC_30fps.py \
/mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps \
4 \
/mnt/twelvetb/vj/dyadic_nn/wnw_table_roi/resized_224_30fps/Dec26/run4/best_epoch50.pth \
/mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps/tst_videos_all.txt

"""

import os
import sys
import pdb
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from barbar import Bar
from torchsummary import summary
from aqua.nn.models import DyadicCNN3D
from aqua.nn.models import DyadicCNN3DV2
from torch.utils.data import DataLoader
from aqua.nn.dloaders import AOLMEValTrmsDLoader
from aqua.nn.validator import SGPU_Val
from sklearn import metrics
import matplotlib.pyplot as plt


def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(description=("""
    Trains (does not validate) dyadic neural network.
        """))

    # Adding arguments
    args_inst.add_argument("vrdir", type=str, help=("Directory to find trimmed videos"))
    args_inst.add_argument("ndyads", type=int, help=("Number of dyads"))
    args_inst.add_argument("best_ckpt",
                           type=str,
                           help=("Path of best check point"))
    args_inst.add_argument("vlist",
                           type=str,
                           help=("Text file having validation list"))

    # Parse arguments
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {
        'vrdir':args.vrdir,
        'ndyads': args.ndyads,
        'best_ckpt': args.best_ckpt,
        'vlist': args.vlist
    }

    # Return arguments as dictionary
    return args_dict


def get_cuda_device(cuda_device_id):
        """ Sets NVIDIA device with id `cuda_device_id` to
        be used for training.

        Parameters
        ----------
        net: Custom Network Instance
            Instance of pytorch custom network we will be training
        cuda_device_id: int
            CUDA device ID
        """
        if not torch.cuda.is_available():
            raise Exception("ERROR: Cuda is not found in this environment")

        num_cuda_devices = torch.cuda.device_count()
        cuda_devices = list(range(0,
                                  num_cuda_devices))  # cuda index start from 0

        if not (cuda_device_id <= num_cuda_devices - 1):
            raise Exception(
                f"ERROR: Cuda device {cuda_device_id} is not found.\n"
                f"ERROR: Found {num_cuda_devices}("
                f"{cuda_devices}), Cuda devices")

        # If no errors load network onto cuda device
        print("INFO: Network sucessfully loaded into "
              f"CUDA device {cuda_device_id}")
        return f"cuda:{cuda_device_id}"
    

def load_net(ckpt, depth, input_shape_30fps):
        """ Load neural network to GPU. """

        print("INFO: Loading Trained network to GPU ...")
        
        # Creating an instance of Dyadic 3D-CNN
        input_shape = input_shape_30fps.copy()
        net = DyadicCNN3DV2(depth, input_shape)
        net.to("cuda:0")
        summary(net, tuple(input_shape_30fps))

        # Loading the net with trained weights to cuda device 0
        ckpt_weights = torch.load(ckpt)
        net.load_state_dict(ckpt_weights['model_state_dict'])
        net.eval()

        return net

    
def eval_(device, net, data):
        """ 
        """
        # Loss and optimizer
        criterion = nn.BCELoss()  # Using Binary Cross Entropy loss
        
        labels, inputs = (data[0].to(device, non_blocking=True),
                          data[1].to(device, non_blocking=True))
        labels = torch.reshape(labels.float(), (-1, 1))
        ilabels = labels.data.clone()
        ilabels = ilabels.to("cpu").numpy().flatten().tolist()
        ilabels = [int(x) for x in ilabels]

        outputs = net(inputs)
        ipred = outputs.data.clone()
        ipred = ipred.to("cpu").numpy().flatten().tolist()

        # Loss
        loss = criterion(outputs, labels)
        iloss = loss.data.clone()
        iloss = iloss.to("cpu").numpy().tolist()

        return iloss, ipred, ilabels


def main():
    # Arguments
    argd = _arguments()

    # Dataset locations and properties
    vrdir = argd['vrdir']
    ndyads = argd['ndyads']
    best_ckpt = argd['best_ckpt']
    vlist = f"{argd['vlist']}"

    # Important architecture parameters
    input_shape_30fps = [3, 90, 224, 224]

    # GPU device. Defaulting to GPU 0
    device = get_cuda_device(0)

    # Loading network
    net = load_net(best_ckpt, ndyads, input_shape_30fps)

    # Loading data into validation loader
    aolme_trims_data_loader = AOLMEValTrmsDLoader(vrdir, vlist, oshape=(224, 224))
    loader = DataLoader(
        aolme_trims_data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    
    # Looping through each batch
    pred_prob_lst = []
    gt_lst = []
    for idx, data in enumerate(Bar(loader)):
        loss, pred, gt = eval_(device, net, data)
        gt_lst += gt
        pred_prob_lst += pred

    fpr, tpr, thresholds = metrics.roc_curve(gt_lst, pred_prob_lst)
    auc = metrics.auc(fpr, tpr)
    print(f"AUC: {auc}")

    # Calculating accuracy
    prob_th = 0.5
    pred_lst = [1*(x >= prob_th) for x in pred_prob_lst]
    acc = metrics.accuracy_score(gt_lst, pred_lst)
    print(f"Accuracy @ {prob_th} threshold: {acc}")

    #create ROC curve
    png_path =  os.path.dirname(best_ckpt)
    png_name = (
        os.path.splitext(os.path.basename(vlist))[0] +
        "_" +
        os.path.splitext(os.path.basename(best_ckpt))[0] +
        ".png"
        )    
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve (Acc = {acc} @ {prob_th} threshold)")
    plt.legend(loc="lower right")

    # Create histograms of 0 and 1 w.r.t. probability
    print(f"Writing {png_name}")
    plt.savefig(f"{png_path}/{png_name}", dpi=300)


    
# Execution starts here
if __name__ == "__main__":
    main()
