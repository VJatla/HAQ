"""Writing and nowriting ROC/AUC for SOTA methods

NOTE:
----
    This script requires that we use python environment that has
    MMACTION2 installed.

Example
-------
# I3D Validation
python calc_plot_ROC_AUC_30fps.py \
    ~/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/i3d/trn_i3d_r50_video_32x2x1_100e_kinetics400_rgb_vj.py \
    /mnt/twelvetb/vj/mmaction2_2022/workdir/wnw_table_roi/resized_224_30fps/i3d/run0_Dec26_2022/best_top1_acc_epoch_40.pth \
    /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps/val_videos_all.txt

# I3d Testing
python calc_plot_ROC_AUC_30fps.py \
    ~/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/sota/2022-mmaction2/wnw-roi-224/i3d/trn_i3d_r50_video_32x2x1_100e_kinetics400_rgb_vj.py \
    /mnt/twelvetb/vj/mmaction2_2022/workdir/wnw_table_roi/resized_224_30fps/i3d/run0_Dec26_2022/best_top1_acc_epoch_40.pth \
    /mnt/twotb/aolme_datasets/wnw_table_roi/resized_224_30fps/tst_videos_all.txt
      
"""


import argparse
import json
import pandas as pd
import os
import sys
import pytkit as pk
from tqdm import tqdm
from mmaction.apis import inference_recognizer, init_recognizer
from sklearn import metrics
import matplotlib.pyplot as plt


def _arguments():
    """Parse input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=(
            """Calculates and plots ROC and AUC curve for State Of The Art
            methods."""),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args_inst.add_argument("cfg_path", type=str, help="Python configuration file having model.")
    args_inst.add_argument("ckpt", type=str, help="Best check point.")
    args_inst.add_argument("vlist", type=str, help="Video list as text file.")
    args = args_inst.parse_args()

    return {
        "cfg_path": args.cfg_path,
        "ckpt": args.ckpt,
        "vlist": args.vlist,
    }


def get_video_list_dataframe(vlist):
    """ Returns a dataframe with the following columns,
    1. vloc
    2. gt_label
    """
    vfile_dir_loc = os.path.dirname(vlist)
    vfile = open(vlist, 'r')

    vloc_list = []
    gt_labels = []
    while True:

        line = vfile.readline()

        if len(line.split(" ")) >= 2:
            vloc_rel = line.split(" ")[0]
            vloc = f"{vfile_dir_loc}/{vloc_rel}"
            vloc_list += [vloc]

            gt_label = int(line.split(" ")[1])
            gt_labels += [gt_label]
        

        if not line:
            break

    return vloc_list, gt_labels
        
    

# Ececution starts here.
if __name__ == "__main__":


    # Input arguments
    args = _arguments()
    cfg = args['cfg_path']
    ckpt = args['ckpt']
    vlist = args['vlist']
    

    # Initialize recognizer network
    net = init_recognizer(cfg, ckpt, device='cuda:0')

    # Get video list and ground truth list
    vlocs, gt_lst = get_video_list_dataframe(vlist)

    # Loop through each video for prediction probability
    pred_prob_lst = []
    for i, vloc in enumerate(vlocs):
        print(i)
        inf = inference_recognizer(net, vloc)
        inf = sorted(inf, key=lambda x: x[0])
        pred_prob_lst += [inf[1][1]]

    fpr, tpr, thresholds = metrics.roc_curve(gt_lst, pred_prob_lst)
    auc = metrics.auc(fpr, tpr)
    print(f"AUC: {auc}")

    # Calculating accuracy
    prob_th = 0.5
    pred_lst = [1*(x >= prob_th) for x in pred_prob_lst]
    acc = metrics.accuracy_score(gt_lst, pred_lst)
    print(f"Accuracy @ {prob_th} threshold: {acc}")

    #create ROC curve
    png_path =  os.path.dirname(ckpt)
    png_name = (
        os.path.splitext(os.path.basename(vlist))[0] +
        "_" +
        os.path.splitext(os.path.basename(ckpt))[0] +
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
