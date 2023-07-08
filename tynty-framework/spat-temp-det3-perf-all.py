"""Calculating activity classification performance

NOTE:
-----
    Before using this please create xlsx files that have true class idexes using
    `tynty-framework/spat-temp-det3-perf.py`

USAGE:
------
python spat-temp-det3-perf-all.py ./cfg-perf-spat-temp-det3/C2L1P-B/20180223/C2L1P-B-20180223.json
"""


import argparse
import json
import pandas as pd
import os
from sklearn import metrics
from openpyxl import load_workbook
import matplotlib.pyplot as plt

import matplotlib


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size' : 11,
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def _arguments():
    """Parse input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=(
            """Calculating activity classification performance on full sessions for all algorithms."""),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args_inst.add_argument("cfg_path", type=str, help="Configuration file.")
    args = args_inst.parse_args()
    args_dict = {"cfg_path": args.cfg_path}
    
    return args_dict

def get_cf_mat_label_notyping(row, df):
    """
    Parameters
    ----------
    row : DataFrame row, DataSeries
        Algorithm data frame row
    df : DataFrame
         Ground truth data frame
    """
    
    # if in any of the check the dataframe becomes empty the,
    # if len(df) == 0
    # gt = notyping, alg = notyping, => True Negatibe

    # Filtering videos
    df = df[df['name'] == row['name']]
    if len(df) <= 0:
        return "TN"

    # Filter with pseudonym
    df = df[df['pseudonym'] == row['pseudonym']]
    if len(df) <= 0:
        return "TN"

    # Remove ground truth instances that end before current roi starts
    df = df[[not(x) for x in (df['f1'] < row['f0']).tolist()]]  
    if len(df) <= 0:
        return "TN"
    
    # Remove ground truth instances that start after current roi ends
    df = df[[not(x) for x in (df['f0'] > row['f1']).tolist()]]  
    if len(df) <= 0:
        return "TN"

    # If there are still ground truth instance left then we look
    # for frames overlap
    f0_overlap = [max(x, row['f0']) for x in df['f0'].tolist()]
    f1_overlap = [min(x, row['f1']) for x in df['f1'].tolist()]
    f_overlap  = sum([f1_overlap[i] - f0_overlap[i] for i in range(0, len(f0_overlap))]) + 1
    
    if f_overlap >= 45:
        # gt = typing, alg = notyping, => False Negatibe
        return "FN"
    else:
        # gt = notyping, alg = notyping, => True Negatibe
        return "TN"
    
    raise Exception(f"It should not execute till here!")

def get_cf_mat_label_typing(row, df):
    """
    Parameters
    ----------
    row : DataFrame row, DataSeries
        Algorithm data frame row
    df : DataFrame
         Ground truth data frame
    """
    
    # if in any of the check the dataframe becomes empty, i.e.,
    # if len(df) == 0
    # gt = notyping, alg = typing, => False Positive

    # Filtering videos
    df = df[df['name'] == row['name']]
    if len(df) <= 0:
        return "FP"

    # Filter with pseudonym
    df = df[df['pseudonym'] == row['pseudonym']]
    if len(df) <= 0:
        return "FP"

    # Remove ground truth instances that end before current roi starts
    df = df[[not(x) for x in (df['f1'] < row['f0']).tolist()]]  
    if len(df) <= 0:
        return "FP"
    
    # Remove ground truth instances that start after current roi ends
    df = df[[not(x) for x in (df['f0'] > row['f1']).tolist()]]  
    if len(df) <= 0:
        return "FP"

    # If there are still ground truth instance left then we look
    # for frames overlap
    f0_overlap = [max(x, row['f0']) for x in df['f0'].tolist()]
    f1_overlap = [min(x, row['f1']) for x in df['f1'].tolist()]
    f_overlap  = sum([f1_overlap[i] - f0_overlap[i] for i in range(0, len(f0_overlap))]) + 1
    
    if f_overlap >= 45:
        # gt = typing, alg = typing, => True Positive
        return "TP"
    else:
        # gt = notyping, alg = typing, => False Positive
        return "FP"
    
    raise Exception(f"It should not execute till here!")

    
def get_cf_mat_label(row, df):
    """
    Parameters
    ----------
    row : DataFrame row, DataSeries
        Algorithm data frame row
    df : DataFrame
        Ground truth data frame
    """

    # Calculate f1 column by adding f to f0 in ground truth
    # data frame
    df['f1'] = df['f0'] + df['f']

    # Based on the activity calculate the confusion matrix label
    if row['activity'] == "notyping":

        # Determining TN or FN
        cf_label = get_cf_mat_label_notyping(row, df)
        
    elif row['activity'] == "typing":

        # Determining TP or FP
        cf_label = get_cf_mat_label_typing(row, df)

    else:
        raise Exception(f"Unknown activity {row['activity']}")

    return cf_label

# Ececution starts here.
if __name__ == "__main__":

    # Load configuration JSON file as a dictionary
    cfg = _arguments()
    cfg_path = cfg['cfg_path']
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Loading ground truth instances as dataframe
    gtdf = pd.read_excel(cfg['gt_path'], sheet_name='Machine readable')

    # Loop over each algorithm
    plt.figure(figsize=(6, 4))
    for alg_idx, alg_path in enumerate(cfg['alg_paths']):
        alg_name = cfg['alg_names'][alg_idx]
        roc_color = cfg['roc_colors'][alg_idx]
        algdf = pd.read_excel(alg_path, sheet_name='all_instances')

        # AUC and ROC curve
        true_labels = algdf['class_idx_true'].tolist()
        pred_labels = algdf['class_idx'].tolist()
        pred_probs = algdf['class_prob'].tolist()
        
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_probs)
        auc = metrics.auc(fpr, tpr)
        acc = metrics.accuracy_score(true_labels, pred_labels)
        
        # Printing performance parameters
        print(
            f"""
            Algorithm {alg_name}:  AUC = {round(auc, 2)}, ACC = {round(acc, 2)}
            """)
        
        #create ROC curve
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color=roc_color,
            lw=lw,
            label=f"{alg_name}, AUC: {round(auc, 2)}",
        )

    # Axis and plot name settings
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    # Create histograms of 0 and 1 w.r.t. probability

    # Commented out to make sure we save a pgf file
    # png_name = cfg['out_roc_path']
    # plt.savefig(png_name, dpi=300)

    plt.tight_layout()
    pgf_name = cfg['out_roc_path_pgf']
    print(f"Saving PGF file to {pgf_name}")
    plt.savefig(pgf_name)

    
