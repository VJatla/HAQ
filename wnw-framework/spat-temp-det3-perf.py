"""

NOTE: ONLY USE THIS FOR SOTA. NOT 3D-CNN

Calculating activity classification performance. This creates

    1. A txt file having confusion matrix labels
    2. A xlsx file having the same lables per activity instance
    3. A png image having ROC curve for curent algorithm
    4. Performance activity map


USAGE:
------
# C2L1P-B, Feb 23
python spat-temp-det3-perf.py \
    ~/Dropbox/writing-nowriting/C2L1P-B/20180223/wnw-roi-ours-3DCNN_kbdet_30fps.csv \
    ~/Dropbox/writing-nowriting/C2L1P-B/20180223/gt-wr-30fps.xlsx

# C3L1P-D, Feb 21
python spat-temp-det3-perf.py \
    ~/Dropbox/writing-nowriting/C3L1P-D/20190221/wnw-roi-ours-3DCNN_kbdet_30fps.csv  \
    ~/Dropbox/writing-nowriting/C3L1P-D/20190221/gt-wr-30fps.xlsx
"""


import argparse
import json
import pandas as pd
import os
from sklearn import metrics
from openpyxl import load_workbook
import matplotlib.pyplot as plt

def _arguments():
    """Parse input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=(
            """
            NOTE: ONLY USE THIS FOR SOTA. NOT 3D-CNN
            Calculating activity classification performance on full sessions.
            """),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args_inst.add_argument("alg_csv", type=str, help="Algorithm writing/nowriting classification")
    args_inst.add_argument("gt_xlsx", type=str, help="Ground truth writing instances as xlsx file.")
    args = args_inst.parse_args()
    args_dict = {"alg_csv": args.alg_csv, "gt_xlsx": args.gt_xlsx}
    
    return args_dict

def get_cf_mat_label_nowriting(row, df):
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
    # gt = nowriting, alg = nowriting, => True Negatibe

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
        # gt = writing, alg = nowriting, => False Negatibe
        return "FN"
    else:
        # gt = nowriting, alg = nowriting, => True Negatibe
        return "TN"
    
    raise Exception(f"It should not execute till here!")

def get_cf_mat_label_writing(row, df):
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
    # gt = nowriting, alg = writing, => False Positive

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
        # gt = writing, alg = writing, => True Positive
        return "TP"
    else:
        # gt = nowriting, alg = writing, => False Positive
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
    if row['activity'] == "nowriting":

        # Determining TN or FN
        cf_label = get_cf_mat_label_nowriting(row, df)
        
    elif row['activity'] == "writing":

        # Determining TP or FP
        cf_label = get_cf_mat_label_writing(row, df)

    else:
        raise Exception(f"Unknown activity {row['activity']}")

    return cf_label

# Get prediction probabilities
def get_pred_probs(algdf):
    """Returns a list of prediction probabilities for label 1"""
    labels = algdf['class_idx'].tolist()
    probs = algdf['class_prob'].tolist()

    for i in range(0, len(labels)):
        if labels[i] == 0:
            probs[i] = 1 - probs[i]

        probs[i] = round(probs[i], 2)

    return probs

# Ececution starts here.
if __name__ == "__main__":

    # Load configuration JSON file as a dictionary
    cfg = _arguments()

    # Loading into necessary dataframes
    algdf = pd.read_csv(cfg['alg_csv'])
    gtdf = pd.read_excel(cfg['gt_xlsx'], sheet_name='Machine readable')

    # Loop over each instance in algorithm data frame
    cfmat_label = []
    class_idx_true = []
    for i, row in algdf.iterrows():

        # Get the confusion matrix label
        cfmat_label_ = get_cf_mat_label(row, gtdf.copy())
        cfmat_label += [cfmat_label_]

        # based on cfmat_label determine class_idx_true
        if cfmat_label_ == "TN" or cfmat_label_ == "FP":
            class_idx_true += [0]
        else:
            class_idx_true += [1]
        

    # Write the new dataframe to an excel file
    oxlsx_name = os.path.splitext(os.path.basename(cfg['alg_csv']))[0] + ".xlsx"
    ofpth = f"{os.path.dirname(cfg['alg_csv'])}/{oxlsx_name}"
    algdf['cm_label'] = cfmat_label
    algdf['class_idx_true'] = class_idx_true
    algdf.to_excel(ofpth, sheet_name="all_instances", index=False)

    # Confusion matrix
    ntp = sum([1*(x == "TP") for x in cfmat_label])
    nfp = sum([1*(x == "FP") for x in cfmat_label])
    ntn = sum([1*(x == "TN") for x in cfmat_label])
    nfn = sum([1*(x == "FN") for x in cfmat_label])
    print(f"TP = {ntp}, FP = {nfp}, TN = {ntn}, FN = {nfn}")

    # Accuracy
    acc = (ntp + ntn)/(ntp + ntn + nfp + nfn)

    # AUC and ROC curve
    true_labels = algdf['class_idx_true'].tolist()
    pred_probs = algdf['class_prob'].tolist()
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_probs)
    auc = metrics.auc(fpr, tpr)

    # Create another dataframe with performance metrics
    columns = ['Metric', 'Value']
    rows = [['TP', ntp], ['FP', nfp], ['FN', nfn], ['TN', ntn],
            ['AUC', auc]]
    perf_df = pd.DataFrame(rows, columns=columns)
    txt_name = os.path.splitext(cfg['alg_csv'])[0] + ".txt"
    with open(txt_name, 'w') as f:
        f.write(f"\nTP = {ntp}, \nFP = {nfp}, \nTN = {ntn}, \nFN = {nfn} \nAcc={acc}")
        

    #create ROC curve
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
    plt.title(f"{os.path.splitext(os.path.basename(cfg['alg_csv']))[0]}")
    plt.legend(loc="lower right")

    # Create histograms of 0 and 1 w.r.t. probability
    png_name = os.path.splitext(cfg['alg_csv'])[0] + ".png"
    plt.savefig(png_name, dpi=300)
    
