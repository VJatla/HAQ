"""

This file calculates peformace of algorithm when compared against ground truth.
It produce
    1. A CSV file that contains performance information
    2. An Activity map that shows performance
"""
import os
import argparse
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=("""
        Evaluates performance and viusalize it via activity maps.
        The input directory should have the following files,
            1. gt-ty-30fps.xlsx       : Excel file having typing instances from ground truth.
            2. alg-ty-30fps.xlsx      : Excel file having typing instances from algorithm.
            3. properties_session.csv : Session properties
            4. groups_jun28_2021.csv  : Groups database from AOLME website

        """),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args_inst.add_argument("idir", type=str, help="Input directory having necessary files")
    args_inst.add_argument("odir", type=str, help="Output directory")
    args_inst.add_argument("vdb", type=str, help="Video database")
    args_inst.add_argument("act", type=str, help="Activity name")
    args_inst.add_argument("dur", type=str, help="Activity duration.")
    args = args_inst.parse_args()

    args_dict = {
        'idir': args.idir,
        'odir': args.odir,
        'vdb': args.vdb,
        'act': args.act,
        'dur': args.dur
    }
    return args_dict


def load_and_process_gt(idir):
    """Load and process ground truth.
    """
    df = pd.read_excel(f"{idir}/gt-ty-30fps.xlsx")
    return df


def load_and_process_pred(idir):
    """Load and process typing predictions.
    """
    df = pd.read_excel(f"{idir}/alg-ty-30fps.xlsx")
    return df


def create_empty_dfs_dict(idir, stu_codes, dur):
    """Create an empty dataframes dictionary per person."""

    # Create a list having start and start times
    sdf = pd.read_csv(f"{idir}/properties_session.csv")

    # Create a list of video start and end times based on the
    # analysis duration.
    vname_list = []
    stime_list = []
    etime_list = []
    dtime_list = []
    for i, vname in enumerate(sdf['name'][0:len(sdf) - 1]):  # The -1 to avoid the last row having total duration

        vdur = sdf.iloc[i]['dur'].item()

        for stime in range(0, vdur, dur):
            vname_list += [vname]
            stime_list += [stime]
            etime_list += [min(stime + dur, vdur)]
            dtime_list += [min(stime + dur, vdur) - stime]


    # Get student codes
    stu_codes = stu_codes

    # Generate an empty dataframe per student
    empty_dfs_dict = {}
    for i, stu_code in enumerate(stu_codes):
        empty_dfs_dict[stu_code] = pd.DataFrame(
            {'vname': vname_list,
             'stime': stime_list,
             'etime': etime_list,
             'dtime': dtime_list}
        )

    return empty_dfs_dict

def get_act(df, row, act, dur):
    """Returns a activity label."""

    
    stime = row['stime']
    etime = row['etime']
    vname = row['vname']

    # Filtering the ground truth dataframe
    df_ = df[df['Video name'] == vname].copy()
    if len(df_) == 0:
        return f"no{act}"

    # Introducing time in seconds for easy comparison
    df_['stime'] = pd.to_timedelta(df_['Start time']).dt.total_seconds().astype(int)
    df_['etime'] = pd.to_timedelta(df_['End time']).dt.total_seconds().astype(int)

    
    # Filtering for activities that overlap with stime and etime
    df_ = df_[(df_['stime'] < etime) & (df_['etime'] > stime)]

    if len(df_) == 0:
        return f"no{act}"

    # Loop through the df_ and add the contribution of each row
    act_dur = 0  # Activity duration
    for i, r in df_.iterrows():

        r_stime = max(r['stime'], stime)  #  Activity start
        r_etime = min(r['etime'], etime)  #  Activity end
        act_dur += r_etime - r_stime

    # If the act_dur is >= 5 seconds label it as activity
    # or it is not activity
    if act_dur >= math.floor(dur/2):
        return act
    else:
        return f"no{act}"

def get_conf_matrix_label(gt, pred, act):
    """Determine confusion matrix label.
    """
    no_act = f"no{act}"
    
    if gt == no_act and pred == no_act:
        return "TN"
    elif gt == no_act and pred == act:
        return "FP"
    elif gt == act and pred == no_act:
        return "FN"
    elif gt == act and pred == act:
        return "TP"
    else:
        raise Exception(f"{act} is not supported.")
    

    
def compare_pred_vs_gt_per_student(perf_df, gt_df, pred_df, act, dur):
    """Compare prediction and ground truth per student.

    It updates the `perf_df` with new columns,
      1. gt_label
      2. pred_label
      3. conf_mat_label: Confusion matrix label, TP, TN, FP, FN
    """

    # Initializing lists that become new columns
    gt_label_list = []
    pred_label_list = []
    conf_mat_label_list = []

    # Loop over each performance dataframe row
    for i, conf_mat_row in perf_df.iterrows():
        
        gt_label = get_act(gt_df, conf_mat_row, act, dur)
        pred_label = get_act(pred_df, conf_mat_row, act, dur)
        conf_mat_label = get_conf_matrix_label(gt_label, pred_label, act)

        gt_label_list.append(gt_label)
        pred_label_list.append(pred_label)
        conf_mat_label_list.append(conf_mat_label)

    # Adding the columns to performance dataframe
    perf_df['gt_label'] = gt_label_list
    perf_df['pred_label'] = pred_label_list
    perf_df['conf_mat_label'] = conf_mat_label_list

    return perf_df

# Save to excel

def save_to_excel(perf_dict, excel_fpath):
    """Save to excel with each sheet corresponding to a student."""
    
    for i, stu_code in enumerate(perf_dict.keys()):
        
        stu_df = perf_dict[stu_code]

        # Write to excel sheet
        if i == 0:
            stu_df.to_excel(excel_fpath, sheet_name=stu_code, index=False)
        else:
            with pd.ExcelWriter(excel_fpath, mode='a', engine='openpyxl') as writer:
                stu_df.to_excel(writer, sheet_name=stu_code, index=False)

def load_from_excel(stu_codes, excel_fpath):
    """Load the performance excel sheet.

    Each sheet contains student_code.
    """
    perf_dict = {}

    for stu_code in stu_codes:
        perf_dict[stu_code] = pd.read_excel(excel_fpath, sheet_name=stu_code)

    return perf_dict

def get_sec_to_add(vname):
    """Due to the stime and etime being reset every video
    in a session I have to add previous session times
    """
    # Load session properties data frame
    sdf = pd.read_csv(f"{args['idir']}/properties_session.csv")

    # remove all the rows from current row with vname till
    # the last row from sdf
    current_idx = sdf[sdf['name'] == vname].index[0]

    # Remove all rows from current row to the last row
    sdf = sdf.loc[:current_idx - 1]

    if len(sdf) == 0:
        return 0
    else:
        return sum(sdf['dur'].tolist())

    
    
def plot_activity_map(perf_dict):
    """Plot the activity map."""
    
    # Define colors for each confusion matrix label
    cfm_colors = {"TP": "green", "TN":(0, 1, 0, 0.3), "FP":"orange", "FN":"red"}
    # Start the figure
    fig, ax = plt.subplots()
    
    # Loop over each stu_code
    for stu_idx, stu_code in enumerate(perf_dict.keys()):

        df = perf_dict[stu_code]

        for i, row in df.iterrows():
            
            sec_to_add = get_sec_to_add(row['vname'])
                
            ax.hlines(
                stu_idx + 1,
                sec_to_add + row['stime'],
                sec_to_add + row['etime'],
                linewidth=20,
                color=cfm_colors[row['conf_mat_label']]
            )
    plt.show()


# Execution starts from here
if __name__ == "__main__":

    # Initializing Activityperf with input and output directories
    args = _arguments()
    
    # Output excel file
    excel_fpath = f"{args['odir']}/alv_vs_gt_{args['dur']}sec.xlsx"

    # Load ground truth dataframe
    gt_df = load_and_process_gt(args['idir'])
    stu_codes = gt_df['Student code']

    # Create algorithm dataframe with each row representing `dur`
    pred_df = load_and_process_pred(args['idir'])

    # Skip to plotting if the output excel already exists
    if not os.path.isfile(excel_fpath):
        
        # Create an empty dataframe dictionary.
        perf_df_dict = create_empty_dfs_dict(args['idir'], stu_codes, int(args['dur']))

        # Loop over each student in the prediction dataframe and
        # mark each row as TP, TN, FP or FN
        for stu_idx, stu_code in enumerate(stu_codes):

            perf_df_stu = perf_df_dict[stu_code].copy()
            gt_df_stu = gt_df[gt_df['Student code'] == stu_code].copy()
            pred_df_stu = pred_df[pred_df['Student code'] == stu_code].copy()

            perf_df_stu = compare_pred_vs_gt_per_student(perf_df_stu.copy(), gt_df_stu, pred_df_stu, args['act'], int(args['dur']))

            perf_df_dict[stu_code] = perf_df_stu

        # Save the dictionary as excel sheet
        save_to_excel(perf_df_dict, excel_fpath)

    else:

        # Load performance excel to a dictionary
        perf_df_dict = load_from_excel(stu_codes, excel_fpath)

    # Plot
    plot_activity_map(perf_df_dict)
