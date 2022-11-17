"""






DESCRIPTION
-----------
This script compares ground truth and algorithm using activity maps 
for a single activity.
















USAGE
-----
python gt_vs_alg_one_activity.py <config json file>


EXAMPLE
-------
# Windows
python gt_vs_alg_one_activity.py C:\\Users\\vj\\Dropbox\\AOLME_Activity_Maps\\GT_vs_Alg\\C1L1P-E\\20170302\\typing_gt_vs_alg_win.json

# Linux
python gt_vs_alg_one_activity.py /home/vj/Dropbox/AOLME_Activity_Maps/GT_vs_Alg/C1L1P-E/20170302/typing_gt_vs_alg_lin.json
"""
import argparse
import numpy as np
import pandas as pd
import json
import math
import os
import pdb
import pretty_errors

import plotly.graph_objects as go


def _arguments():
    """Parses input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=(
            """
            DESCRIPTION
            -----------
            This script compares ground truth and algorithm using 
            activity maps for a single activity.
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Adding arguments
    args_inst.add_argument(
        "cfg",
        type=str,
        help=("Configuration as JSON file")
    )
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {
        'cfg': args.cfg,
    }

    # Return arguments as dictionary
    # Hello world how are you doing
    return args_dict

def check_file(pth):
    """ Checks if a file exists

    Parameters
    ----------
    pth : Str
            Path to the file

    Returns
    -------
    bool
        Returns `True` if the file exists else `False`.
    """
    if not os.path.exists(pth):
        return False
    else:
        return True

def get_session_dur(sess_prop_path):
    """ Returns total duration of a session in seconds

    Parameters
    ----------
    sess_prop_path : Str
        File having session properties

    Returns
    -------
    int 
        Duration in seconds
    """
    df = pd.read_excel(sess_prop_path)
    tdur = df[df['name'] == "total"].dur.item()
    return tdur

def get_persons(cfg_dict):
    """ Returns total duration of a session in seconds

    Parameters
    ----------
    cfg_dict : Dictionary
        Configuration dictionary

    Returns
    -------
    List
        List of persons
    """
    persons = cfg_dict["Persons"]
    pseudonyms = cfg_dict["Pseudonyms"]
    return persons, pseudonyms
        


def get_session_properties(cfg_dict):
    """
    Returns a dictionary having important session properties.

    Parameters
    ----------
    cfg_dict : Dictionary
    
    Returns
    -------
    Dict
        Dictionary having important session properties
    """

    # Initializing properties dictionary
    props = {}

    # total session duration
    tdur = get_session_dur(cfg_dict["session_properties"])

    # Adding data to session properties dictionary
    props["activity"] = cfg_dict['activity']
    props["tdur"] = tdur

    return props

def get_df(cfg_dict, person, activity):
    """ Returns a DataFrame corresponding to a particular person
    and activity.

    Parameters
    ----------
    cfg_dict : Dict
        Configuration dictionary
    person : Str
        Person under consideration
    activity : Str
        Activity under consideration

    Returns
    -------
    DataFrame
        A dataframe having acitivty and person under consideration
    """
    xlsx_pth = cfg_dict[f"{activity.strip()}_xlsx"]
    df = pd.read_excel(xlsx_pth, sheet_name="Human readable")
    df = df[df["Numeric code"] == person].copy()
    return df


def get_time_in_sec(t):
    """
    Convert HH:MM:SS time to Seconds

    Parameters
    ----------
    t : String
        String having time in HH:MM:SS format
    """
    hh, mm, ss = [int(x) for x in t.split(":")]
    return 3600*hh + 60*mm + ss


def trace_activity(fig, df, sdf, sess_props, y_dict):
    """
    Trace activity for a session.

    Parameters
    ----------
    
    """
    
    # Initializing x with minutes and y with nans
    y = [np.nan]*sess_props['tdur']
    x = [t/60 for t in list(range(0, sess_props['tdur']))]

    # Video loop
    uniq_videos = df['Video name'].unique()
    for vidx, video_name in enumerate(uniq_videos):

        # A dataframe having activity instances for a video
        dfv = df[df['Video name'] == video_name].copy()
        prev_dur = int(sdf[sdf['name'] == video_name].prev_dur.item())

        # Activity instance loop
        for inst_idx, act in dfv.iterrows():
            
            start_sec = get_time_in_sec(act['Start time'])
            end_sec = get_time_in_sec(act['End time'])
            start_idx = prev_dur + start_sec
            stop_idx = prev_dur + end_sec

            y[start_idx:stop_idx] = [y_dict['value']]*(stop_idx - start_idx)

            legend_group = f"{y_dict['label']}"
            legend_name = f"{y_dict['label']}"

    # Tracing
    fig.add_trace(
        go.Scatter(x=x,
                   y=y,
                   mode='lines+text',
                   line=dict(
                       color=y_dict['color'],
                       width=50),
                   legendgroup=legend_group,
                   name=legend_name
                   ))

    return fig
            
        

    
def main():
    # Arguments
    argd = _arguments()
    cfg_path = argd["cfg"]

    # Check and load configuration file
    if not check_file(cfg_path):
        raise Exception(f"INFO: \n\t{cfg_path} does not exist")
    with open(cfg_path) as cfg_file:
        cfg_dict = json.load(cfg_file)

    # Creating session properties dictionary
    sess_props = get_session_properties(cfg_dict)
    sdf = pd.read_excel(cfg_dict["session_properties"])

    # Initalize plotly figure
    fig = go.Figure()

    # Load labels into data frames
    gtdf = pd.read_excel(cfg_dict['gt_xlsx'],
                         sheet_name="Human readable")
    audf = pd.read_excel(cfg_dict['auto_xlsx'],
                         sheet_name="Human readable")

    # Initializing x array
    x = [t/60 for t in list(range(0, sess_props['tdur']))]
    
    # Tracing ground truth
    y_dict = {
        "color" : "green",
        "value" : 1,
        "label" : "Ground Truth"
    }
    fig = trace_activity(fig, gtdf, sdf, sess_props, y_dict)

    # Tracing algorithm
    y_dict = {
        "color" : "blue",
        "value" : 2,
        "label" : "Algorithm"
    }
    fig = trace_activity(fig, audf, sdf, sess_props, y_dict)



    # Font settings
    tick_font = 18
    hover_font = 18
    legend_font = 12
    title_font = 28
    axes_title_font = 24
    
    # Y axis
    y_ticks = ["GT", "Alg"]
    fig.update_layout(yaxis_range=[0,len(y_ticks) + 1])
    fig.update_yaxes(
        title="GT Vs Algorithm",
        title_font={"size": axes_title_font},
        tickvals=np.arange(1, len(y_ticks) + 1),
        ticktext=y_ticks,
        tickfont=dict(size=tick_font))

    # Updating X axis
    fig.update_xaxes(title_text="Time in minutes",
                     title_font={"size": axes_title_font},
                     tickfont=dict(size=tick_font))

    # Legend fonts
    fig.update_layout(
        legend = dict(font = dict(size = legend_font)),
        legend_title = dict(font = dict(size = legend_font + 4))
    )


    fig.update_layout(
        title=f"{cfg_dict['Title']}",
        font_size=title_font,
    )

    # Show and writte
    fig.show()
    fig.write_html(cfg_dict['save_loc'])
    
# Execution starts here
if __name__ == "__main__":
    main()
