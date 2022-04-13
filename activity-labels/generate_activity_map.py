"""
WARNING: DESCRIPTION
--------------------
This script generates activity map from activity labels. Activity 
labels are given using excel sheets.

USAGE
-----
python generate_activity_map.py <config json file>


EXAMPLE
-------
# Windows
python generate_activity_map.py C:\\Users\\vj\\Dropbox\\AOLME_Activity_Maps\\GT\\C1L1P-E\\20170302\\all_activities_win.json

# Linux
python generate_activity_map.py /home/vj/Dropbox/AOLME_Activity_Maps/GT/C1L1P-E/20170302/writing_lin.json

IDEAS
-----
1. Excel should have links to videos
2. On the click video link in plotly
"""
import pretty_errors
import argparse
import numpy as np
import pandas as pd
import json
import math
import os
import pdb
import plotly.graph_objects as go


def _arguments():
    """Parses input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=(
            """
            DESCRIPTION
            -----------
            This script generates activity map (map) from writing and
            typing activity labels. Activity labels are given using 
            excel sheets.
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
    student_code = cfg_dict["student_code"]
    return persons, pseudonyms, student_code
        


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

    # Activities of interest
    activities = cfg_dict["activities"].split(",")

    # total session duration
    tdur = get_session_dur(cfg_dict["session_properties"])

    # Get persons and pseudonyms
    persons, pseudonyms, student_code = get_persons(cfg_dict)

    # Adding data to session properties dictionary
    props["activities"] = activities
    props["tdur"] = tdur
    props["persons"] = persons
    props["pseudonyms"] = pseudonyms
    props["student_code"] = student_code

    return props

def get_df(cfg_dict, cluster_name, activity, clus_col):
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
    clus_col : Str
        Column name in excel sheet having activity ids

    Returns
    -------
    DataFrame
        A dataframe having acitivty and person under consideration
    """
    xlsx_pth = cfg_dict[f"{activity.strip()}_xlsx"]
    if clus_col == "cluster_id":
        df = pd.read_excel(xlsx_pth, sheet_name="BLC-HR")
    elif clus_col == "cluster_id_annotated":
        df = pd.read_excel(xlsx_pth, sheet_name="BLC-HR")
    elif clus_col == "Pseudonym":
        df = pd.read_excel(xlsx_pth, sheet_name="Human readable")
    elif clus_col == "Student code":
        df = pd.read_excel(xlsx_pth, sheet_name="Human readable")
    else:
        raise Exception(f"Unknown column {clus_col}")

    df = df[df[clus_col] == cluster_name].copy()
    
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


def get_vidx_from_groups_db(name, dbcsv):
    """
    Returns video index from groups database file.

    Parameters
    ----------
    name : str
        Video name
    dbcsv : str
        Path to csv file having groups data-base entries
    """
    # removing _30fps from name
    name = name.replace("_30fps","")
    df = pd.read_csv(dbcsv)
    vidx = df[df['video_name'] == name]['idx'].item()
    return vidx

    


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

    # Cluster loop (person or cluster_name) loop
    y_ticks = []
    for pidx in range(0,len(cfg_dict['Persons'])):
        if cfg_dict['cluster_col_name'] == "Pseudonym":
            cluster_name = sess_props['pseudonyms'][pidx]
            cluster_id = cluster_name
        elif cfg_dict['cluster_col_name'] == "cluster_id":
            cluster_name = f"clus_{pidx}"
            cluster_id = pidx
        elif cfg_dict['cluster_col_name'] == "cluster_id_annotated":
            cluster_name = sess_props['pseudonyms'][pidx]
            cluster_id = cluster_name
        elif cfg_dict['cluster_col_name'] == "Student code":
            cluster_name = sess_props['student_code'][pidx]
            cluster_id = cluster_name
        else:
            raise Exception (f"{cfg_dict['cluster_col_name']} is not supported")

        y_ticks += [cluster_name]

        # Activity loop
        y_offsets = cfg_dict['activity_offset_val']
        y_colors = cfg_dict['activity_colors']

        for aidx, activity in enumerate(sess_props['activities']):

            # Value of current activity based on person
            y_val = (pidx + 1) + y_offsets[aidx]

            # Color to use
            y_color = y_colors[aidx]

            # Load data frame corresponding to person and activity
            df = get_df(cfg_dict, cluster_id, activity, cfg_dict['cluster_col_name'])
            
            # Video loop
            show_legend = True
            uniq_videos = df['Video name'].unique()

            for vidx, video_name in enumerate(uniq_videos):
                vidx_db = get_vidx_from_groups_db(video_name, cfg_dict['groups_db'])
                dfv = df[df['Video name'] == video_name].copy()
                prev_dur = int(sdf[sdf['name'] == video_name].prev_dur.item())

                # Activity instance loop
                for inst_idx, act in dfv.iterrows():
                    # Initialize a list with np.nan with length equal to total
                    # number of seconds in session
                    y = [np.nan]*sess_props['tdur']
                    x = [t/60 for t in list(range(0, sess_props['tdur']))]

                    # Get start and stop of envents in seconds
                    start_sec = get_time_in_sec(act['Start time'])
                    end_sec = get_time_in_sec(act['End time'])
                    
                    start_idx = prev_dur + start_sec
                    stop_idx = prev_dur + end_sec

                    x_ann = (start_idx + ((stop_idx - start_idx)/2))/60
                    y_ann = y_val
                    
                    y[start_idx:stop_idx] = [y_val]*(
                        stop_idx - start_idx
                    )
                    import pdb; pdb.set_trace()
                    
                    yaxis_range = [0, max(y)+1]
                    legend_group = f"{activity}"
                    legend_name = f"{cluster_name}, {activity}"

                    # hover template
                    hover_tempalte = (
                        f"<b>Pseudonym:</b> {cluster_name}<br>"
                        f"<b>Activity:</b> {activity}<br>"
                        f"<b>Video:</b> {act['Video name']}<br>"
                        f"<b>Time:</b> {act['Start time']} to {act['End time']}"
                    )

                    # Add url parameters to the links
                    link = "https://aolme.unm.edu/researcher/activity_maps/video.php"
                    link = f"{link}?vidx={vidx_db}&start_time={start_sec}&end_time={end_sec}"
                    
                    # Show legend the first time per person.
                    fig.add_trace(
                        go.Scatter(x=x,
                                   y=y,
                                   hovertemplate=hover_tempalte,
                                   mode='lines+text',
                                   line=dict(
                                       color=y_color,
                                       width=20),
                                   legendgroup=legend_group,
                                   name=legend_name,
                                   showlegend=show_legend
                                   ))
                    fig.add_annotation(
                        x=x_ann, y=y_ann,
                        width=5,
                        text=f"<a href='{link}'>*</a>",
                        showarrow=True)
                    show_legend = False
                        
    

    # Font settings
    tick_font = 18
    hover_font = 18
    legend_font = 12
    title_font = 28
    axes_title_font = 24

    # Updating Y axis to be more useful
    try:
        yaxis_range = [0, max(y)+1]
    except:
        pdb.set_trace()
    fig.update_layout(yaxis_range=[0,len(y_ticks) + 1])
    fig.update_yaxes(
        title=cfg_dict['cluster_col_name'],
        title_font={"size": axes_stitle_font},
        tickvals=np.arange(1, len(y_ticks) + 1),
        ticktext=y_ticks,
        tickfont=dict(size=tick_font))
    
    # Updating X axis
    fig.update_xaxes(title_text="Time in minutes",
                     title_font={"size": axes_title_font},
                     tickfont=dict(size=tick_font))
    
    # Hover text font
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=hover_font,
            font_family="Time New Roman"))

    # Legend fonts
    # fig.update_layout(legend_title="Activity per person") # <-- Removed to save space
    fig.update_layout(
        legend = dict(font = dict(size = legend_font)),
        legend_title = dict(font = dict(size = legend_font + 4))
    )

    # Title <--- Commented out to save space
    # fig.update_layout(
    #     title=f"{cfg_dict['Title']}",
    #     font_size=title_font,
    # )

    fig.show()
    fig.write_html(cfg_dict['save_loc'])
    
    
# Execution starts here
if __name__ == "__main__":
    main()
