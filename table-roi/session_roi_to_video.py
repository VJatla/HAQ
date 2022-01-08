"""
DESCTIPTION
-----------
Maps session level region of interests to video level.

USAGE
----- 
```sh
python session_to_video.py /home/vj/Dropbox/table_roi_annotation
```
"""

# Libraries
import argparse
import numpy as np
import pandas as pd
import pdb
import math
from tqdm import tqdm

# User library imports
import pytkit as pk


def _arguments():
    """Parses input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=("Maps session level region of interests to video level."),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Adding arguments
    args_inst.add_argument(
        "rdir",
        type=str,
        help='Directory having session level region of interest csv files.'
        )
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {'rdir': args.rdir}

    # Return arguments as dictionary
    # Hello world how are you doing
    return args_dict


def get_nframes_and_names(vlist):
    """
    Return a list having number of frames.
    """
    nfrms = []
    names = []
    for vpth in vlist:
        vo = pk.Vid(vpth, 'read')
        nfrms += [vo.props['num_frames']]
        names += [f"{vo.props['name']}.mp4"]
    return nfrms, names


def get_session_videos_info(scsv):
    """
    Extracts session information using `properties_session.csv`. It returns a
    dictionary with following keys,
    1. vpaths  : video paths
    2. nframes : Number of frames
    """
    # Reading properties_session.csv file
    dir_pth, fname, fext = pk.file_parts(scsv)

    # Get list of video paths in the session directory
    vlist = pk.get_file_paths_with_kws(
        dir_pth, ['.mp4'], no_kw_lst=['session']
        )
    vlist.sort()

    # Get number of frames in each video
    nframes, names = get_nframes_and_names(vlist)

    return {"vpaths":vlist, "nframes":nframes, "vnames": names}


def nframes_check(sinfo, scsv):
    """
    sfrms = number of frames in `session_video.mp4`
    vfrms = number of frames calculated by taking one frame every second from
            videos.
    sfrms should be equal to vfrms, if not throws error
    """
    dir_pth, fname, fext = pk.file_parts(scsv)

    # Geting sfrms
    svo = pk.Vid(f"{dir_pth}/session_video.mp4", 'read')
    sfrms = svo.props['num_frames']

    # Getting vfrms
    nfrms_lst = sinfo['nframes']
    vfrms = sum([math.floor(x/30) + 1 for x in nfrms_lst])

    # Throw error if they do not match
    if not vfrms == sfrms:
        print(
            f"Frames calculated from videos: {vfrms}\n"
            f"Frames calculsated from session: {sfrms}"
            )
        return False
    else:
        return True
    


# Execution starts here
if __name__ == "__main__":

    # Directory having session level roi csv files
    args = _arguments()
    rdir = args['rdir']

    # Loop through session level csv file paths
    session_csvs = pk.get_file_paths_with_kws(rdir, ['session_roi.csv'])

    for scsv in tqdm(session_csvs):
        print(f"Processing {scsv}")
        # Load the session_roi data as dataframe
        sdf = pd.read_csv(scsv)

        # Current session videos information
        sinfo = get_session_videos_info(scsv)

        # Checking if we have correct number of frames in `session_video.mp4`
        if not nframes_check(sinfo, scsv):
            raise Exception(f"ERROR: Frames mismatch")

        # Loop through each video
        f0 = []
        f  = [30]*len(sdf)
        video_names = []
        start_row_idx = 0
        for idx, vname in enumerate(sinfo['vnames']):
            
            # Calculate ending row index
            vnframes = sinfo['nframes'][idx]
            vnframes_extracted = math.floor(vnframes/30) + 1
            end_row_idx = start_row_idx + vnframes_extracted

            # Adding current video informaton to the list
            video_names += [vname]*vnframes_extracted
            f0 += np.arange(0, vnframes, 30).tolist()

            # update start row index
            start_row_idx = end_row_idx

        # Adding columns
        sdf['video_names'] = video_names
        sdf['f0'] = f0
        sdf['f'] = f

        # Write csv file
        dir_pth, _, _ = pk.file_parts(scsv)
        sdf.to_csv(f'{dir_pth}/video_roi.csv',index=False)
