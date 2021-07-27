""" 
Standardizes activity labels csv files to match ground truth so that we can
easily compare and reuse tools from `aqua.data_tools.aolme`.

NOTE:
    1. Assumes existance of `properties_session.csv` file with session properties.

Example:
    python standardize-act-labels.py ~/Dropbox/typing-notyping/C1L1P-E/20170302 ~/Dropbox/typing-notyping/C1L1P-E/20170302/alg-tynty_30fps.csv "ty_using_alg"

"""
import argparse
import pdb
import pandas as pd
import os
from aqua.fd_ops import get_file_paths_with_kws

def _arguments():
    """Parses input arguments."""

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=(
            """ Standardizes activity labels csv files to match 
            ground truth so that we can
            easily compare and reuse tools from `aqua.data_tools.aolme`.
            """))

    # Adding arguments
    args_inst.add_argument("rdir", type=str, help=("Session directory"))
    args_inst.add_argument("ocsv", type=str, help=("Output CSV file"))
    args_inst.add_argument("suffix", type=str, help=(
        "Suffix of csv file name. Name of csv file is assumed to be <video name>_<suffix>.csv"))
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {'rdir': args.rdir,
                 'ocsv': args.ocsv,
                 'suffix':args.suffix}

    # Return arguments as dictionary
    # Hello world how are you doing 
    return args_dict


def main():
    """Main function."""
    argd = _arguments()
    rdir = argd['rdir']
    suffix = argd['suffix']
    ocsv = argd['ocsv']

    # Dataframe that will contain all the video information
    session_df = pd.DataFrame()
    
    # Load CSV files are to be processed
    csv_files = get_file_paths_with_kws(rdir, [f"_{suffix}.csv"])

    # Load session properties file.
    sdf = pd.read_csv(f"{rdir}/properties_session.csv")

    # Loop through each csv file
    for csv_idx, ccsv in enumerate(csv_files):

        # Getting name of video from name of csv
        csv_name = os.path.basename(ccsv)
        vid_name = csv_name.replace(f"_{suffix}.csv", "")

        # Getting video duration from session properties
        vid_dur = sdf[sdf['name'] == vid_name]['dur'].item()
        
        # Read current csv
        cdf = pd.read_csv(ccsv)

        # (+ "activity" column) and (- "typing" column)
        cdf_only_typing = cdf[cdf["typing"] == 1].copy()
        cdf_only_typing["activity"] = "typing"
        del cdf_only_typing['typing']

        # Adding person column(For now everyone is labeled "Kidx")
        cdf_only_typing['person'] = "Kidx"

        # Adding video name
        cdf_only_typing['name'] = vid_name + ".mp4"

        # Adding "T" video duration
        cdf_only_typing['T'] = vid_dur

        if len(cdf_only_typing) > 0:
            if csv_idx > 0:
                session_df = pd.concat([session_df, cdf_only_typing])
            else:
                session_df = cdf_only_typing
    session_df.to_csv(ocsv)

# Execution starts here
if __name__ == "__main__":
    main()
