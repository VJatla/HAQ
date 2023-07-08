"""Summarizes activity instances at session level. It outputs two files.
1. A JSON file containing activity insances properties.
2. A CSV file, gt_summary_per_session.csv, having the following columns,
    - group
    - date
    - <act>_dur_labeled
    - num_<act>_inst_labeled
    - no<act>_dur_labeled
    - num_no<act>_inst_labeled
    - session_fully_labeled

Example:
python activity_instances_per_session_summary.py ~/Dropbox/typing-notyping gTruth-tynty_30fps.csv typing
python activity_instances_per_session_summary.py ~/Dropbox/writing-nowriting gTruth-wnw_30fps.csv writing

"""
import pdb
import argparse
from aqua.data_tools.aolme import AOLMEActivityLabels

def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(description=(
        "Prints activity label summary to standard console"))

    # Adding arguments
    args_inst.add_argument("rdir",
                           type=str,
                           help=("root directory having activity labels"))
    args_inst.add_argument("labels_fname",
                           type=str,
                           help=("Activity labels file name (csv)"))
    args_inst.add_argument("act_name",
                           type=str,
                           help=("Activity name, in my case it is typing or writing."))

    # Parse arguments
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {'rdir': args.rdir, 'labels_fname': args.labels_fname, 'act_name': args.act_name}

    # Return arguments as dictionary
    return args_dict


# Execution starts here
if __name__ == "__main__":
    args = _arguments()

    # Summarizing ground truth properties to JSON file
    act_labels = AOLMEActivityLabels(args['rdir'], args['labels_fname'])
    act_labels.save_summary_to_json()

    # Summarizing Ground truth at session level to gt_summary_per_session.csv
    act_labels.save_summary_per_session(
        args['act_name'],
        "/home/vj/Dropbox/typing-notyping/trn-val-tst-splits-backup-20220820.csv"
    )
