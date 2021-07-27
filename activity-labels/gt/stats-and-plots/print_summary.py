"""
Prints activity label summary to standard console.

Example:
$ python print_summary.py ~/Dropbox/typing-notyping gTruth-tynty_30fps.csv
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

    # Parse arguments
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {'rdir': args.rdir, 'labels_fname': args.labels_fname}

    # Return arguments as dictionary
    return args_dict


# Execution starts here
if __name__ == "__main__":
    args = _arguments()

    # Creating ActivityLabels instance
    act_labels = AOLMEActivityLabels(args['rdir'], args['labels_fname'])
    act_labels.save_summary_to_json()
    pdb.set_trace()
