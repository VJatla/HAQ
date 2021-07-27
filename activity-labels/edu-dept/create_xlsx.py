"""
Description
-----------
    The following code parses ground truth csv file to xlsx sheets.
    This is done to provide easy access to education researchers.

Output
------
    The output of this script is an xlsx file located in the same
    directory as input csv file.

Example
-------
    python create_xlsx.py ~/Dropbox/typing-notyping/C1L1P-E/20170302/gTruth-tynty_30fps.csv ~/Dropbox/typing-notyping/kid-pseudonym-mapping.csv ~/Dropbox/typing-notyping/C1L1P-E/20170302/typing_instances.xlsx typing
"""


import argparse
import os
from aqua.data_tools.aolme import AOLMEActivityLabels


def _arguments():
    """Parses input arguments."""

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=(
            """The following code parses ground truth csv file to xlsx sheets.
            This is done to provide easy access to education researchers.
            """))

    # Adding arguments
    args_inst.add_argument("acty_csv", type=str, help=(
        "CSV file having activity instance"))
    args_inst.add_argument("names_csv", type=str,
                           help=("CSV file having names."))
    args_inst.add_argument("out_xlsx", type=str, help=("XLSX file path"))
    args_inst.add_argument("activity", type=str, help=("Activity that is being processed."))
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {'acty_csv': args.acty_csv, 'names_csv': args.names_csv,
                 'out_xlsx': args.out_xlsx, 'activity':args.activity}

    # Return arguments as dictionary
    # Hello world how are you doing
    return args_dict


def main():
    """Main function."""
    argd = _arguments()
    acty_csv = argd['acty_csv']
    names_csv = argd['names_csv']
    out_xlsx = argd['out_xlsx']
    activity = argd['activity']

    # Extract root directory and labels file name from csv file name
    rdir = os.path.dirname(acty_csv)
    acty_csv_name = os.path.basename(acty_csv)

    act_labels = AOLMEActivityLabels(rdir, acty_csv_name)
    act_labels.create_xlsx(names_csv, out_xlsx, activity)

# Execution starts here
if __name__ == "__main__":
    main()
