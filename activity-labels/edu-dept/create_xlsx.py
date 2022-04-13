"""
Description 
----------- 
The following code parses ground truth csv
file to xlsx sheets.  This is done to provide easy access to education
researchers.

Output
------
The output of this script is an xlsx file located in the same
directory as input csv file.

Example
-------
python create_xlsx.py \
   ~/Dropbox/typing-notyping/C1L1P-E/20170302/gTruth-tynty_30fps.csv \
   ~/Dropbox/typing-notyping/kid-pseudonym-mapping.csv \
   ~/Dropbox/typing-notyping/C1L1P-E/20170302/gt-ty-30fps.xlsx \
   typing person numeric_code

"""


import argparse
import os
from aqua.data_tools.aolme import AOLMEActivityLabels


def _arguments():
    """Parse input arguments."""
    try:
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

        args_inst.add_argument("person_code", type=str, help=("Column name having student identification"))
        args_inst.add_argument("person_code_type", type=str, help=("Type of student identification used, {numeric_code, student_code, pseudonym}"))
        args = args_inst.parse_args()

        # Crate a dictionary having arguments and their values
        args_dict = {
            'acty_csv': args.acty_csv,
            'names_csv': args.names_csv,
            'out_xlsx': args.out_xlsx,
            'activity': args.activity,
            'person_code': args.person_code,
            'person_code_type': args.person_code_type
        }

        # Return arguments as dictionary
        return True, args_dict
    
    except:
        return False, {}


# Execution starts here
if __name__ == "__main__":
    
    args_flag, argd = _arguments()
    
    if not args_flag:
        # Manually initializing arguments
        acty_csv         = "~/Dropbox/typing-notyping/C1L1P-E/20170302/gTruth-tynty_30fps.csv"
        names_csv        = "~/Dropbox/typing-notyping/kid-pseudonym-mapping.csv"
        out_xlsx         = "~/Dropbox/typing-notyping/C1L1P-E/20170302/gt-ty-30fps.xlsx"
        activity         = "typing"
        person_code      = "person"
        person_code_type = "numeric_code"
    else:
        # Parsing arguments from command line
        acty_csv         = argd['acty_csv']
        names_csv        = argd['names_csv']
        out_xlsx         = argd['out_xlsx']
        activity         = argd['activity']
        person_code      = argd['person_code']
        person_code_type = argd['person_code_type']
        
    # Extract root directory and labels file name from csv file name
    rdir = os.path.dirname(acty_csv)
    acty_csv_name = os.path.basename(acty_csv)

    act_labels = AOLMEActivityLabels(rdir, acty_csv_name)
    act_labels.create_xlsx(names_csv, out_xlsx, activity, person_code, person_code_type)
