"""
Prints "best" epoch value.

Model producing highest validation accuracy is considered to be the best.

Ex:
python exp2_print_best_epoch.py /mnt/twotb/dyadic_nn/workdir/tynty/one_trim_per_instance_3sec_224/exp2_group_leave_one_out/C1L1P-A/dyad_2/val_log.json
"""
import argparse
import pdb
from aqua.nn.log_analyzer import LogAnalyzer

def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(description=("""
    Prints best validation accuracy epoch values.
        """))

    # Adding arguments
    args_inst.add_argument("val_log", type=str, help=("Validation log file path"))

    
    # Parse arguments
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {'val_log': args.val_log}

    # Return arguments as dictionary
    return args_dict

def main():
    """ Main function """
    argd = _arguments()

    val_log = argd['val_log']


    # Loading validation log
    val = LogAnalyzer(val_log)
    val_df = val.df
    val_df['acc_diff'] = abs(val_df['trnacc'] - val_df['acc'])

    # Best epoch
    val_acc_max = val_df['acc'].max()
    best_epoch = val_df[val_df["acc"] == val_df["acc"].max()].iloc[0]

    # Creating string to print
    print_str = (f"best epoch: {best_epoch['epoch']}\n"
                 f"trn acc   : {best_epoch['trnacc']}\n"
                 f"val acc   : {best_epoch['acc']}\n"
                 f"acc diff  : {best_epoch['acc_diff']}\n"
                 )
    print(print_str)
    

# Execution starts here
if __name__ == "__main__":
    main()
    

