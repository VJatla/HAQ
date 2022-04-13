""" This is description. """


import argparse
import numpy as np
import pandas as pd


def _arguments():
    """Parse input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=("???"), formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args_inst.add_argument("arg1", type=str, help="help")
    args_inst.add_argument("arg2", type=str, help="help")
    args = args_inst.parse_args()

    args_dict = {"arg1": args.arg1, "arg2": args.arg2}
    return args_dict


# Execution starts from here
if __name__ == "__main__":
    args = _arguments()
    print(args)
