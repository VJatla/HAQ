"""
DESCRIPTION
-----------

USAGE
-----

EXAMPLE
-------
"""
import argparse
import pdb
import pandas as pd

# User defined libraries
import pytkit as pk

def _arguments():
    """Parses input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=(
            """
            DESCRIPTION
            -----------
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Adding arguments
    args_inst.add_argument(
        "<arg1>",
        type=str,
        help=(
            """
            <HELP>
            """)
    )
    args_inst.add_argument(
        "<arg2>",
        type=str,
        help=("""<HELP>"""))
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {
        'arg1' : args.<arg1>,
        'arg2' : args.<arg2>
    }

    # Return arguments as dictionary
    # Hello world how are you doing
    return args_dict


# Calling as script
if __name__ == "__main__":
    # Input argumetns
    args = _arguments()
    print(args)
