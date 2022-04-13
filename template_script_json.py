"""
DESCRIPTION
-----------

USAGE
-----

EXAMPLE
-------
"""


import sys
import argparse
import keyboard

# Default configuration
DEF_CFG = "???"

def get_json_file_path():
    """ Jason file input """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description="""
        Description
        -----------
        <description>

        Example usage:
        --------------
        <example usage>
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args_inst.add_argument("cfg", type=str, help="Path to configuration file")

    if len(sys.argv) == 1:
        return ""

    args = args_inst.parse_args()
    return args.cfg


# Execution starts from here
if __name__ == "__main__":

    # Getting configuration path
    cfg_path = get_json_file_path()

    # Use default configuration if it is not provided
    if cfg_path == "":
        key_press = input(f"Enter c to continue using configuration\n{DEF_CFG}\n")
        if key_press == 'c':
            cfg_path = DEF_CFG
        else:
            sys.exit()

    # The script code starts from here
