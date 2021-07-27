"""
NOTE: This might not work as I changed LogAnalyzer class
Prints highest validation accuracy in log file
"""
import argparse
from aqua.nn.log_analyzer import LogAnalyzer


def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(description=("""
    Prints best validation accuracy epoch values.
        """))

    # Adding arguments
    args_inst.add_argument("loc", type=str, help=("Validation log file directory location"))
    args_inst.add_argument("name", type=str, help=("Name of validation file"))
    
    # Parse arguments
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {'loc': args.loc, 'name':args.name}

    # Return arguments as dictionary
    return args_dict


def main():
    """ Main function """
    argd = _arguments()
    log_loc = argd['loc']
    log_name = argd['name']

    log_analyzer = LogAnalyzer(log_loc)

    print(log_analyzer.get_highest_vacc_epoch(log_name))


# Execution starts here
if __name__ == "__main__":
    main()
