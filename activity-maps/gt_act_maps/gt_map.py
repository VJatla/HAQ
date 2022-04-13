import argparse
from aqua.act_maps import ActMapsPlotly

def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description="""
        Visualizes and saves activity ground truth activity maps.

        Example usage:
        --------------
        python gt_map.py ~/Dropbox/AOLME_Activity_Maps/typing_writing/C1L1P-C/20170413/cfg.json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args_inst.add_argument("cfg", type=str, help="Path to configuration file")

    args = args_inst.parse_args()

    args_dict = {
        'cfg': args.cfg,
    }
    return args_dict


# Execution starts from here
if __name__ == "__main__":

    # Initializing Activityperf with input and output directories
    args = _arguments()
    ty_perf = ActMapsPlotly(args['cfg'])

    # Write the map produced by confusion matrix
    ty_perf.write_activity_map("gt")
