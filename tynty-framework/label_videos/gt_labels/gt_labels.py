import argparse
from aqua.video_labeler import FFMPEGLabeler


def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description="""
        Visualizes and saves activity ground truth activity maps.

        Example usage:
        --------------
        python gt_map.py ~/Dropbox/typing-notyping/C1L1P-C/20170413/gt-ty-30fps.xlsx \
                         ~/Dropbox/typing-notyping/C1L1P-C/20170413/properties_session.csv \
                         typing
                         gt_lables
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args_inst.add_argument("activity_instances", type=str, help="Path to ground truth excel file")
    args_inst.add_argument(
        "session_properties", type=str, help="Path to file having current session properties"
    )
    args_inst.add_argument("act", type=str, help="Activity name")
    args_inst.add_argument(
        "name_post_fix", type=str, help="Name that should be postfixed to output video."
    )

    args = args_inst.parse_args()

    args_dict = {
        'activity_instances': args.activity_instances,
        'session_properties': args.session_properties,
        'act': args.act,
        'name_post_fix': args.name_post_fix
        
    }
    return args_dict


# Execution starts from here
if __name__ == "__main__":

    # Initializing Activityperf with input and output directories
    args = _arguments()
    ty_labels = FFMPEGLabeler(**args)

    # Write the map produced by confusion matrix
    ty_labels.write_labels_to_video()

    # Prints confusion matrix per person to a csv file
    # output_pth = f"{args['odir']}/cf_mat_per_person.csv"
    # ty_perf.save_cf_mat_per_person(output_pth)
