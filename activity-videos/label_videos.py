import argparse
from aqua.video_labeler.cv2_labeler import CV2Labeler


def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description="""
        Creates videos with activity labels marked using bounding boxes.

        Example usage:
        --------------
        python label_videos.py ~/Dropbox/AOLME_Activity_videos/C1L1P-C/20170413/gt/cfg.json
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
    cv2_labeler = CV2Labeler(args['cfg'])

    # Writing labels to videos
    cv2_labeler.write_labels_to_video()
