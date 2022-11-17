"""
Checks video files within a root directory and prints out file that
are not valid.
"""
import argparse
from aqua.data_tools.aolme import AOLMETrimmedVideos


def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(description=("""
    Checks video files within a root directory.
        """))

    # Adding arguments
    args_inst.add_argument("rdir",
                           type=str,
                           help=("Root directory having trimmed videos"))

    args_inst.add_argument("ext",
                           type=str,
                           help=("Video format. Ex: .mp4, .avi"))
    # Parse arguments
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {'rdir': args.rdir, 'ext': args.ext}

    # Return arguments as dictionary
    return args_dict


def main():
    """ Main function """
    argd = _arguments()

    # Trimmed videos object
    tv = AOLMETrimmedVideos(argd['rdir'], argd['ext'])

    # Check videos
    tv.check_videos()


# Execution starts here
if __name__ == "__main__":
    main()
