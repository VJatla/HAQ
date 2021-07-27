"""
Trimmed videos are resized to a different resolution

Example:
```bash
python resize_trimmed_videos.py 128 \
/mnt/twotb/aolme_datasets/tynty/trimmed_videos/full_trims/typing \
.mp4 \
/mnt/twotb/aolme_datasets/tynty/trimmed_videos/full_trims_resized_128/typing
```
"""
import pdb
import argparse
from aqua.data_tools.aolme import AOLMETrimmedVideos


def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(description=("""
    Trimmed videos are resized without losing the aspect ratio.
        """))

    # Adding arguments
    args_inst.add_argument("vsize",
                           type=int,
                           help=("Trimmed video long edge size"))
    args_inst.add_argument("rdir",
                           type=str,
                           help=("Trimmed videos root directory"))
    args_inst.add_argument("ext", type=str, help=("Trimmed videos extension"))
    args_inst.add_argument("odir", type=str, help=("Output directory"))
    # Parse arguments
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {
        'vsize': args.vsize,
        'rdir': args.rdir,
        'ext': args.ext,
        'odir': args.odir
    }

    # Return arguments as dictionary
    return args_dict


def main():
    """ Main function """
    argd = _arguments()

    # Initializing trimmed video intance
    tv = AOLMETrimmedVideos(argd['rdir'], argd['ext'])

    # resize trimmed videos to a new directory
    tv.resize(argd['vsize'], argd['odir'])


# Execution starts here
if __name__ == "__main__":
    main()
