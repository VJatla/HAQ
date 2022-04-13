"""
Creates a video at same frame rate as original with following information
overlayed,
1. Ground truth bbox
2. Algorithm bbox
3. IoU
4. Confusion matrix labels

Before using this make sure to run `write_spatio_tempo_perf_tocsv.py` script to
create a csv file necessary for running this script.

Example
-------
Linux:
    python write_spatio_tempo_perf_tovideo.py ~/Dropbox/typing-notyping/C1L1P-E/20170302/perf-tynty_30fps.csv

    python write_spatio_tempo_perf_tovideo.py ~/Dropbox/typing-notyping/C1L1P-C/20170413/perf-tynty_30fps.csv

    python write_spatio_tempo_perf_tovideo.py ~/Dropbox/typing-notyping/C1L1P-C/20170330/perf-tynty_30fps.csv

    python write_spatio_tempo_perf_tovideo.py ~/Dropbox/typing-notyping/C2L1P-B/20180223/perf-tynty_30fps.csv

    python write_spatio_tempo_perf_tovideo.py ~/Dropbox/typing-notyping/C2L1P-D/20180308/perf-tynty_30fps.csv

    python write_spatio_tempo_perf_tovideo.py ~/Dropbox/typing-notyping/C3L1P-C/20190411/perf-tynty_30fps.csv

    python write_spatio_tempo_perf_tovideo.py ~/Dropbox/typing-notyping/C3L1P-D/20190221/perf-tynty_30fps.csv
"""
import pdb
import cv2
import shutil
import argparse
from aqua.frameworks.typing import TypingPerfViz


def _arguments():
    """Parses input arguments."""
    args_inst = argparse.ArgumentParser(
        description=("""
            Creates a video at same frame rate as original with following information
            overlayed,
            1. Ground truth bbox
            2. Algorithm bbox
            3. IoU
            4. Confusion matrix labels

            Before using this make sure to run `write_spatio_tempo_perf_tocsv.py` script to
            create a csv file necessary for running this script.
        """))

    args_inst.add_argument("perf_csv", type=str, help=("CSV file containing performance data."))

    args = args_inst.parse_args()
    args_dict = {'perf_csv': args.perf_csv}

    return args_dict


def main():
    """Main function."""
    argd = _arguments()
    viz = TypingPerfViz(argd['perf_csv'])
    viz.write_to_video()




# execution starts here
if __name__ == "__main__":
    main()
