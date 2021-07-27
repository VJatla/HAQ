"""
Temporal:
    Creates a csv file with each `t` second video segment from automation algorithm classified as
    tp, tn, fp or fn (Confusion matirx). A `t` second instance from ground truth is considered to
    belong to typing if >=50% of frames (>= 15 frames if `t` == 1 sec) belong to typing.The output
    is a CSV file classifying every `t` second video segment in the session as tp, tn, fp, fn.


Example
-------
Linux:
    python write_spatio_tempo_perf_tocsv.py\
    ~/Dropbox/typing-notyping/C1L1P-E/20170302/gTruth-tynty_30fps.csv\
    ~/Dropbox/typing-notyping/C1L1P-E/20170302/alg-tynty_30fps.csv\
    ~/Dropbox/typing-notyping/C1L1P-E/20170302/properties_session.csv\
    ~/Dropbox/typing-notyping/C1L1P-E/20170302/perf-tynty_30fps.csv
"""
import pdb
import argparse
from aqua.frameworks.typing import TypingPerfEval


def _arguments():
    """Parses input arguments."""
    args_inst = argparse.ArgumentParser(
        description=(
            """
            Temporal:
            Creates a csv file with each `t` second video segment from automation algorithm classified as
            tp, tn, fp or fn (Confusion matirx). A `t` second instance from ground truth is considered to
            belong to typing if >=50% of frames (>= 15 frames if `t` == 1 sec) belong to typing.The output
            is a CSV file classifying every `t` second video segment in the session as tp, tn, fp, fn.
            Spatial:
            IoU + bounding box coordinates (GT + Automation) are documented. Color code:
            1. tp = Green
            2. tn = Green
            3. fp = Yello (we are ok with fp)
            4. fn = Red (we don't like fn).
            """))

    args_inst.add_argument("gt_csv", type=str, help=("ground truth csv file."))
    args_inst.add_argument("alg_csv", type=str, help=("Algorithm csv file."))
    args_inst.add_argument("prop_csv", type=str, help=("CSV file having session properties."))
    args_inst.add_argument("out_csv", type=str, help=("Output CSV file."))

    args = args_inst.parse_args()
    args_dict = {'gt_csv': args.gt_csv,
                 'alg_csv': args.alg_csv,
                 'prop_csv': args.prop_csv,
                 'out_csv': args.out_csv}

    return args_dict


def main():
    """Main function."""
    argd = _arguments()
    perf_eval = TypingPerfEval(argd['gt_csv'], argd['alg_csv'], argd['prop_csv'])
    iou_per_sec = perf_eval.write_spatio_tempo_perf_tocsv(1, argd['out_csv'])



# Execution starts here
if __name__ == "__main__":
    main()
