# -*- coding: utf-8 -*-
""" Generates summary of ground truth

The following script summarizes ground truth as csv files at group and AOLME
level(all cohorts).

Group level summary is stored as `group_summary.csv` in each group directory,
ex: `~/Dropbox/typing-notyping/C1L1P-A`.

AOLME level summary is stored as `gt_summary.csv` above group directories,
ex: `~/Dropbox/typing-notyping`.

Each csv file has following columns(for typing, notyping),
1. group
2. date
3. tot-vids
4. tot-dur
5. FPS
6. resolution
7. ana-vids
8. ana-time
9. typing
10. notyping


Usage
-----
$ python summarize_gt.py <rdir> <gt_csv_name> <groups_db> '<activities>'


Example
-------
python summarize_gt.py ~/Dropbox/typing-notyping gTruth-tynty.csv\
    ~/Dropbox/typing-notyping/groups_db.csv 'typing,notyping'


Note
-----
1. This script assumes the following directory structure,
    typing-notyping
        ├── C1L1P-A
        │   ├── 20170216
        │   │   ├── G-C1L1P-Feb16-A-Cesar_q2_08-09.mp4
        │   │   ├── G-C1L1P-Feb16-A-Cesar_q2_08-09-wrong.mat
        │   │   ├── gTruth-tynty.csv
        │   │   └── gTruth-tynty-G-C1L1P-Feb16-A-Cesar_q2_08-09.mat

2. Each session directory should have `session_info.json` with
   `split-type` information.
"""

import pdb
from aqua import GTSummary
import argparse
from argparse import RawTextHelpFormatter


def cmd_line_arguments():
    """Defines and parses command line arguments

    Returns
    -------
    dict
        Dictionary containing parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="""
description:
  This script summarizes ground truth as csv files at group and AOLME
  level(all cohorts).
    - Group level summary = `group_summary.csv`
    - AOLME level summary = `gt_summary.csv`""",
                                     formatter_class=RawTextHelpFormatter)

    # Arguments
    parser.add_argument('rdir',
                        type=str,
                        help='Root directory of activity ground truth')
    parser.add_argument('gt_csv_name',
                        type=str,
                        help='name of ground truth file. ex:gTruth-tynty.csv')
    parser.add_argument('groups_vid_db',
                        type=str,
                        help='Groups database from AOLME website as CSV')
    parser.add_argument('split_info_file',
                        type=str,
                        help=''''CSV with Training, validation and testing
                        'split information.''')
    parser.add_argument('activities',
                        type=str,
                        help='Groups database exported from AOLME website.')

    # Parse arguments
    args = parser.parse_args()
    args_dict = {
        'rdir': args.rdir,
        'gt_csv_name': args.gt_csv_name,
        'groups_vid_db': args.groups_vid_db,
        'split_info_file': args.split_info_file,
        'activities': args.activities
    }

    return args_dict


if __name__ == '__main__':

    # Command line arguments
    arg_dict = cmd_line_arguments()

    # Initialize ground truth summary object
    typing_summary = GTSummary(**arg_dict)

    # Get all the session paths having `<ground truth>.csv` file
    all_session_paths = typing_summary.get_all_session_paths()

    # Check for `session_properties.csv` in each session. If it does not
    # exist create it. The sencond argument can overwrite existing files.
    typing_summary.session_properties_as_csv(all_session_paths,
                                             rewrite_flag=False)

    # Produce `gtsummary_session.csv` summarizing at session level
    typing_summary.gt_summary_session_as_csv(all_session_paths)

    # `gtsummary_group.csv` summarizing at group level
    typing_summary.gt_summary_group_as_csv(all_session_paths)

    # `gtsummary_AOLME.csv` summarizing at AOLME level
    typing_summary.gt_summary_AOLME_as_csv(all_session_paths)

    # `gtsummary_AOLME_with_splits.csv` summarizing at AOLME level
    typing_summary.gt_split_summary_as_csv()

