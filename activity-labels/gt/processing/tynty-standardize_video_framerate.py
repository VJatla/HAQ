"""
The following script standardizes videos to a
required frame rate (to 30 FPS).

The output video might have an error of -1 second
"""
import os
from aqua.data_tools import AOLMEActivityLabels

if __name__ == "__main__":

    if os.name == 'nt':
        #raise Exception("Windows is not supported yet")
        rdir = "C:/Users/vj/Dropbox/typing-notyping/C1L1P-C/"
        labels_fname = "gTruth-tynty.csv"
        video_db_path = "C:/Users/vj/Dropbox/typing-notyping/groups_db.csv"
    else:
        rdir = "/home/vj/Dropbox/typing-notyping/"
        labels_fname = "gTruth-tynty.csv"
        video_db_path = "/home/vj/Dropbox/typing-notyping/groups_db.csv"

    # Initialize activity labels instance
    act_labels = AOLMEActivityLabels(rdir, labels_fname)

    # Standardize video frame rate
    act_labels.standardize_videos(
        video_db_path,
        fr=30,
        overwrite=False)
