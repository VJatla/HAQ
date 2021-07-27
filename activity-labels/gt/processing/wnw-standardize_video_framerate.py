"""
The following script standardizes videos to a
required frame rate (to 30 FPS).

The output video might have an error of -1 second
"""
import os
from aqua.data_tools import AOLMEActivityLabels

if __name__ == "__main__":

    if os.name == 'nt':
        raise Exception("Windows is not supported yet")
    else:
 #       rdir= "/home/vj/Dropbox/writing-nowriting-GT/C1L1P-D/20170330"
        rdir = "/home/vj/Dropbox/writing-nowriting-GT"
        labels_fname = "gTruth-wnw.csv"

    # Initialize activity labels instance
    act_labels = AOLMEActivityLabels(rdir, labels_fname)

    # Standardize videos to required frame rate (30 FPS)
    act_labels.standardize_videos(
        "/home/vj/Dropbox/typing-notyping/groups_db.csv",
        fr=30,
        overwrite=False)

