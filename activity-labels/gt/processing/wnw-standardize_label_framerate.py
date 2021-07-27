"""
The following script standardizes activity labels to 
a standard frame rate (to 30 FPS).
"""
import os
from aqua.data_tools import AOLMEActivityLabels

if __name__ == "__main__":

    if os.name == 'nt':
        raise Exception("Windows is not supported yet")
    else:
        rdir = "/home/vj/Dropbox/writing-nowriting-GT"
        labels_fname = "gTruth-wnw.csv"

    # Initialize activity labels instance
    act_labels = AOLMEActivityLabels(rdir, labels_fname)

    # Standardize video frame rate
    act_labels.standardize_activity_labels(fr=30, overwrite=True)
