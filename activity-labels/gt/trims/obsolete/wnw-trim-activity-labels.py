"""


-----------------------

       OBSOLETE

-----------------------



Creates spatiotemporal trims having activities from activity labels.
"""

import os
from aqua.data_tools import AOLMEActivityLabels

if __name__ == "__main__":

    if os.name == 'nt':
        raise Exception("Windows is not supported yet")
    else:
        rdir = "/home/vj/Dropbox/writing-nowriting-GT"
        odir = "/mnt/twotb/aolme_datasets/wnw/trimmed_videos/full_trims"
        labels_fname = "gTruth-wnw_30fps.csv"

    # Initialize activity labels instance
    act_labels = AOLMEActivityLabels(rdir, labels_fname)

    # trims_per_instance = -1 implies we are trimming completely
    act_labels.create_spatiotemporal_trims(odir,
                                           trims_per_instance=-1,
                                           overwrite=True)
