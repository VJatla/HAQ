"""
Creates spatiotemporal trims having activities from activity labels. The trim is
taken from the middle of an activity instance.
"""

import os
from aqua.data_tools import AOLMEActivityLabels

if __name__ == "__main__":

    if os.name == 'nt':
        raise Exception("Windows is not supported yet")
    else:
        rdir = "/home/vj/Dropbox/typing-notyping"
        odir = "/mnt/twotb/aolme_datasets/tynty/trimmed_videos/one_trim_per_instance_3sec"
        labels_fname = "gTruth-tynty_30fps.csv"

    # Initialize activity labels instance
    act_labels = AOLMEActivityLabels(rdir, labels_fname)

    # One trim per instance
    act_labels.create_spatiotemporal_trims(odir,
                                           trims_per_instance=1,
                                           overwrite=True)

    # Trims completely
    # act_labels.create_spatiotemporal_trims(odir,
    #                                        trims_per_instance=-1,
    #                                        overwrite=True)
