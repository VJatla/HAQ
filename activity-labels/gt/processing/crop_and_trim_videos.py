"""
The following script trims and crops activity labels from videos.
"""
import os
from aqua.data_tools.aolme import ActivityPreprocessor

if __name__ == '__main__':

    if os.name == 'nt':
        # Windows: Arguments initialized with typing ground truth
        args_dict = {
            'rdir': "C:/Users/vj/Dropbox/typing-notyping",
            'labels_fname': "gTruth-tynty_30fps.csv",
        }
        odir = "C:/Users/vj/Dropbox/typing-notyping/crop_and_trimmed_3sec"
    else:
        # Linux: Arguments initialized with typing ground truth
        args_dict = {
            'rdir': "/home/vj/Dropbox/typing-notyping",
            'labels_fname': "gTruth-tynty_30fps.csv",
        }
        odir = "/mnt/twotb/vjdata/AOLME/tynty/crop_and_trim_3sec"

    # Initialize preprocessing object
    preproc = ActivityPreprocessor(**args_dict)

    # Crop and trim videos

    preproc.corp_and_trim(odir)
