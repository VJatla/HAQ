"""
                     DEPRICATION WARNING

THIS IS OLD. IT IS REPLACED BY THE SCRIPT LOCATED AT
`./activity-labels/generate_activity_map.py`




This script generates activity maps at session tier and
group tier as png images.

Below you can see description of arguments

    rdir: str
        Root directory having activity labels
    fname: str
        Activity labels file name with extension. Expects a csv file.
    pseudonyms_fpath: str
        Path to file that maps kids numeric code
    tiers: list[str]
        List having data organization tiers which require activity maps.
        By default generates session tier maps only.
    lib: str
        Plotting library to use for current plots. Fow now it only
        supports **"plotly"**.
    show_flag: bool
        Displays plots before saving them. For `plotly` library it
        requires user key press to continue. For `matplotlib` the
        user should colse the plot window to continue. Defaults to
        `False`.
"""
import os
from aqua.data_tools.aolme.activity_labels import AOLMEActivityMaps

if __name__ == '__main__':

    if os.name == 'nt':
        # Windows: Arguments initialized with typing ground truth
        args_dict = {
            'rdir':
            "C:/Users/vj/Dropbox/typing-notyping",
            'labels_fname':
            "gTruth-tynty_30fps.csv",
            'pseudonyms_fpath': ("C:/Users/vj/Dropbox/typing-notyping/"
                                 "kid-pseudonym-mapping.csv"),
            'tiers': ["session"],
            'lib':
            "plotly",
            'Show_flag':
            False
        }
    else:
        # Linux: Arguments initialized with typing ground truth
        args_dict = {
            'rdir':
            "/home/vj/Dropbox/typing-notyping",
            'labels_fname':
            "gTruth-tynty_30fps.csv",
            'pseudonyms_fpath': ("/home/vj/Dropbox/typing-notyping/"
                                 "kid-pseudonym-mapping.csv"),
            'tiers': ["session"],
            'lib':
            "plotly",
            'show_flag':
            False
        }

    # Initialize AOLME activity map instance using ground truth data
    tynty_act_maps = AOLMEActivityMaps(**args_dict)

    # Generate session and group tier activity maps
    tynty_act_maps.save_activity_maps()
