import pdb


def parse_video_name(vname):
    """
    Parses AOLME video name, creating a dictionary of useful informaiton.
    Supports both trimmed video names and full video names.

    Parameters
    ----------
    vname: str
        video name. The assuption is the name follows this covention
        G-C1L1P-Apr06-A-*

    Todo
    ----
    1. How to support group name having two cameras?
    """
    vname_dict = {}

    vname_arr = vname.split("-")

    # Cohort
    vname_dict['cohort'] = vname_arr[1][1:2]

    # Level
    vname_dict['level'] = vname_arr[1][3:4]

    # School
    vname_dict['school'] = vname_arr[1][4:5]

    # Date
    vname_dict['date'] = vname_arr[2]

    # Group
    vname_dict['group'] = vname_arr[3]

    return vname_dict
