import os
def get_file_paths_with_kws(pth, kw_lst):
    """
    Lists full paths of files having certian keywords in their names

    Parameters
    ----------

    pth: str Path to the root directory containing files.  kw_lst:
    list of str List of key words the files have
    """
    # Check if directory is valid
    if not (os.path.exists(pth)):
        raise Exception(f"The path {pth} is not valid.")

    # create a list using comma separated values
    kw_lst_csv = []
    for idx, litem in enumerate(kw_lst):
        litem_split = litem.split(",")
        if len(litem_split) > 1:
            kw_lst_csv = kw_lst_csv + litem_split
        else:
            kw_lst_csv.append(litem_split[0])

    # Loop through each file
    files = []
    for r, d, f in os.walk(pth):
        for file in f:
            # Break comma separated values
            # Check if current file contains all of the key words
            is_valid_file = all(kw in file for kw in kw_lst_csv)
            if is_valid_file:
                files.append(os.path.join(r, file))

    # return
    return files

def files_exist(list_of_file_paths):
    """ Checks if all the files in a list exist.

    Parameters
    ----------
    list_of_file_paths: List of str
        List of file paths to check
    """
    for path in list_of_file_paths:
        if not os.path.exists(path):
            raise Exception(f"Path does not exist\n\t{path}")
