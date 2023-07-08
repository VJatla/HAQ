"""
Description
-----------
Distribute the images to training, validation and testing as recommended by
mmclassification tutorial.
The recommendation is as follows, (https://mmclassification.readthedocs.io/en/latest/tutorials/new_dataset.html#reorganize-dataset-to-existing-format)

```sh
train/
├── cat
│   ├── xxx.png
│   ├── xxy.png
│   └── ...
│       └── xxz.png
├── bird
│   ├── bird1.png
│   ├── bird2.png
│   └── ...
└── dog
    ├── 123.png
    ├── nsdf3.png
    ├── ...
    └── asd932_.png
val/

test/
```

Output
------
train, val and test directories having images.

Example
-------
python split_data.py \
    /home/vj/twotb/aolme_datasets/wnw_table_roi/hands_with_withot_pen_images \
    /home/vj/Dropbox/writing-nowriting/trn-val-tst-splits.csv

"""


import argparse
import os
import sys
import pytkit as pk
import pandas as pd
import shutil


def get_year(group_name):
    """Returns year based on group name"""
    if "C1" in group_name:
        return "2017"
    elif "C2" in group_name:
        return "2018"
    elif "C3" in group_name:
        return "2019"
    else:
        sys.exit(f"The cohort {group_name} is not supported")

def get_month(month_str):
    """Returns month as a numerical value"""
    if month_str == "Jan":
        return "01"
    elif month_str == "Feb":
        return "02"
    elif month_str == "Mar":
        return "03"
    elif month_str == "Apr":
        return "04"
    elif month_str == "May":
        return "05"
    elif month_str == "Jun":
        return "06"
    elif month_str == "Jul":
        return "07"
    elif month_str == "Aug":
        return "08"
    elif month_str == "Sep":
        return "09"
    elif month_str == "Oct":
        return "10"
    elif month_str == "Nov":
        return "11"
    elif month_str == "Dec":
        return "12"
    else:
        sys.exit(f"Unsupported month {month_str}")

def get_session_info(file_path):
    """Returns session information as a dictionary from the file_path.
    The dictionary has following keys,
    1. Group, example C1L1P-D
    2. Date, example 20170330
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    file_name_split = file_name.split("-")
    group_name = f"{file_name_split[1]}-{file_name_split[3]}"

    month = get_month(file_name_split[2][0:3])
    day = file_name_split[2][3:5]
    year = get_year(group_name)
    date = int(f"{year}{month}{day}")

    return {"group": group_name, "date": date}

def copy_data(rdir, split_df, cls_labels, cls_dirs, split_label):
    """Copies data into appropriate split.

    Parameters
    ----------
    rdir: Str
        Root directory
    split_df : DataFrame
        Dataframe with split information. Each session is labeled
        as belonging to training, validation or testing.
    cls_labels : List[Str]
        Class labels
    cls_dirs : List[Str]
        Class directories w.r.t. root directory.
    split_label : Str
        Current split label. It can take any of the following values
        {"train", "val", "test"}
    """

    # Creating data split directory. If it exists we delete it and
    # create a new one
    split_dir = f"{rdir}/{split_label}"
    if os.path.isdir(split_dir):
        shutil.rmtree(split_dir)
    os.mkdir(split_dir)

    # train, val and test directories
    trn_dir = f"{rdir}/train"
    val_dir = f"{rdir}/val"
    tst_dir = f"{rdir}/test"

    if os.path.isdir(trn_dir):
        shutil.rmtree(trn_dir)
    os.mkdir(trn_dir)
    if os.path.isdir(val_dir):
        shutil.rmtree(val_dir)
    os.mkdir(val_dir)
    if os.path.isdir(tst_dir):
        shutil.rmtree(tst_dir)
    os.mkdir(tst_dir)
    
    
    for i in range(0, len(cls_labels)):

        cls_label = cls_labels[i]
        cls_dir = f"{rdir}/{cls_dirs[i]}"
        cls_files = pk.get_file_paths_with_kws(cls_dir, [".png"])

        for cls_file_path in cls_files:

            session_info = get_session_info(cls_file_path)

            # Get label, trn, val or tst
            split_df_fil = split_df.loc[
                (split_df["date_full"] == session_info["date"])
                &
                (split_df["group"] == session_info["group"])
            ].copy()
            if len(split_df_fil) > 1:
                sys.exit("More than two sessions!!!")
                
            split_label = split_df_fil["label"].item()

            if split_label == "trn":

                label_dir = f"{trn_dir}/{cls_label}"
                if not os.path.isdir(label_dir):
                    os.mkdir(label_dir)
                    
            elif split_label == "val":
                label_dir = f"{val_dir}//{cls_label}"
                if not os.path.isdir(label_dir):
                    os.mkdir(label_dir)

            elif split_label == "tst":
                label_dir = f"{tst_dir}//{cls_label}"
                if not os.path.isdir(label_dir):
                    os.mkdir(label_dir)

            cmd = f"cp {cls_file_path} {label_dir}/"
            os.system(cmd)

                    

            
    
    



# Execution starts from here???
if __name__ == "__main__":

    # Inputs, change as needed
    rdir = "/mnt/twotb/aolme_datasets/wnw_table_roi/hands_with_withot_pen_images"
    cls_labels = ["hands-with-pen", "hands-without-pen"]
    cls_dirs = ["hands_with_pen_clean", "hands_without_pen_clean"]
    split_csv = "~/Dropbox/writing-nowriting/trn-val-tst-splits.csv"
    
    # Parsing input arguments to variables
    split_df = pd.read_csv(split_csv)

    # Copy training dataset to `rdir/train`
    print(f"Copying training data")
    copy_data(rdir, split_df, cls_labels, cls_dirs, "train")

    # Copy validation dataset to `rdir/val`
    print(f"Copying validation data")
    copy_data(rdir, split_df, cls_labels, cls_dirs, "val")

    # Copy testing dataset to `rdir/test`
    print(f"Copying testing data")
    copy_data(rdir, split_df, cls_labels, cls_dirs, "test")
