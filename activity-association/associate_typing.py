"""
DESCRIPTION
-----------
The following script associates spatio-temporal typing instances to a
person.


USAGE
-----
python associate_typing.py <Spatio-termporal typing instances> <Table region of interests>

EXAMPLE
-------
python associate_typing.py ~/Dropbox/typing-notyping/C1L1P-C/20170330/alg-tynty_30fps.csv ~/Dropbox/table_roi_annotation/C1L1P-C/20170330/video_roi.csv
"""


import argparse
import pdb
import pandas as pd
import numpy as np
import pytkit as pk
import math
import sys


def _arguments():
    """ Parses input arguments """
    
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=("""The following script associates spatio-temporal
        typing instances to a person."""),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Adding arguments
    args_inst.add_argument(
        "ty",
        type=str,
        help="CSV file having spatio-temporal typing instances."
    )
    args_inst.add_argument(
        "roi",
        type=str,
        help="Table regions associated to a person."
    )
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {
        'ty': args.ty,
        'roi': args.roi
    }

    # Return argument dictionary
    return args_dict


def get_regions_centroid(act_row, roi_df):
    """ Returns a dictionary with
    1. key: having pseudonym of the person
    2. value: having medin centorid
    """
    act_vname = act_row['name']
    act_f0    = act_row['f0']
    act_f     = act_row['f']
    act_bbox  = [act_row['w0'], act_row['h0'], act_row['w'], act_row['h']]

    # bounding boxes at temporal activity location
    roi_df2 = roi_df.copy()
    roi_df2 = roi_df2[roi_df2['video_names'] == act_vname]
    roi_df2 = roi_df2[roi_df2['f0'] >= act_f0]
    roi_df2 = roi_df2[roi_df2['f0'] < act_f0 + act_f]

    # Remove unnecessary columns
    roi_df2 = roi_df2.drop(columns=['Time', 'f0','video_names', 'f'])

    # Loop over each row
    regions_dict = {}
    for person_ in roi_df2.columns:

        # Getting region bboxes for a person and removing nan
        bboxes_str = roi_df2[person_]
        bboxes_str = bboxes_str.dropna()
        bbox_centroid_lst = np.zeros((len(bboxes_str), 2), dtype='int')

        for idx, bbox_str in enumerate(bboxes_str):
            try:
                bbox_lst = [int(x) for x in bbox_str.split('-')]
                bbox_centroid_lst[idx, :] = [
                    bbox_lst[0] + bbox_lst[2]/2,
                    bbox_lst[1] + bbox_lst[3]/2
                ]
            except:
                import pdb; pdb.set_trace()
        
        bbox_centroid_median = np.median(bbox_centroid_lst, axis=0).astype('int')
        regions_dict[person_] = bbox_centroid_median

    return regions_dict


def calculate_centroid_dist(roi_bbox_centroid, act_row):
    """ Returns centroid distance between activity bounding box and
    table region.
    """
    
    
    # ROI centroid
    roi_wc = roi_bbox_centroid[0]
    roi_hc = roi_bbox_centroid[1]

    # Activity centorid
    act_wc = act_row['w0'] + act_row['w']/2
    act_hc = act_row['h0'] + act_row['h']/2

    # Centroid distance
    dist = math.sqrt(
        (roi_wc-act_wc)*(roi_wc-act_wc) +
        (roi_hc-act_hc)*(roi_hc-act_hc)
    )

    return dist

    
def associate_ty_to_nearest(act_df, roi_df):
    """ Associate activity to the nearest, centroid distance, person.
    
    Parameters
    ----------
    act_df : DataFrame
        DataFrame containing activity(typing) instances 
    roi_df : DataFrame
        DataFrame containing regions labeled
    """

    act_df2 = act_df.copy()

    for ridx, row in act_df.iterrows():

        # bounding boxes at temporal activity location
        rois_centroids = get_regions_centroid(row, roi_df)

        # Label the activity with nearest region
        nearest_centroid_dist = math.inf
        for rois_key in rois_centroids:
            
            roi_bbox_centroid = rois_centroids[rois_key]
            centroid_dist = calculate_centroid_dist(roi_bbox_centroid, row)
            
            if centroid_dist <= nearest_centroid_dist:
                nearest_centroid_dist = centroid_dist
                person_ = rois_key

        # Setting person for current activity
        act_df2.at[ridx, 'person_nearest_centroid'] = person_
    return act_df2.copy()

# Execution starts from here
if __name__ == "__main__":

    if len(sys.argv) > 1:
         # Input arguments
        args = _arguments()

        # Load activity and regions dataframes
        ty_df = pd.read_csv(args['ty'])
        roi_df = pd.read_csv(args['roi'])

        # Create a column with 'nearest_person' column
        ty_df2 = associate_ty_to_nearest(ty_df, roi_df)

        # Write dataframe
        ty_df2.to_csv(args['ty'], index=False)        

    else:
       
        roi_files = [
            "/home/vj/Dropbox/table_roi_annotation/C1L1P-C/20170330/video_roi.csv",
            "/home/vj/Dropbox/table_roi_annotation/C1L1P-C/20170413/video_roi.csv",
            "/home/vj/Dropbox/table_roi_annotation/C1L1P-E/20170302/video_roi.csv",
            "/home/vj/Dropbox/table_roi_annotation/C2L1P-B/20180223/video_roi.csv",
            "/home/vj/Dropbox/table_roi_annotation/C2L1P-D/20180308/video_roi.csv",
            "/home/vj/Dropbox/table_roi_annotation/C3L1P-C/20190411/video_roi.csv",
            "/home/vj/Dropbox/table_roi_annotation/C3L1P-C/20190411/video_roi.csv"
        ]
        ty_files = [
            "/home/vj/Dropbox/typing-notyping/C1L1P-C/20170330/alg-tynty_30fps.csv",
            "/home/vj/Dropbox/typing-notyping/C1L1P-C/20170413/alg-tynty_30fps.csv",
            "/home/vj/Dropbox/typing-notyping/C1L1P-E/20170302/alg-tynty_30fps.csv",
            "/home/vj/Dropbox/typing-notyping/C2L1P-B/20180223/alg-tynty_30fps.csv",
            "/home/vj/Dropbox/typing-notyping/C2L1P-D/20180308/alg-tynty_30fps.csv",
            "/home/vj/Dropbox/typing-notyping/C3L1P-C/20190411/alg-tynty_30fps.csv",
            "/home/vj/Dropbox/typing-notyping/C3L1P-C/20190411/alg-tynty_30fps.csv"
        ]

        for i, ty_file in enumerate(ty_files):
            
            print(f"{ty_file}")
            roi_file = roi_files[i]

            # Load data frames
            roi_df = pd.read_csv(roi_file)
            ty_df = pd.read_csv(ty_file)

            # Create a column with 'nearest_person' column
            ty_df2 = associate_ty_to_nearest(ty_df, roi_df)

            # Write dataframe
            ty_df2.to_csv(ty_file, index=False)
