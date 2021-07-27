"""
DESCRIPTION
-----------
Activity association parent class.

USAGE
-----

EXAMPLE
-------
"""
import argparse
import pdb
import numpy as np
import pandas as pd
import os
import math
import cv2
from matplotlib import cm
from tqdm import tqdm
from sklearn.metrics.cluster import adjusted_rand_score, rand_score

# User defined libraries
import pytkit as pk
from hand_size import HandSize

class ActivityAssociation:
    """ 
    Parent class having common methods and
    variables.
    """

    
    actdf = pd.DataFrame()
    """ Activity dataframe """

    odf = pd.DataFrame()
    """ 
    Output data frame built from `actdf` with activities
    associations. The `person` column should have the associations.
    """

    actprops = {}
    """ Dictionary having some common activity properties """
    
    
    def __init__(self, actdf, stime, etime):
        """
        Parameters
        ----------
        act_csv_path : Str
            CSV file containing activities
        """

        # We only process activity labels between [stime, etime)
        fps = actdf['FPS'].unique().item()
        actdf = actdf[actdf['f0'] >= fps*stime ].copy()
        actdf = actdf[actdf['f0'] < fps*etime ].copy()

        # Copy into class varaible
        self.actdf = actdf.reset_index().copy()

        # Create a dictionary having activity properties
        self.actprops = {
            "stime" : stime,
            "etime" : etime,
            "fps"   : fps
        }

    def to_csv(self, loc, vname, csv_post_fix):
        """ 
        Writes base activity clusters to CSV file. Please take a note
        of the following columns.
        1. `cluster_id`: Another column is added having cluster id

        Parameters
        ----------
        loc : Str
            Location to store the labeled activities
        vname : Str
            Name of the video being processed
        csv_post_fix : Str
            We add this string to the csv name
        """
        
        # Output csv file name.
        ocsv_path = loc
        vname_noext = os.path.splitext(vname)[0]
        ocsv_path = (
            f"{ocsv_path}/{vname_noext}_{csv_post_fix}.csv"
        )

        # Write to csv file
        print(f"USER_INFO: Writing to \n\t{ocsv_path}")
        self.odf.to_csv(ocsv_path, index=False)

    def to_video(self, vpth, vid_post_fix):
        """
        Parameters
        ----------
        vname : Str
            Full video path
        csv_post_fix : Str
            We add this string to the csv name
        
        Returns
        -------
        A video with clusters of same activity in same color. The
        centroid of the clusters is marked with red dot.
        """        
        # Creating video object
        vi = pk.Vid(vpth, "read")

        # Create a video object for writing
        ovid_path = vi.props['dir_loc']
        vname = vi.props['name']
        ovid_path = (
            f"{ovid_path}/{vname}_{vid_post_fix}.mp4"
        )
        vo = pk.Vid(ovid_path, "write")
        
        # A black image to store clusters
        ccanvas = np.zeros(vi.props['frame_dim']).astype('uint8')

        # Draw clusters every second
        for t in tqdm(range(0, vi.props['duration'])):

            # Get current frame
            fnum = t*vi.props['frame_rate']
            frm = vi.get_frame(fnum)
            
            ccanvas, frm = (
                self._draw_clusters(ccanvas, fnum, frm)
            )

            # Write frame
            vo.writer.writeFrame(frm)

        # Close the writer
        vo.writer.close()

        
    def _draw_clusters(self, ccanvas, fnum, frm):
        """
        Parameters
        ----------
        ccanvas : Numpy array
            RGB image with clusters in different colors
        fnum : int
            Frame number
        frm : Numpy array
            Frame extracted from video
        """
        # Activities that happended at frame number
        df = self.odf.copy()
        df = df[df['f0'] <= fnum].copy()
        df = df[df['f0'] + df['f'] > fnum].copy()

        # Loop through activities and
        for idx, row in df.iterrows():

            # Cluster centroid
            ccw = int(row['cluster_coord'].split("-")[0])
            cch = int(row['cluster_coord'].split("-")[1])

            # Draw a red dot for cluster centroid
            color = tuple([
                255*x for x in list(cm.Set3(row['cluster_id']))[0:3]
            ])
            ccanvas = cv2.circle(
                ccanvas,
                (ccw,cch),
                radius=0,
                color=(0,0,255),
                thickness=5
            )

            # Draw activity bounding box
            try:
                ccanvas = cv2.rectangle(
                    ccanvas,
                    (int(row['w0']), int(row['h0'])),
                    (int(row['w0'] + row['w']), int(row['h0'] + row['h'])),
                    color,
                    1)
            except:
                pdb.set_trace()

        # Blend canvas and video frame
        alpha = 0.5
        beta = (1.0 - alpha)
        frm = cv2.addWeighted(frm, alpha, ccanvas, beta, 0.0)
        frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        
        return ccanvas, frm

    def evaluate_clusters(self, gt_col_name, metrics):
        """
        Parameters
        ----------
        gt_col_name : Str
            Name of the column in the DataFrame that has ground truth
        metrics : List of Str
            Name of the metrics to evaluate
            1. adj_RI = Adjusted Rand Index
            2. RI = Rand Index

        Returns
        -------
        Dictionary with the evaluated metrics

        Note
        ----
        Please use this function only if you have ground truth labels.
        """
        # Relabling cluster 1
        df = self.odf.copy()
        gt_labels = df[gt_col_name].astype('category').cat.codes.tolist()
        alg_labels = df['cluster_id'].astype('category').cat.codes.tolist()
        
        
        # Initializing performance metric
        perf = {}

        # Loop over each metric and update performance dictionary
        for metric in metrics:
            if metric == "adj_RI":
                perf["adj_RI"] = adjusted_rand_score(gt_labels, alg_labels)
            elif metric == "RI":
                perf["RI"] = rand_score(gt_labels, alg_labels)
            else:
                raise Exception(f"{metric} is not supported.")

        return perf
        
