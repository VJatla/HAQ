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

    num_clusters = 0
    """
    Number of clusters from algorithm
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
        actdf = actdf[actdf['f0'] >= fps * stime].copy()
        actdf = actdf[actdf['f0'] < fps * etime].copy()

        # Number of unique cluster as per gt
        num_clus_gt = len(actdf['person'].unique())

        # % of activity duration
        total_act_dur = self._get_total_activity_duration(actdf, fps)

        # Copy into class varaible
        self.actdf = actdf.reset_index().copy()

        # Create a dictionary having activity properties
        self.actprops = {
            "stime": stime,
            "etime": etime,
            "fps": fps,
            "total_act_dur": total_act_dur,
            "num_clus_gt":num_clus_gt
        }

    def _get_total_activity_duration(self, df, fps):
        """ Get total activity duration.

        Parameters
        ----------
        df : DataFrame
            DataFarame containing activities
        fps : int
            FPS of current session under consideration
        """
        df_copy = df.copy()
        total_act_dur = int(df_copy['f'].sum()/fps)
        df_copy['f'].sum()/fps

        return total_act_dur

    def to_video(self, vdir, vid_post_fix):
        """
        Parameters
        ----------
        vname : Str
            Directory containing videos
        csv_post_fix : Str
            We add this string to the csv name

        Returns
        -------
        A video with clusters of same activity in same color. 
        """
        
        # Class variables to Function variables
        df = self.odf.copy()

        # Unique videos
        videos = df["name"].unique().tolist()

        # Loop through each video creating an overly
        for vname_ext in videos:
            # Input video path
            ivid_path = (f"{vdir}/{vname_ext}")

            # Current video activity dataframe
            vdf = df[df["name"] == vname_ext].copy()

            # Output video path
            vname_no_ext = os.path.splitext(vname_ext)[0]
            ovid_path = (
                f"{vdir}/{vname_no_ext}_{vid_post_fix}.webm"
            )

            # If video with clusters is not present create it
            if not os.path.isfile(ovid_path):
                
                # Creating the video
                print(f"Creating : {vname_no_ext}_{vid_post_fix}.webm")
                self._create_video_with_clusters(ivid_path, ovid_path, vdf)



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
                perf["RI"] = round(
                    rand_score(gt_labels, alg_labels), 2
                )
            else:
                raise Exception(f"{metric} is not supported.")

        return perf


    
    def _create_video_with_clusters(self, ipth, opth, df):
        """ A very fast video playback with clusters. This also returns
        a canvas having all the clusters
        
        Parameters
        ----------
        ipth : Str
            Input video path
        opth : Str
            Output video path
        df : Pandas DataFrame
            Dataframe containing video activities of current video
        """

        # Input and output video instances using pytkit
        vi = pk.Vid(ipth, "read")
        vo = pk.Vid(opth, "write")

        # An image to store clusters
        ccanvas = np.zeros(vi.props['frame_dim']).astype('uint8')

        # Adding labels to the canvas
        ccanvas = self._add_cluster_labels(ccanvas, df.copy())

        # Draw clusters every second
        for t in tqdm(range(0, vi.props['duration'],3)):

            # Get current frame
            fnum = t * vi.props['frame_rate']
            frm = vi.get_frame(fnum)

            ccanvas, frm = (self._draw_clusters(df, ccanvas, fnum, frm))

            # Write frame
            vo.writer.writeFrame(frm)

        # Close the writer
        vo.writer.close()

    def _add_cluster_labels(self, ccanvas, df):
        """ Adds cluster labels to the canvas
        
        Parameters
        ----------
        ccanvas : ndarray
            Canvas used to create clusters
        df : Pandas Dataframe
            Data frame having activity clusters
        """
        cluster_ids = df['cluster_id'].unique()

        for cluster_id in cluster_ids:
            color = tuple(
                    [255 * x for x in list(cm.Set3(cluster_id))[0:3]]
            )
            
            ccanvas = cv2.putText(
                ccanvas,
                f"Cluster - {cluster_id}",
                (5, 20 + cluster_id*20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA
            )
        return ccanvas
                                  
        
    def _draw_clusters(self, df, ccanvas, fnum, frm):
        """
        Parameters
        ----------
        df : Pandas DataFrame
            Dataframe containing video activities of current video
        ccanvas : Numpy array
            RGB image with clusters in different colors
        fnum : int
            Frame number
        frm : Numpy array
            Frame extracted from video
        """

        # Activities at fnum
        adf = df[df['f0'] <= fnum].copy()
        adf = adf[adf['f0'] + adf['f'] > fnum].copy()

        # If adf is not empty
        if not adf.empty:
            
            # Loop through activities and
            for idx, row in adf.iterrows():
               
                # Color
                color = tuple(
                    [255 * x for x in list(cm.Set3(row['cluster_id']))[0:3]])

                # Draw activity bounding box
                ccanvas = cv2.rectangle(
                    ccanvas, (int(row['w0']), int(row['h0'])),
                    (int(row['w0'] + row['w']), int(row['h0'] + row['h'])),
                    color, 2)
     
        # Blend canvas and video frame
        alpha = 0.5
        beta = (1.0 - alpha)
        frm = cv2.addWeighted(frm, alpha, ccanvas, beta, 0.0)
        frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

        return ccanvas, frm

        
