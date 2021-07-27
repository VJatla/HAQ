import argparse
import pdb
import numpy as np
import pandas as pd
import os
import math

# User defined libraries
import pytkit as pk
from activity_association import ActivityAssociation

class BaseClusters(ActivityAssociation):
    """ 
    This class associates activities that are very close to each
    other to same cluster. 
    
    We use  `factor*hand_size` to determine if two activities are 
    close to each other.
    """

    hand_size = None
    """ Hand size """
    
    def __init__(self, hand_size, activity_df, stime=0, etime=math.inf):
        """
        Parameters
        ----------
        hand_size : int
            Area of hand bounding box
        """
        # Calling ActivityAssociation class init function
        super(BaseClusters, self).__init__(
            activity_df, stime, etime
        )

        # Set hand size
        self.hand_size = hand_size


    def cluster_activities(self):
        """ 
        Create a dataframe with cluster id.
        """
        # Creating a dataframe with `cluster_id` column initialized to -1
        actdf = self.actdf.copy()

        # Initializing cluster variables
        cdict = {}

        # Loop thrugh each activity
        cluster_id = []
        cluster_coord = []
        for ridx, row in actdf.iterrows():
            
            # Current centorid coordinates
            cur_coord = self._get_cur_centroid(row)
            cur_clust, clust_coord, cdict = (
                self._assign_and_update_clusters(cur_coord, cdict)
            )

            # Updating cluster id in dataframe
            cluster_id += [cur_clust]
            cluster_coord += [f"{clust_coord[0]}-{clust_coord[1]}"]
            
        # write the actdf to Class variable
        actdf['cluster_id'] = cluster_id
        actdf['cluster_coord'] = cluster_coord
        self.odf = actdf.copy()



            
    def _assign_and_update_clusters(self, act_coord, cdict):
        """
        This is the core of base clustering. This method determines
        the cluster id of an activity.
        
        Parameters
        ----------
        act_coord : Numpy array
            Current activity centroid coordinates
        cdict : Dictionary
            A dictionary to keep track of cluster label and centroids
        """
        # hand size multiplication factor
        alpha = 2
        
        # First cluster
        if len(cdict) == 0:
            cdict = {
                "0" : {
                    "clus_coord" : act_coord,
                    "nsamples" : 1
                }
            }
            return 0, act_coord, cdict
        
        else:
            
            # Calculating min distance to clusters
            dist = self._get_distances_wrt_clusters(act_coord, cdict)
            dist_min = min(dist)
            dist_min_idx = dist.index(dist_min)

            # If the activity is not far from known clusters
            if dist_min <= alpha*math.sqrt(self.hand_size):
                new_coord = self._get_new_clus_coord(
                    act_coord,
                    cdict[f"{dist_min_idx}"] 
                )
                cdict[f"{dist_min_idx}"]["clus_coord"] = new_coord
                cdict[f"{dist_min_idx}"]["nsamples"] += 1
                return dist_min_idx, new_coord, cdict

            # If the activity is far from known clusters
            else:
                n_clusters = len(cdict)
                new_clus_idx = n_clusters
                cdict[f"{new_clus_idx}"] = {
                    "clus_coord" : act_coord,
                    "nsamples" : 1
                }
                return new_clus_idx, act_coord, cdict

    def _get_new_clus_coord(self, act_coord, clus_props):
        """
        Parameters
        ----------
        act_coord : Numpy array
            Activity coordinate under consideration
        clus_props : Dictionary
            Cluster properties we wish to update with new activity
            coordinate.
        """
        (w, h) = clus_props['clus_coord']
        (wc, hc) = act_coord
        n = clus_props['nsamples']

        # Calcualte new values for cluster coordinate
        new_w = ((n*w) + wc)/(n+1)
        new_h = ((n*h) + hc)/(n+1)
        new_coord = (int(new_w), int(new_h))
        
        return np.array(new_coord)

    def _get_distances_wrt_clusters(self, cur_coord, cdict):
        """
        Parameters
        ---------
        cur_coord : Numpy array
            Current activity centroid coordinate
        cdict : Dictionary
            A dictionary containing all the cluster centroids along
            with labels.
        """
        dist = []
        for cluster_label in cdict:
            cluster_coord = cdict[cluster_label]["clus_coord"]
            cur_dist = np.linalg.norm(cluster_coord - cur_coord)
            dist += [cur_dist]
        return dist
        
    def _get_cur_centroid(self, row):
        """
        Parameters
        ----------
        row : Pandas Series
            Pandas series containing one instance of activity
        """
        wc = int(row['w0'] + (row['w']/2))
        wh = int(row['h0'] + (row['h']/2))
        act_coord = (wc, wh)
        return np.array(act_coord)
