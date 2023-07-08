import os
import sys
import pdb
import cv2
import aqua
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from aqua.nn.dloaders import AOLMETrmsDLoader
import pytkit as pk
from aqua.nn.models import DyadicCNN3D
from torchsummary import summary


class Writing3:
    
    wdf = pd.DataFrame()

    wdf_roi_only = pd.DataFrame()

    wrp = pd.DataFrame()

    wrp_roi_only = pd.DataFrame()

    cfg = {}
    """Configuration dictionary"""

    dur = 3
    """Duration of spati-temporal trims to classify. Defaults to 3 seconds."""

    fps = 30
    """Frame rate of videos currently analyzed. It defaults to 30"""

    roi_df = None
    """"Dataframe containing table region of interests"""

    sprop_df = None
    """Dataframe containing session properties."""
    
    
    
    def __init__(self, cfg):
        """ Spatio temporal writing detection using,

        Parameters
        ----------
        cfg : Str
            Configuration file. The configuration has the following entries.
        """

        # Configuration dictionary
        self.cfg = cfg

        # Loading required csv and xlsx files to dataframe
        self.roi_df = pd.read_csv(cfg['roi'])
        self.sprop_df = pd.read_csv(cfg['prop'])

        
    def generate_writing_proposals_using_roi(self, dur=3, fps=30, overwrite = True):
        """ Calculates writing region proposals using ROI.

        It write the output to `wrp_only_roi.csv`

        Parameters
        ----------
        dur : int, optional
            Duraion of each writing proposal
        fps : Frames per second, optional
            Framerate of 
        """

        # Check for writing region proposals file and load it
        wrp_roi_only_loc = f"{self.cfg['oloc']}/wrp_only_roi.csv"
        wrp_roi_only_exists, self.wrp_roi_only = self._read_from_disk(wrp_roi_only_loc)
        
        # if overwrite == False then we check of existing csv with region proposals
        if not overwrite:
            if wrp_roi_only_exists:
                print(f"Reading {wrp_roi_only_loc}")
                return True

        # if overwrite == True, and file exists delete it
        if wrp_roi_only_exists:
            print(f"Deleting {wrp_roi_only_loc}")
            os.remove(wrp_roi_only_loc)

        # Creating writing region proposals csv
        print(f"Creating {wrp_roi_only_loc}")
        self.dur = dur
        self.fps = fps
        vid_names = self.cfg['vids']
        
        # Loop over each video every 3 seconds
        wprop_lst = []
        for vid_name in vid_names:
            
            # Properties of current Video
            T = int(self.sprop_df[self.sprop_df['name'] == vid_name]['dur'].item())
            W = int(self.sprop_df[self.sprop_df['name'] == vid_name]['width'].item())
            H = int(self.sprop_df[self.sprop_df['name'] == vid_name]['height'].item())
            FPS = int(self.sprop_df[self.sprop_df['name'] == vid_name]['FPS'].item())

            # ROI for current video
            roi_df_vid = self.roi_df[self.roi_df['video_names'] == vid_name].copy()

            # Current video duration from session properties
            dur_vid = self.sprop_df[self.sprop_df['name'] == vid_name]['dur'].item()
            vor = pk.Vid(f"{self.cfg['vdir']}/{vid_name}", 'read')
            f0_last = vor.props['num_frames'] - self.dur*self.fps
            vor.close()

            # Loop over each 3 second instance
            for i, f0 in enumerate(range(0, f0_last, self.dur*self.fps)):
                f = self.dur*self.fps
                f1 = f0 + f - 1

                # 3 second dataframe instances
                roi_df_3sec = roi_df_vid[roi_df_vid['f0'] >= f0].copy()
                roi_df_3sec = roi_df_3sec[roi_df_3sec['f0'] <= f1].copy()

                # skip this 3 seconds if there is thare are no rois in then
                # in the interval.
                skip_flag = self._skip_this_3sec_roi_only(roi_df_3sec)

                # If not skipping 
                if not skip_flag:
                    
                    # Get 3 second proposal regions
                    wprop_3sec = self._get_3sec_proposal_df_roi_only(roi_df_3sec.copy())

                    # Adding to prop_lst
                    for wprop_3sec_i in wprop_3sec:
                        wprop_lst_temp = [vid_name, W, H, FPS, T, f0, f, f1] + wprop_3sec_i
                        wprop_lst += [wprop_lst_temp]

        # Creating writing proposal dataframe
        wrp = pd.DataFrame(
            wprop_lst,
            columns=['name', 'W', 'H', 'FPS', 'T', 'f0', 'f', 'f1', 'pseudonym', 'w0', 'h0', 'w', 'h']
        )
        self.wrp_roi_only = wrp
        self.wrp_roi_only.to_csv(wrp_roi_only_loc, index=False)
        return True

    def classify_writing_proposals_roi(self, overwrite=False):
        """ Classify each proposed region as writing / no-writing.

        Todo
        ----
        This function evaluates one proposal at a time. This is note
        optimal. I have to redo this to evaluate multiple proposals
        at a time. 
        """
        # if the file is already present load it if overwrite == True
        out_file = f"{self.cfg['oloc']}/wnw-roi-ours-3DCNN_30fps.csv"
        if not overwrite:
            if os.path.isfile(out_file):
                print(f"Reading {out_file}")
                wdf = pd.read_csv(out_file)
                self.wdf = wdf
                return wdf
        
        # Loading neural network into GPU
        print(f"Creating {out_file}")
        net = self._load_net(self.cfg)

        # Loop through each video
        video_names = self.wrp_roi_only['name'].unique().tolist()
        for i, video_name in enumerate(video_names):

            # Writing proposals for current dataframe
            print(f"Classifying writing in {video_name}")
            wrp_video = self.wrp_roi_only[self.wrp_roi_only['name'] == video_name].copy()
            wrp_video['activity'] = ""
            wrp_video['class_idx'] = -1
            wrp_video['class_prob'] = "-1"
            ivid = pk.Vid(f"{self.cfg['vdir']}/{video_name}", "read")

            # Loop through each instance in the video
            for ii, row in tqdm(wrp_video.iterrows(), total=wrp_video.shape[0], desc="INFO: Classifying"):
                
                # Spatio temporal trim coordinates
                bbox = [row['w0'],row['h0'], row['w'], row['h']]
                sfrm = row['f0']
                efrm = row['f1']
                opth = (f"{self.cfg['oloc']}/temp.mp4")

                # Spatio temporal trim
                ivid.save_spatiotemporal_trim(sfrm, efrm, bbox, opth)

                # Creating a temporary text file `temp.txt` having
                # temp.mp4 and a dummy label (100)
                with open(f"{self.cfg['oloc']}/temp.txt", "w") as f:
                    f.write("temp.mp4 100")
            
                # Intialize AOLME data loader instance
                tst_data = AOLMETrmsDLoader(
                    self.cfg['oloc'], f"{self.cfg['oloc']}/temp.txt", oshape=(224, 224)
                )
                tst_loader = DataLoader(
                    tst_data, batch_size=1, num_workers=1
                )

                # Loop over tst data (??? goes over only once. I am desparate so I kept the loop)
                for data in tst_loader:
                    dummy_labels, inputs = (
                        data[0].to("cuda:0", non_blocking=True),
                        data[1].to("cuda:0", non_blocking=True)
                    )
                    
                    with torch.no_grad():
                        outputs = net(inputs)
                        ipred = outputs.data.clone()
                        ipred = ipred.to("cpu").numpy().flatten().tolist()

                    ipred_class_prob = round(ipred[0], 2)
                    ipred_class_idx = round(ipred_class_prob)
                    if ipred_class_idx == 1:
                        ipred_class = "writing"
                    else:
                        ipred_class = "nowriting"
                        
                    # This is because for 0.5 I am having problems in ROC curve
                    if ipred_class_prob == 0.5:
                        if ipred_class_idx == 1:
                            ipred_class_prob = 0.51
                        else:
                            ipred_class_prob = 0.49

                    wrp_video.at[ii, 'activity'] = ipred_class
                    wrp_video.at[ii, 'class_prob'] = ipred_class_prob
                    wrp_video.at[ii, 'class_idx'] = ipred_class_idx


                # Close the vide
                ivid.close()

                        
            # If this is the first time, copy the proposal dataframe to writing dataframe
            # else concatinate
            if i == 0:
                wdf = wrp_video
            else:
                wdf = pd.concat([wdf, wrp_video])

        # Save the dataframe
        self.wdf = wdf.copy()
        self.wdf.to_csv(f"{out_file}", index=False)
        return wdf


    def classify_proposals_using_hand_det(self, overwrite=False):
        """ Classify each proposed region as writing / no-writing. In 
        this method I use hand detection information to further improve
        performance.

        Parameters
        ----------
        overwrite : bool
            Overwrite existing writing csv file.
        """
        
        # if the file is already present load it if overwrite == True
        out_file = f"{self.cfg['oloc']}/wnw-roi-ours-3DCNN_handdet_30fps.csv"
        if not overwrite:
            if os.path.isfile(out_file):
                print(f"Reading {out_file}")
                wdf = pd.read_csv(out_file)
                self.wdf = wdf
                return wdf
        
        # Loading neural network into GPU
        print(f"Creating {out_file}")
        net = self._load_net(self.cfg)

        # Loop through each video
        video_names = self.wrp_roi_only['name'].unique().tolist()
        for i, video_name in enumerate(video_names):

            # video_name_no_ext
            video_name_no_ext = os.path.splitext(video_name)[0]

            # Loading relavent files
            ivid = pk.Vid(f"{self.cfg['vdir']}/{video_name}", "read")  # Video
            wrp_video = self.wrp_roi_only[self.wrp_roi_only['name'] == video_name].copy()  # Region proposals
            hand_det = pd.read_csv(f"{self.cfg['hand_detdir']}/{video_name_no_ext}_12sec_interval.csv")


            # Loop through each instance in the video
            print(f"Classifying writing in {video_name}")
            wrp_video['activity'] = ""
            wrp_video['class_idx'] = -1
            wrp_video['class_prob'] = "-1"
            for ii, row in tqdm(wrp_video.iterrows(), total=wrp_video.shape[0], desc="INFO: Classifying"):
                
                # Spatio temporal trim coordinates
                bbox = [row['w0'],row['h0'], row['w'], row['h']]
                sfrm = row['f0']
                efrm = row['f1']
                opth = (f"{self.cfg['oloc']}/temp.mp4")



                # Get hand detections inside current bounding box
                bbox_valid_flag = self.is_bbox_valid(bbox, hand_det, sfrm, efrm)

                # ifq the intersection area is less than 25th percentile of hand area we mark the instance as no writing
                if bbox_valid_flag > 0:
                    
                    # If they don't overlap then mark the proposal as nowriting
                    wrp_video.at[ii, 'activity'] = 'nowriting'
                    wrp_video.at[ii, 'class_prob'] = 0.49
                    wrp_video.at[ii, 'class_idx'] = 0
                    
                else:
                    # Spatio temporal trim
                    ivid.save_spatiotemporal_trim(sfrm, efrm, bbox, opth)

                    # Creating a temporary text file `temp.txt` having
                    # temp.mp4 and a dummy label (100)
                    with open(f"{self.cfg['oloc']}/temp.txt", "w") as f:
                        f.write("temp.mp4 100")

                    # Intialize AOLME data loader instance
                    tst_data = AOLMETrmsDLoader(
                        self.cfg['oloc'], f"{self.cfg['oloc']}/temp.txt",
                        oshape=(224, 224)
                    )
                    tst_loader = DataLoader(
                        tst_data, batch_size=1, num_workers=1
                    )

                    # Loop over tst data (??? goes over only once.
                    # I am desparate so I kept the loop)
                    for data in tst_loader:
                        dummy_labels, inputs = (
                            data[0].to("cuda:0", non_blocking=True),
                            data[1].to("cuda:0", non_blocking=True)
                        )

                        with torch.no_grad():
                            outputs = net(inputs)
                            ipred = outputs.data.clone()
                            ipred = ipred.to("cpu").numpy().flatten().tolist()

                        ipred_class_idx = round(ipred[0])
                        if ipred_class_idx == 1:
                            ipred_class = "writing"
                            ipred_class_prob = round(ipred[0], 2)
                        else:
                            ipred_class = "nowriting"
                            ipred_class_prob = round(ipred[0], 2)

                        # This is because for 0.5 I am having problems in ROC
                        # curve
                        if ipred_class_prob == 0.5:
                            if ipred_class_idx == 1:
                                ipred_class_prob = 0.51
                            else:
                                ipred_class_prob = 0.49

                        wrp_video.at[ii, 'activity'] = ipred_class
                        wrp_video.at[ii, 'class_prob'] = ipred_class_prob
                        wrp_video.at[ii, 'class_idx'] = ipred_class_idx
                        


                # Close the vide
                ivid.close()

                        
            # If this is the first time, copy the proposal dataframe to writing
            # dataframe else concatinate
            if i == 0:
                wdf = wrp_video
            else:
                wdf = pd.concat([wdf, wrp_video])

        # Save the dataframe
        self.wdf = wdf.copy()
        self.wdf.to_csv(f"{out_file}", index=False)
        return wdf

    def is_bbox_valid(self, bbox, hand_det, sfrm, efrm):
        """Returns True if the current bounding box is valid. Valid implies that
        we have atleast one hand detection inside the bounding box that has 0.5 
        IoU.
        
        Parameters
        ----------
        bbox : 
            Regin proposal bounding box. 
            The bounding box has the following coordinates, ['w0','h0', 'w', 'h']
        hand_det : 
            Hand detection dataframe
        sfrm :
            Starting frame
        efrm :
            Ending frame
        """

        # Width and height of image
        hdf = hand_det.copy()
        W = hdf['W'].unique().item()
        H = hdf['H'].unique().item()

        # Get hand detections within the starting and ending frames
        hdf = hdf[hdf['f0'] <= sfrm].copy()  # Detection that start before or equal to the starting frame.
        hdf = hdf[hdf['f0'] + hdf['f'] > efrm].copy()  # Hand detections that end after or equal to the ending frame.

        # If we do not have any hand detections we mark the proposal as invalid
        if len(hdf) == 0:
            return False

        # There should be a maximum of one row in hdf
        if len(hdf) > 1:
            import pdb; pdb.set_trace()

        # Loop though each hand detection.
        dets = hdf['props'].item().split(':')
        dets.remove('')

        for det in dets:
            det = [int(x) for x in det.split('-')]
            IoU = self._get_iou_using_image(det, bbox, W, H)
            if IoU >= 0.5:
                return True
        
        # There are no hand detections in the proposal region that have 0.5 IoU. Hence  the current
        # proposal is invalid
        return False


    def _get_iou_using_image(self, bbox1, bbox2, W, H):
        """ Returns IoU score for bounding boxes.

        Parameters
        ----------
        bbox1 : list[int]
            [x_tl, y_tl, x_w, y_h]
        bbox2 : list[int]
            [x_tl, y_tl, x_w, y_h]
        """
        # Changing bboxes to [x_tl, y_tl, x_br, y_bl]
        bbox1 = [bbox1[0], bbox1[1], bbox1[0]+bbox1[2], bbox1[1]+bbox1[3]]
        bbox2 = [bbox2[0], bbox2[1], bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]]

        # Creating two images with bounding boxes marked as 1
        img1 = np.zeros((H, W))
        img2 = np.zeros((H, W))
        img1[ bbox1[1] : bbox1[3], bbox1[0] : bbox1[2] ] = 1
        img2[ bbox2[1] : bbox2[3], bbox2[0] : bbox2[2] ] = 1

        # Intersection image
        imgi = img1*img2
        pixi = imgi.sum()

        # Union image
        imgu = 1*((img1 + img2) >= 1)
        pixu = imgu.sum()

        # IoU
        iou = pixi/pixu

        return iou
         















    def _get_hand_det_intersection(self, hand_det):
        """Determines hand detection intersection"""

        df = hand_det.copy()

        # If we do not have any detection we will send [0, 0, 0, 0]
        if len(df) == 0:
            return 0, [0, 0, 0, 0]
        
        df['w1'] = df['w0'] + df['w']
        df['h1'] = df['h0'] + df['h']

        # Top left intersection coordinates
        tl_w = max(df['w0'].tolist())
        tl_h = max(df['h0'].tolist())

        # Bottom right intersection coordinates
        br_w = min(df['w1'].tolist())
        br_h = min(df['h1'].tolist())
        w = br_w - tl_w
        h = br_h - tl_h

        return w*h, [tl_w, tl_h, w, h]
        
        

    def _get_kbdet_intersection(self, kb_det, sfrm, efrm):
        """

        WARNING:
        --------
        DEPRICATION THIS METHOD IS USED FOR TYPING, IT IS NO LONGER USED IN WRITING.

        Determines hand detection intersectin bouhnding box between
        sfrm and efrm"""

        # Snipping hand detection dataframe between sfrm and efrm
        kdf = kb_det.copy()
        kdf = kdf[kdf['f0'] >= sfrm].copy()
        kdf = kdf[kdf['f0'] <= efrm].copy()

        # If we do not have any detection we will send [0, 0, 0, 0]
        if len(kdf) == 0:
            return [0, 0, 0, 0]
        
        kdf['w1'] = kdf['w0'] + kdf['w']
        kdf['h1'] = kdf['h0'] + kdf['h']

        # Top left intersection coordinates
        tl_w = max(kdf['w0'].tolist())
        tl_h = max(kdf['h0'].tolist())

        # Bottom right intersection coordinates
        br_w = min(kdf['w1'].tolist())
        br_h = min(kdf['h1'].tolist())
        w = br_w - tl_w
        h = br_h - tl_h

        return [tl_w, tl_h, w, h]
    
    def _read_from_disk(self, wrp_csv):
        """Load the file if it exists"""
        if os.path.isfile(wrp_csv):
            wrp = pd.read_csv(wrp_csv)
            return True, wrp
        else:
            return False, None

    def _get_3sec_proposal_df(self, roi_df, kdf):
        """Returns a dataframe with writing region proposals using
        1. Table ROI
        2. Hand detections

        Parameters
        ----------
        roi_df : Pandas Dataframe
            Table ROI for 3 seconds
        kdf : Pandas Dataframe
            Hand detection for 3 seconds
        """

        # ROI column names (persons sitting around the table)
        roidf_temp = roi_df.copy()
        roidf_temp = roidf_temp.drop(['Time', 'f0', 'f', 'video_names'], axis=1)
        persons_list = roidf_temp.columns.tolist()
        
        # Loop over each person ROI and check for hand detection
        prop_lst = []
        roi_coords_lst = []  # ROI coordinates list
        iarea_lst = [] # Intersection area
        roi_person_lst = []
        for person in persons_list:
            
            # Process further only if ROIs are available for more than
            # 1/2 the duration of the time
            roi_lst = [str(x) for x in roi_df[person].tolist()]
            num_roi = len(roi_lst) - np.sum([1*(x == 'nan') for x in roi_lst])
            
            if num_roi > math.floor(self.dur/2):

                # Adding person to person list
                roi_person_lst += [person]

                # Get ROIs intersection bounding box
                roi_coords = self._get_roi_intersection(roi_df, person)
                roi_coords_lst += [roi_coords]
                
                # Get Table detectin intersectin bounding box
                det_coords = self._get_det_intersection(kdf)

                # Intersection area between roi_coords and det_coords
                iarea = self._get_intersection_area(roi_coords, det_coords)
                iarea_lst += [iarea]

        # Return Table ROI coordinates that has the highest intersection area
        if max(iarea_lst) > 0:
            max_iarea_roi_coords = roi_coords_lst[iarea_lst.index(max(iarea_lst))]
            max_iarea_person = roi_person_lst[iarea_lst.index(max(iarea_lst))]
            prop_lst += [max_iarea_person]
            prop_lst += max_iarea_roi_coords
            return [prop_lst]
        
        return []


    def _get_intersection_area(self, roi, det):
        """Returns intersection area between roi coordinates and
        detection.
        
        Parameters
        ----------
        roi : List[int]
            Region of Interest coordinates
        det : List[int]
            Detection coordinates
        """
        iflag, icoords = self._get_intersection(roi, det)
        return icoords[2]*icoords[3]
    
    def _get_3sec_proposal_df_roi_only(self, roi_df):
        """Returns a dataframe with writing region proposals using
        1. Table ROI

        Parameters
        ----------
        roi_df : Pandas Dataframe
            Table ROI for 3 seconds
        """

        # ROI column names (persons sitting around the table)
        roidf_temp = roi_df.copy()
        roidf_temp = roidf_temp.drop(['Time', 'f0', 'f', 'video_names'], axis=1)
        persons_list = roidf_temp.columns.tolist()
        
        # Loop over each person ROI and check for hand detection
        prop_lst = []
        for person in persons_list:
            
            # Process further only if ROIs are available for more than
            # 1/2 the duration of the time
            roi_lst = [str(x) for x in roi_df[person].tolist()]
            num_roi = len(roi_lst) - np.sum([1*(x == 'nan') for x in roi_lst])
            
            if num_roi > math.floor(self.dur/2):

                # Get ROIs intersection bounding box
                roi_coords = self._get_roi_intersection(roi_df, person)
                prop_lst += [
                    [person, roi_coords[0], roi_coords[1], roi_coords[2], roi_coords[3]]
                ]
                
        return prop_lst

    
    def _get_det_intersection(self, det_df):
        """Returns intersection of detection coordinates

        Parameters
        ----------
        det_df : Pandas DataFrame
            Hand detection dataframe for current duration.
        """
        detdf_temp = det_df.copy()

        det_i = []
        for i, row in detdf_temp.iterrows():
            
            # First time load the bonding box
            if len(det_i) == 0:
                det_i = [row['w0'], row['h0'], row['w'], row['h']]

            else:
                det_c = [row['w0'], row['h0'], row['w'], row['h']]
                overlap_flag, det_i = self._get_intersection(det_i, det_c)

                # If not intersecting update to current coordinates
                if not overlap_flag:
                    det_i = det_c
                else:
                    det_i = det_i
                
        return det_i
    

    def _get_roi_intersection(self, roi_df, person):
        """Returns intersection ROI coordinates

        Parameters
        ----------
        df : pandas DataFrame
            ROI dataframe
        person : Str
            The name of the person currently under consideration
        """
        rois = [str(x) for x in roi_df[person].tolist()]

        roi_i = []
        for roi in rois:
            
            if not roi == 'nan':
                roi = [int(x) for x in roi.split('-')]
                
                if len(roi_i) < 1:
                    roi_i = roi

                else:
                    # Get intersection coordinates of two boxes
                    overlap_flag, roi_i = self._get_intersection(roi_i, roi)
                    
                    # If not intersecting update to current coordinates
                    if not overlap_flag:
                        roi_i = roi
                    else:
                        roi_i = roi_i
                        
        return roi_i



    def _get_intersection(self, rect1, rect2):
        """Get intersectin of two rectangles based on image coordinate
        system

        Parameters
        ----------
        rect1 : List[int]
            Coordinates of one bounding box
        rect2 : List[int]
            Coordinates of second bounding box.
        """
        
        # Intersection box coordinates
        wp_tl = rect1[0]
        hp_tl = rect1[1]
        wp_br = wp_tl + rect1[2]
        hp_br = hp_tl + rect1[3]

        # Current bounding box coordinates
        wc_tl = rect2[0]
        hc_tl = rect2[1]
        wc_br = wc_tl + rect2[2]
        hc_br = hc_tl + rect2[3]

        # Intersection
        wi_tl = max(wp_tl, wc_tl)
        hi_tl = max(hp_tl, hc_tl)
        wi_br = min(wp_br, wc_br)
        hi_br = min(hp_br, hc_br)

        # Updating the intersection
        if (wi_br - wi_tl <= 0) or (hi_br - hi_tl <= 0):

            return False, [0, 0, 0, 0]

        else:

            # If overlapping update to intersection
            intersection_coords = [wi_tl, hi_tl, wi_br-wi_tl, hi_br-hi_tl]
            return True, intersection_coords
            



    def _skip_this_3sec_roi_only(self, roidf):
        """Skip the current 3 second interval if
        1. We do not have table region of interest
        2. ROI should be available for more than half of the duration.

        Parameters
        ----------
        roidf : Pandas DataFrame instance
            ROI dataframe

        Returns
        -------
        bool
            True  = skip the current 3 seconds
            False = Do not skip the current 3 seconds.
        """
        roidf_temp = roidf.copy()

        # Return True if there are no ROI regions
        roidf_temp = roidf_temp.drop(['Time', 'f0', 'f', 'video_names'], axis=1)
        if roidf_temp.isnull().values.all():
            return True

        # There should be atleast one Table ROI  for 2 seconds or more.
        # If not we will skip analyzing the current 3 seconds
        for col_name in roidf_temp.columns.tolist():
            cur_col = [str(x) for x in roidf_temp[col_name].tolist()]
            num_nan = np.sum([1*(x == 'nan') for x in cur_col])
            if num_nan > math.floor(self.dur/2):
                continue
            else:
                return False

        return True

        
    def _skip_this_3sec(self, roidf, detdf):
        """Skip the current 3 second interval if
        1. We do not have table region of interest or hand detection
        2. Hand detections should be available for more than half
           of the duration.
        3. ROI should be available for more than half of the duration.

        Parameters
        ----------
        roidf : Pandas DataFrame instance
            ROI dataframe
        detdf : Pandas DataFrame instance
            Hand detection instances

        Returns
        -------
        bool
            True  = skip the current 3 seconds
            False = Do not skip the current 3 seconds.
        """
        roidf_temp = roidf.copy()
        detdf_temp = detdf.copy()

        # Return True if there are no ROI regions
        roidf_temp = roidf_temp.drop(['Time', 'f0', 'f', 'video_names'], axis=1)
        if roidf_temp.isnull().values.all():
            return True

        # Return True if there is no hand detection. If we don't
        # detect hand we keep widht and heigh to 0.
        w_sum = detdf_temp['w'].sum()
        if w_sum <= 0:
            return True

        # There should be atleast two hand detections, otherwise
        # we skip analyzing the current 3 seconds
        w_lst = detdf_temp['w'].tolist()
        num_zeros = sum([1*(x==0) for x in w_lst])
        if num_zeros > math.floor(self.dur/2):
            return True

        # There should be atleast one Table ROI  for 2 seconds or more.
        # If not we will skip analyzing the current 3 seconds
        for col_name in roidf_temp.columns.tolist():
            cur_col = [str(x) for x in roidf_temp[col_name].tolist()]
            num_nan = np.sum([1*(x == 'nan') for x in cur_col])
            if num_nan > math.floor(self.dur/2):
                continue
            else:
                return False

        return True
        




                    
    def _get_union_of_hand_detections(self, df, f0, f):
        """ Returns hand detection regions using union of all the detections
        in an interval.
        
        Parameters
        ----------
        df : DataFrame
            A DataFrame having hand detections.

        Returns
        -------
        Returns list of new rows with following columns
        [W, H, FPS, f0, f, class, table_boundary, w0, h0, w, h]
        """
        # Flag to turn off/on images
        show_images = False
        
        # Creating an image with zeros
        W = df['W'].unique().item()
        H = df['H'].unique().item()
        FPS = df['FPS'].unique().item()
        oclass = "hand"
        table_boundary = df['table_boundary'].unique().item()
        uimg = np.zeros((H, W ))

        # Creating a binary image that is union of all the bounding boxes.
        for i, r in df.iterrows():
            [w0, h0, w, h] = [r['w0'], r['h0'], r['w'], r['h']]
            uimg[h0 : h0 + h, w0 : w0 + w] = 1

        if show_images:
            plt.subplot(111)
            ax = plt.subplot(1, 1, 1)
            ax.imshow(uimg, cmap='gray')
            plt.show()

        # Connected components
        cc = cv2.connectedComponentsWithStats(uimg.astype('uint8'), 4, cv2.CV_32S)
        cc_img = cc[1]
        cc_labels = np.unique(cc_img).tolist()

        # Loop through each label and find bounding box coordinates
        new_rows = []
        for cc_label in cc_labels[1:]:
            
            new_row = [W, H, FPS, f0, f, oclass, table_boundary]

            cc_label_img = 1*(cc_img == cc_label).astype('uint8')
            active_px = np.argwhere(cc_label_img!=0)
            active_px = active_px[:,[1,0]]
            w0,h0,w,h = cv2.boundingRect(active_px)

            new_row += [w0, h0, w, h]

            new_rows += [new_row]

        return new_rows

    
    def _remove_outside_hand_detections(self, df, th=0.5):
        """ Removes all the hand detections that are less that are
        50% not inside the table boundary.

        Parameters
        ----------
        df : DataFrame
            DataFrame having hand detections with `roi-overlap-ratio`
            column.
        th : Float
            Detectons which are < th are removed.
        """
        for ridx, row in df.iterrows():
            if row['roi-overlap-ratio'] < th:
                df.drop([ridx], inplace = True)
        return df

    
    def _get_roi_overlap_ratio(self, hdf, table_boundary):
        """ Adds a column to hand detections, roi-overlap-ratio.
            
            - Table boundary = T
            - Hand detection = H
            - Overlap (O) = Intersection(T, H)
                overlap-ratio = Area(O) / Area(H)

        Parameters
        ----------
        hdf : DataFrame
            Dataframe having hand detections

        table_boundary : List[Int]
            Table boundary as list, [w0, h0, w, h]
        """
        # Flag to turn off/on images
        show_images = False
        
        # Uncompressing boundary
        [tw0, th0, tw, th] = table_boundary
        
        # Creating an image with zeros
        W = hdf['W'].unique().item()
        H = hdf['H'].unique().item()
        zimg = np.zeros((H,W))

        # Creating a binary image with table boundary marked as 1s
        timg = zimg.copy()
        timg[th0 : th0 + th, tw0 : tw0 + tw] = 1

        # Loop over each hand detection
        o_area_ratio_lst = []
        for ridx, row in hdf.iterrows():

            # hand detection is loaded into proper variables
            [hw0, hh0, hw, hh] = [row['w0'], row['h0'], row['w'], row['h']]
            h_area = hw*hh

            # Creating a binary image with hand detection as 1
            himg = zimg.copy()
            himg[hh0 : hh0 + hh, hw0 : hw0 + hw] = 1

            # Overlap image
            oimg = himg + timg
            oimg = 1*(oimg == 2)
            o_area = oimg.sum()

            if show_images:
                plt.subplot(221)
                ax1 = plt.subplot(2, 2, 1)
                ax2 = plt.subplot(2, 2, 2)
                ax3 = plt.subplot(2, 2, 3)
                ax1.imshow(timg)
                ax2.imshow(himg)
                ax3.imshow(oimg)
                plt.show()
                import pdb; pdb.set_trace()

            # Drop the hand detection if the overlap area is less than
            # 50% of hand area
            o_area_ratio = o_area/h_area
            o_area_ratio_lst += [o_area_ratio]
            
        hdf['table_boundary'] = f"{tw0}-{th0}-{tw}-{th}"
        hdf['roi-overlap-ratio'] = o_area_ratio_lst

        return hdf

    
    def _have_sufficient_rois(self, df):
        """ Returns true if there there is a vlaid region of interest of atleast one
        student. A student having atleast tow rois out of three is considered valid.

        Parameters
        ----------
        df : DataFrame
            A data frame having roi entries for `self.wdur`.
        """

        # Dropping unnecessary columns
        df.drop(['Time', 'f0', 'video_names', 'f'], axis = 1, inplace=True)

        # Looping over each column and if atleast one column contain 2 valid entries
        # return true
        for col in df.columns.tolist():
            
            valid_bboxes = 0
            for i in range(0, self.wdur):

                # Bounding box in current column
                bbox_coords = df[col].iloc[i]

                # A bounding box is valid if it has four coordinates
                # separated by "-"
                if len(bbox_coords.split("-")) == 4:
                    valid_bboxes += 1

                # If we are able to get more than 2 valid bounding
                # boxes return True
                if valid_bboxes >= 2:
                    return True

        # If thre are no valid bounding boxes return False
        return False

    
    def _get_table_boundary(self, df):
        """ Return table boundary as list, [w0, h0, w, h]

        Parameters
        ----------
        df : DataFrame
            A data frame having roi entries for `self.wdur`.
        """

        # Dropping unnecessary columns
        df.drop(['Time', 'f0', 'video_names', 'f'], axis = 1, inplace=True)

        # Creating list of bounding box coordinates
        w_min = math.inf
        h_min = math.inf
        w_max = 0
        h_max = 0
        for col in df.columns.tolist():
            for ridx in range(0, len(df[col])):
                
                [w0, h0, w, h] = [int(x) for x in df[col].iloc[ridx].split('-')]

                if w_min > w0:
                    w_min = w0
                if h_min > h0:
                    h_min = h0
                if w_max < w0 + w:
                    w_max = w0 + w
                if h_max < h0 + h:
                    h_max = h0 + h

        boundary = [w_min, h_min, w_max - w_min, h_max - h_min]

        return boundary


        
    def _load_net(self, cfg):
        """ Load neural network to GPU. """

        print("INFO: Loading Trained network to GPU ...")

        # Checkpoint and depth from cfg file.
        ckpt = cfg['ckpt']
        depth = cfg['depth']

        # Creating an instance of Dyadic 3D-CNN
        net = DyadicCNN3D(depth, [3, 90, 224, 224])
        net.to("cuda:0")

        # Print summary of network.
        # summary(net, (3, 90, 224, 224))

        # Loading the net with trained weights to cuda device 0
        ckpt_weights = torch.load(ckpt)
        net.load_state_dict(ckpt_weights['model_state_dict'])
        net.eval()

        return net
        
        

    def _check_for_writing(self, proposal_df):
        """ Checks for writing in the proposal data frame.

        Parameters
        ----------
        proposal_df: DataFrame
            Proposal dataframe having hands bounding boxes.
        Todo
        ----
        1. Here I am trimming -> writing to hdd -> loading. This is not
           recommended for speed. Please try to improve.
        """
        import pdb; pdb.set_trace()
        # Loop over proposal dataframe
        for i, row in proposal_df.iterrows():

            # if w or h == 0 then there is no hands
            if row['w'] == 0 or row['h'] == 0:
                proposal_df.at[i, 'writing'] = -1
            else:
                # Creating temporal trim
                bbox = [row['w0'],row['h0'], row['w'], row['h']]
                sfrm = row['f0']
                efrm = sfrm + row['f']
                oloc = f"{os.path.dirname(self._vid.props['dir_loc'])}"
                opth = (f"{oloc}/temp.mp4")

                # Spatio temporal trim
                self._vid.spatiotemporal_trim(sfrm, efrm, bbox, opth)
                 
                # Creating a temporary text file `temp.txt` having
                # temp.mp4 and a dummy label (100)
                with open(f"{oloc}/temp.txt", "w") as f:
                    f.write("temp.mp4 100")

                # Intialize AOLME data loader instance
                tst_data = AOLMETrmsDLoader(oloc, f"{oloc}/temp.txt", oshape=(224, 224))
                tst_loader = DataLoader(tst_data, batch_size=1, num_workers=1)

                # Loop over tst data (goes over only once. I am desparate so I kept the loop)
                for data in tst_loader:
                    dummy_labels, inputs = (data[0].to("cuda:0", non_blocking=True),
                                             data[1].to("cuda:0", non_blocking=True))
                    with torch.no_grad():
                        outputs = self._net(inputs)
                        ipred = outputs.data.clone()
                        ipred = ipred.to("cpu").numpy().flatten().tolist()
                        proposal_df.at[i, 'writing'] = round(ipred[0])
                        
        return proposal_df



    def _get_proposal_df(self, bboxes, wdur):
        """
        OBSOLETE SHOULD BE DELETED IN CLEANUP PHASE.
        Creates a data frame with each row representing 3 seconds.

        Parameters
        ----------
        bboxes: str
            path to file having hands bounding boxes
        wdur: int, optional
            Each writing instance duration considered in seconds. 
            Defaults to 3.
        """
        # Video properties
        num_frames = self._vid.props['num_frames']
        fps = self._vid.props['frame_rate']

        # Creating f0 and f lists
        num_trims = math.floor(num_frames/(wdur*fps))
        f0 = [x*(wdur*fps) for x in range(0, num_trims)]
        f = [wdur*fps]*num_trims

        # Creating W, H and FPS lists
        W = [self._vid.props['width']]*num_trims
        H = [self._vid.props['height']]*num_trims
        fps_lst = [fps]*num_trims

        # Get bounding boxes
        w0, h0, w, h = self._get_proposal_bboxes(bboxes, f0, f)

        # Intializing all writing instances are marked nan(numpy)
        writing_lst = [np.nan]*(num_trims)

        # Creating data frame with all the lists
        df = pd.DataFrame(list(zip(W, H, fps_lst, f0, f, w0, h0, w, h, writing_lst)),
                          columns=["W","H", "FPS", "f0", "f", "w0", "h0", "w", "h", "writing"])
        return df


    
    def write_to_csv(self):
        """ Writes writing instances to a csv file. The name of the file is `<video name>_wr_using_alg.csv` and has
        following columns,

            1. f0      : poc of starting frame
            2. f       : number of frames
            3. W, H    : Video width and height
            4. w0, h0  : Bounding box top left corner
            5. w, h    : width and height of bounding box
            6. FPS     : Frames per second
            7. writing : {-1, 0, 1}.
                -1 => Hands not found
                0  => nowriting
                1  => writing

        """
        # Update writing instances in writing dataframe by processing
        # valid instances to 0 or 1
        self.wdf = self._check_for_writing(self._proposal_df.copy())
        
        vname = self._vid.props['name']
        vloc = self._vid.props['dir_loc']
        csv_pth = f"{vloc}/{vname}_wr_using_alg.csv"
        self.wdf.to_csv(csv_pth)
        

        
    def _get_proposal_bboxes(self, bboxes, f0_lst, f_lst):
        """ Creates proposal bounding boxes. Trims from these bounding boxes are
        later processed via writing detection algorithm.

        If there are multiple bounding boxes in the duration we consider the
        union of bounding boxes.

        Parameters
        ----------
        bboxes: str
            Path to file having hands detection bounding boxes
        f0_lst: List of int
            List having starting frame poc
        f_lst: List of int
            List having poc lenght
        """
        df_bb = pd.read_csv(bboxes)
        num_trims = len(f0_lst)
        
        w0_lst = [0]*num_trims
        h0_lst = [0]*num_trims
        w_lst = [0]*num_trims
        h_lst = [0]*num_trims
        
        for i in range(0, num_trims):
            f0 = f0_lst[i]
            f = f_lst[i]

            # Data frame having detections from f0 to f0+f
            df_bbi = pd.DataFrame()
            df_bbi = df_bb[df_bb['f0'] >= f0].copy()
            df_bbi = df_bbi[df_bbi['f0'] < f0 + f]
            if len(df_bbi) > 0:
                w0i = df_bbi['w0'].tolist()
                h0i = df_bbi['h0'].tolist()
                wi = df_bbi['w'].tolist()
                hi = df_bbi['h'].tolist()
                w1i = [sum(x) for x in zip(w0i, wi)]
                h1i = [sum(x) for x in zip(h0i, hi)]

                # Taking union
                w0_tl = min(w0i)
                h0_tl = min(h0i)
                w0_br = max(w1i)
                h0_br = max(h1i)

                # Adding to the list
                w0_lst[i] = w0_tl
                h0_lst[i] = h0_tl
                w_lst[i] = w0_br - w0_tl
                h_lst[i] = h0_br - h0_tl
        return (w0_lst, h0_lst, w_lst, h_lst)
