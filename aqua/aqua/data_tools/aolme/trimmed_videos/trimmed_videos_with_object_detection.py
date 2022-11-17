import os
import sys
import pdb
import aqua
import random
import shutil
import numpy as np
import pandas as pd
import json
import math
import re
import cv2
import pytkit as pk
from tqdm import tqdm
import shutil
from operator import add

class ObjDetGTTrims:

    cfg = None
    """ Configuration dictionary having required information """

    df = None
    """ DataFrame that gets updated with 3 second activity instaces and
    bounding box coordinates derived from object detection """

    
    def __init__(self, cfg):
        """Generates trims to train a activity classifier.

        In addition to ground truth it uses object detection, keyboard
        and hand, to trim the videos that are representative of
        testing.

        Parameters
        ----------
        cfg : str
            JSON configuration file having the following information.
        """
        
        # Load json file to class object
        with open(cfg, 'r') as f:
            cfg_dict = json.load(f)
            self._verify_cfg_dict(cfg_dict)
            self.cfg = cfg_dict

        # Loading activity dataframe
        self.df = pd.read_csv(self.cfg['gt_csv_path'])
            

    def _verify_cfg_dict(self, cfg):
        """Verify configuration dictionary

        Parameters
        ----------
        cfg : Dict
            A dictionary having current trimming session configuration
        """



        # Checking videos
        if not os.path.isdir(cfg['videos_dir']):
            sys.exit(f"ERROR: Directory {cfg['videos_dir']} does not exist")
            
        video_names = cfg['video_names']
        for video_name in video_names:
            vpath = f"{cfg['videos_dir']}/{video_name}"
            if not os.path.isfile(vpath):
                sys.exit(f"ERROR: Video {vpath} does not exist")

        # Checking if the activity is supported. For now supports typing and writing
        if not cfg['activity'] in ["typing", "writing"]:
            sys.exit(f"ERROR: Activity {cfg['activity']} is not supported")

        # Checking for ground truth file
        if not os.path.isfile(cfg['gt_csv_path']):
            sys.exit(f"ERROR: Cannot find ground truth, {cfg['gt_csv_path']}")

        if not os.path.isdir(cfg['obj_det_dir']):
            sys.exit(f"ERROR: Cannot find obj detection directory, {cfg['obj_det_dir']}")
        obj_dets = cfg['obj_det_csvs']
        for obj_det in obj_dets:
            obj_det_path = f"{cfg['obj_det_dir']}/{obj_det}"
            if not os.path.isfile(obj_det_path):
                sys.exit(f"ERROR: cannot find obj detection file, {obj_det_path}")

        # Checking if table roi files are valid
        if not os.path.isdir(cfg['table_roi_dir']):
            sys.exit(f"ERROR: Cannot find directory, {cfg['table_roi_dir']}")
        roi_fpath = f"{cfg['table_roi_dir']}/{cfg['table_roi_csv']}"
        if not os.path.isfile(roi_fpath):
            sys.exit(f"ERROR: Cannot find file, {roi_fpath}")


    def get_3sec_activity_instance(self):
        """Saves a CSV file with 3 second activity instances.

        These activity instances are derived from middle of ground truth activity
        instances. The newly calculated instances are added as column to the
        class dataframe under the name, `f0_3sec`.
        """

        # copying class object dataframe temporarly to outside
        df = self.df.copy()

        # Loop through each instance and add f0_3sec column.
        # f0_3sec = f0 + f/2 - 45
        f0_3sec_list = []
        for i, row in df.iterrows():
            f0 = row['f0']
            f = row['f']
            f0_3sec_list += [math.floor(f0 + f/2 - 45)]

        # Add f0_3sec_list column to the class object dataframe
        self.df['f0_3sec'] = f0_3sec_list
        

    def get_obj_bounding_boxes(self):
        """Generates bounding box coordinates in the 3sec interval.

        The bounding box is derived by considering object detections
        in the 3 second intervals. The new bounding box coordinates
        are added to the class DataFrame column, `obj_bbox`.

        if the object is not located in the 3 second region, we
        mark this by `0-0-0-0` value.
        """

        # Copying activity instances to a temporary dataframe
        df = self.df.copy()

        # Loading object detections csvs into one dataframe
        obj_det_dir = self.cfg['obj_det_dir']
        for i, obj_csv in enumerate(self.cfg['obj_det_csvs']):
            video_name = re.sub("_30fps_.*?csv","", obj_csv, flags=re.DOTALL) + "_30fps.mp4"
            obj_det_csv_path = f"{obj_det_dir}/{obj_csv}"
            if i == 0:
                obj_df = pd.read_csv(obj_det_csv_path)
                obj_df['video_name'] = video_name
            else:
                temp_df = pd.read_csv(obj_det_csv_path)
                temp_df['video_name'] = video_name
                obj_df = pd.concat([obj_df, temp_df])

        # Loop through each activity instance
        obj_bbox_list = []
        for i, row in df.iterrows():

            # Filtering object detection dataframe.
            fdf = obj_df[obj_df['video_name'] == row['name']].copy()
            fdf = fdf[fdf['f0'] >= row['f0_3sec']].copy()
            fdf = fdf[fdf['f0'] < row['f0_3sec'] + 90].copy()



            # Removing all object detections that do not occupy atleaset
            # 50% of ground truth activity bounding boxes
            fdf = self._remove_distant_objects(fdf, row)

            # If there are no object instances in current activity
            # make the object bounding box to 0-0-0-0
            if fdf.empty:
                obj_bbox_list += [f"0-0-0-0"]
                continue

            # Calculating bottom right coordinates, w1 and h1
            fdf['w1'] = fdf['w0'] + fdf['w']
            fdf['h1'] = fdf['h0'] + fdf['h']

            # Calculating the box coordinates that encompasses all the
            # bounding boxes
            w0_ = fdf['w0'].min()
            h0_ = fdf['h0'].min()
            w1_ = fdf['w1'].max()
            h1_ = fdf['h1'].max()
            w_ = w1_ - w0_
            h_ = h1_ - h0_

            # Append to bounding box list
            obj_bbox_list += [f"{w0_}-{h0_}-{w_}-{h_}"]

            # Visually verifying
            if False:
                W = fdf['W'].unique().item()
                H = fdf['H'].unique().item()
                img = 255*np.ones((H,W,3))
                
                for i, row in fdf.iterrows():
                    tl = (row['h0'], row['w0'])
                    br = (row['h0'] + row['h'], row['w0'] + row['w'])
                    img = cv2.rectangle(img, tl, br, (255, 0, 0), 1)
                tl = (h0_, w0_)
                br = (h1_, w1_)
                img = cv2.rectangle(img, tl, br, (255, 255, 0), 2)
                cv2.imshow('image', img)
                cv2.waitKey(0)

        # Append the object detection bounding box as another column
        df['obj_bbox'] = obj_bbox_list
        self.df = df

    def get_roi_bounding_boxes(self):
        """Generates roi coordinates in the 3sec interval.

        The roi coordinates is derived by considering roi annotations
        in the 3 second intervals. The new bounding box coordinates
        are added to the class DataFrame column, `roi_bbox`.

        if the roi is not annotatied in the 3 second region, we
        mark this by `0-0-0-0` value.
        """

        # Copying activity instances to a temporary dataframe
        df = self.df.copy()

        # Loading roi dataframe into one dataframe
        table_roi_dir = self.cfg['table_roi_dir']
        table_roi_csv = self.cfg['table_roi_csv']
        roi_df = pd.read_csv(f"{table_roi_dir}/{table_roi_csv}")

        # Loop through each activity instance
        roi_bbox_list = []
        roi_iou_list = []
        for i, row in df.iterrows():

            # Activity bounding box
            w0 = row['w0']
            h0 = row['h0']
            w = row['w']
            h = row['h']

            # Filtering object detection dataframe.
            fdf = roi_df[roi_df['video_names'] == row['name']].copy()
            fdf = fdf[fdf['f0'] >= row['f0_3sec']].copy()
            fdf = fdf[fdf['f0'] < row['f0_3sec'] + 90].copy()
            
            # If there are no bounding boxes we mark it as 0-0-0-0.
            # Typically the datraframe is not empty
            if fdf.empty:
                roi_bbox_list += [f"0-0-0-0"]
                roi_iou_list +=[0.25]
                continue
            
            # Removing rois that are far from the activity instance. If all the ROIs
            # give less than or equal to 0.25 IoU we will ignore that instance by
            # marking its coordinate as "0-0-0-0" and IoU as 0.25
            [w0_, h0_, w_, h_], roi_iou_ = self._get_best_roi(fdf, row)

            # Append to bounding box list
            roi_bbox_list += [f"{w0_}-{h0_}-{w_}-{h_}"]
            roi_iou_list += [roi_iou_]

            # Visually verifying
            if False:
                print(f"roi = {w0_}-{h0_}-{w_}-{h_}")
                print(f"activity bbox = {w0}-{h0}-{w}-{h}")
                W = 858
                H = 480
                img = 255*np.ones((H,W,3))
                # ROI bbox
                tl = (int(h0_), int(w0_))
                br = (int(h0_) + int(h_), int(w0_) + int(w_))
                img = cv2.rectangle(img, tl, br, (255, 0, 0), 1)
                # G.T.
                tl = (int(h0), int(w0))
                br = (int(h0) + int(h), int(w0) + int(w))
                img = cv2.rectangle(img, tl, br, (0, 255, 0), 2)
                cv2.imshow('Green == G.T., ROIs == Blue', img)
                cv2.waitKey(0)

        # Append the object detection bounding box as another column
        df['roi_bbox'] = roi_bbox_list
        df['roi_iou'] = roi_iou_list
        self.df = df

    def trim_bboxes(self):
        """Extract three second trims.

        For each activity instace we ectract two trims,
        1. using the bounding box from ground truth
        2. using the bounding box from object detection. If we do not
           not find valid object instances in the location, we used
           the coordinates from activity instance.
        3. Using bounding boxes from ROIs.
        """

        # Clean output directory
        odir = self.cfg['outdir']
        shutil.rmtree(odir)

        # loading activity instace dataframe to a temporary one
        df = self.df

        # looping through each activity instace
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):

            # Loading the video object pytkit
            vpth = self.cfg['videos_dir'] + "/" + row['name']
            vo = pk.Vid(vpth, 'read')

            # trim using ground truth bounding box
            self._trim_gt_bbox(vo, row)

            # trim using object detections
            self._trim_obj_bbox(vo, row)

            # trim using table region of interests
            self._trim_roi_bbox(vo, row)

            # Destroying the video object
            vo.close()
            

    def _trim_gt_bbox(self, vo, row):
        """Extract using ground truth bounding boxe
        """
        # Object
        obj_name = self.cfg['object']
        
        sfrm = row['f0_3sec']
        efrm = sfrm + 90
        bbox = [
            row['w0'], row['h0'],
            row['w'], row['h']
        ]

        # Output path of trim
        odir = self.cfg['outdir']
        odir = f"{odir}/gt-bbox/{row['activity']}/"
        if not os.path.isdir(odir):
            os.makedirs(odir)
        oname = f"{os.path.splitext(os.path.basename(row['name']))[0]}_{row['person']}_{sfrm}_{efrm}.mp4"
        opth = f"{odir}/{oname}"

        # trim
        vo.save_spatiotemporal_trim(sfrm, efrm, bbox, opth)
        
        

    def _trim_obj_bbox(self, vo, row):
        """Extract using ground truth bounding boxe
        """
        sfrm = row['f0_3sec']
        efrm = sfrm + 90
        bbox_string = row['obj_bbox']
        obj_name = self.cfg['object']
        
        if bbox_string == "0-0-0-0":
            # if the object is not detected we will use the groundtruth
            # coordinates
            obj_flag = False
            bbox = [
                row['w0'], row['h0'],
                row['w'], row['h']
            ]
        else:
            obj_flag = True
            bbox_array = bbox_string.split('-')
            bbox = [
                bbox_array[0], bbox_array[1],
                bbox_array[2], bbox_array[3]
            ]

        # Output path of trim
        odir = self.cfg['outdir']
        if obj_flag:
            odir = f"{odir}/obj-bbox/{row['activity']}/with-{obj_name}"
        else:
            odir = f"{odir}/obj-bbox/{row['activity']}/without-{obj_name}"
            
        if not os.path.isdir(odir):
            os.makedirs(odir)
        oname = f"{os.path.splitext(os.path.basename(row['name']))[0]}_{row['person']}_{sfrm}_{efrm}.mp4"
        opth = f"{odir}/{oname}"

        # trim
        vo.save_spatiotemporal_trim(sfrm, efrm, bbox, opth)


    def _trim_roi_bbox(self, vo, row):
        """Extract using region of interest bounding boxes
        """

        sfrm = row['f0_3sec']
        efrm = sfrm + 90
        bbox_string = row['roi_bbox']

        bbox_array = bbox_string.split('-')
        bbox = [
            bbox_array[0], bbox_array[1],
            bbox_array[2], bbox_array[3]
        ]
        
        odir = self.cfg['outdir']
        odir = f"{odir}/roi-bbox/{row['activity']}/"
        if not os.path.isdir(odir):
            os.makedirs(odir)
            
        oname = f"{os.path.splitext(os.path.basename(row['name']))[0]}_{row['person']}_{sfrm}_{efrm}.mp4"
        opth = f"{odir}/{oname}"

        # trim
        vo.save_spatiotemporal_trim(sfrm, efrm, bbox, opth)

    def _get_best_roi(self, roi_df, act_inst):
        """Returns the ROI that has the most IoU score w.r.t the activity bounding box.

        Loop through each person (column) save the bounding box that
        gives the highest IoU score. If there is no ROI that scores atleast
        0.25 IoU we note down 0-0-0-0.

        Parameters
        ----------
        roi_df : DataFrame
            A dataframe that has object detections in the 3 second interval.
        row : Series
        """
        
        # If empty return the dataframe without processing
        if roi_df.empty:
            bbox = [0, 0, 0, 0]
            bbox_string = "0-0-0-0"
            iou_max = 0.25
            return bbox, iou_max

        # Get the bounding box from the activity instance
        act_bbox = [
            int(act_inst['w0']),
            int(act_inst['h0']),
            int(act_inst['w0'] + act_inst['w']),
            int(act_inst['h0'] + act_inst['h'])
        ]
        # Get Column name list without 'Time' 'f0', 'video_names' and 'f'
        person_names = roi_df.columns.tolist()
        for x in ['Time', 'f0', 'video_names', 'f']:
            person_names.remove(x)

        # Loop through each person
        iou_max = 0
        for person_name in person_names:
            
            # Combining all the roi bounding boxes
            bboxes = roi_df[person_name].tolist()
            w0 = [int(x.split('-')[0]) for x in bboxes]
            h0 = [int(x.split('-')[1]) for x in bboxes]
            w = [int(x.split('-')[2]) for x in bboxes]
            h = [int(x.split('-')[3]) for x in bboxes]
            w1 = list(map(add, w0, w))
            h1 = list(map(add, h0, h))
            w0_ = min(w0)
            h0_ = min(h0)
            w1_ = max(w1)
            h1_ = max(h1)
            w_ = w1_ - w0_
            h_ = h1_ - h0_
            roi_bbox = [w0_, h0_, w1_, h1_]

            # Calculating IoU for the bounding boxes
            iou = self._get_iou_using_image(act_bbox, roi_bbox, act_inst['W'], act_inst['H'])

            # Storing the maximum iou
            if iou > iou_max:
                iou_max = iou
                bbox_string = f"{w0_}-{h0_}-{w_}-{h_}"
                bbox = [w0_, h0_, w_, h_]

            # If the IoU is <= 0.25
            if iou_max <= 0.25:
                bbox_string = f"0-0-0-0"
                bbox = [0, 0, 0, 0]
                iou_max = 0.25

        return bbox, iou_max

    
    def _get_iou_using_image(self, bbox1, bbox2, W, H):
        """ Returns IoU score for bounding boxes.

        Parameters
        ----------
        bbox1 : List[int]
            [x_tl, y_tl, x_br, y_br]
        bbox2 : List[int]
            [x_tl, y_tl, x_br, y_br]
        """
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
        
        
    def _remove_distant_objects(self, odf, act_inst):
        """Removes object detection instances that are far from activity instance.

        Parameters
        ----------
        odf : DataFrame
            A dataframe that has object detections in the 3 second interval.
        row : Series
        """
        # If empty return the dataframe without processing
        if odf.empty:
            return odf

        # Loop over each detection instance
        for i, row in odf.iterrows():

            # Activity bounding box coordinates
            act_bbox = [
                act_inst['w0'], act_inst['h0'], act_inst['w'], act_inst['h']
                ]

            # Hand detection bounding box
            obj_bbox = [
                row['w0'], row['h0'], row['w'], row['h']
            ]

            # Object detection area % inside activity bounding box
            oarea = self._overlapping_area(act_bbox, obj_bbox)
            oarea_percent = (oarea*100)/(row['w']*row['h'])

            # If the overlap area is < 50% remove the hand instance from
            # odf
            if oarea_percent < 50:
                odf = odf.drop(i)

        return odf


    def _overlapping_area(self, bbox1, bbox2):
        """Returns overlapping area of two rectangles

        Parameters
        ----------
        bbox1 : List[int]
            Bounding box 1 coordinates
        bbox2 : List[int]
            Bounding box 1 coordinates

        Note
        ----
        The bounding box coordinates is assumed to be as follows
        [top left width coordinates, top left height coordinate,
        width of bounding box, height of bounding box]

        Returns
        -------
        overlapping area
        """

        # Casting bounding boxes to integers
        bbox1 = [int(x) for x in bbox1]
        bbox2 = [int(x) for x in bbox2]

        # Copying the coordinates to descriptive variables
        [xtl1, ytl1, xbr1, ybr1] = [
            bbox1[0],
            bbox1[1],
            bbox1[0] + bbox1[2],
            bbox1[1] + bbox1[3]
        ]

        [xtl2, ytl2, xbr2, ybr2] = [
            bbox2[0],
            bbox2[1],
            bbox2[0] + bbox2[2],
            bbox2[1] + bbox2[3]
        ]

        # Top left coordinate of overlapping bounding box
        xtl3 = max(xtl1, xtl2)
        ytl3 = max(ytl1, ytl2)

        # Bottom right coordinates of overlapping bounding box
        xbr3 = min(xbr1, xbr2)
        ybr3 = min(ybr1, ybr2)

        # Width and height
        w3 = xbr3 - xtl3
        h3 = ybr3 - ytl3

        if w3 > 0 and h3 > 0:
            oarea = h3*w3
        else:
            oarea = 0

        # Checking with another method if my calculated oarea
        # is correct
        img1 = np.zeros((480, 848))
        img2 = np.zeros((480, 848))
        img1[ytl1:ybr1, xtl1:xbr1] = 1
        img2[ytl2:ybr2, xtl2:xbr2] = 1
        img3 = img1*img2
        img_oarea = img3.sum()

        # Have a test to verify we are getting the correct area
        if not img_oarea == oarea:
            import pdb; pdb.set_trace()

        # return overlapping area
        return oarea
