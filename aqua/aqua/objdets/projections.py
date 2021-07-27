import os
import pdb
import sys
import cv2
import skimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aqua.video_tools import Vid
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


class ObjDetProjs:
    def __init__(self, bboxes_csv, proj_interval):
        """ Creates heat maps using bounding boxes.
        The csv file having bounding boxes is expedted to have following
        columns,
        `{f0, class, W, H, w0, h0, w, h}`.

        It removes bounding boxes which are very small. The threshold
        is dynamically determined per video. Bounding boxes less than
        Q1(25 percentile) size `(w * h)` are removed.

        Parameters
        ----------
        bboxes_csv: str
            Path to csv file having bounding boxes with required columns.
        proj_interval: int
            Projections are generated ever 'proj_interval' seconds. A value
            of -1 assumes that we are projecting for entire video
        """
        # Load file
        if not os.path.isfile(bboxes_csv):
            raise Exception(f"{bboxes_csv} does not exist")
        self._bboxes_df = pd.read_csv(bboxes_csv)

        self._t = proj_interval

    def display_on_video(self, vipth, ws=False):
        """ Saves projections to video

        Parameters
        ----------
        vipth: str
            Input video path
        ws: bool, optional
            Applies water shed when true. Defaults to `False`.
        """
        vin = Vid(vipth)
        nfrms = vin.props['num_frames']

        # Loop over projection intervals
        if self._t < 0:
            frms_to_skip = nfrms
        else:
            frms_to_skip = self._t * vin.props['frame_rate']
        for spoc in range(0, nfrms, frms_to_skip):
            epoc = spoc + frms_to_skip  # end

            cur_bboxes_df = self._bboxes_df[
                self._bboxes_df['f0'] > spoc].copy()
            cur_bboxes_df = cur_bboxes_df[cur_bboxes_df['f0'] < epoc]

            # Only if the detections are presend do these
            if len(cur_bboxes_df) > 0:
                proj_img = self._create_proj_map(cur_bboxes_df.copy())
                if ws:
                    proj_rgb = 255 * (
                        self._apply_watershed(proj_img > 0))
                else:
                    proj_img_scaled = 255*(proj_img)
                    proj_rgb = np.zeros((vin.props['height'], vin.props['width'], 3))
                    proj_rgb[:,:,0] = proj_img_scaled
                    proj_rgb[:,:,1] = proj_img_scaled
                    proj_rgb[:,:,2] = proj_img_scaled


                # Get the middle frame
                # Video loop
                
                for poc in range(spoc, epoc, vin.props['frame_rate']):
                    frm = vin.get_frame(poc)

                    # Blend images
                    alpha = 0.3
                    beta = 1 - alpha
                    blended_frm = cv2.addWeighted(
                        frm, alpha, proj_rgb.astype('uint8'), beta, 0.0)
                    cv2.imshow("Blended(q to quit, p to pause)", blended_frm)
                    cv2.imshow(
                        "Projection map(q to quit, p to pause)", proj_img)
                    key = cv2.waitKey(33)
                    if key == ord('q'):
                        break
                    if key == ord('p'):
                        cv2.waitKey(-1)  # wait until any key is pressed

    def _apply_watershed(self, image):
        """ Applies watershed to gray scale image.

        image: ndarray
            A gray scale image.
        """
        image = image.astype("uint8").copy()
        distance = ndi.distance_transform_edt(image)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=image)
        labels_rgb = skimage.color.label2rgb(labels, bg_label=0, bg_color=(0,0,0))
        return 255*labels_rgb


    def write_to_video(self, vipth, vopth):
        """ Saves projections to video

         Parameters
         ----------
         vipth: Input video path
         vopth: Output path
         """
        vin = Vid(vipth)
        nfrms = vin.props['num_frames']

        # output writer
        vout = cv2.VideoWriter(
            vopth,
            cv2.VideoWriter_fourcc('M','J','P','G'), 30,
            (vin.props['width'],vin.props['height'])
        )

        # Loop over projection intervals
        if self._t < 0:
            frms_to_skip = nfrms
        else:
            frms_to_skip = self._t*vin.props['frame_rate']
        for spoc in range(0, nfrms, frms_to_skip):
            mpoc = spoc + (frms_to_skip/2) # middle
            epoc = min(spoc + frms_to_skip, nfrms-1) # end

            cur_bboxes_df = self._bboxes_df[self._bboxes_df['f0'] > spoc].copy()
            cur_bboxes_df = cur_bboxes_df[cur_bboxes_df['f0'] < epoc]

            # Only if the detections are presend do these
            if len(cur_bboxes_df) > 0:
                proj_img = self._create_proj_map(cur_bboxes_df.copy())
                proj_img_scaled = 255*(proj_img)
                proj_rgb = np.zeros((vin.props['height'], vin.props['width'], 3))
                proj_rgb[:,:,0] = proj_img_scaled
                proj_rgb[:,:,1] = proj_img_scaled
                proj_rgb[:,:,2] = proj_img_scaled


                # Get the middle frame
                # Video loop
                for poc in range(spoc, epoc, vin.props['frame_rate']):
                    print(poc)
                    frm = vin.get_frame(poc)

                    # Blend images
                    alpha = 0.3
                    beta = 1 - alpha
                    blended_frm = cv2.addWeighted(frm, alpha, proj_rgb.astype('uint8'), beta, 0.0)
                    vout.write(blended_frm)
        vout.release()






    def get_proj_map_for_full_video(self):
        """
        Returns projection map as numpy array for complete video
        """
        full_video_proj = self._create_proj_map(self._bboxes_df.copy())
        return full_video_proj

    def _create_proj_map(self, df_bboxes):
        """ Creates a projection for full video as numpy
        array

        Parameters
        ----------
        df_bboxes: DataFrame
            DataFrame having bounding box values
        """
        try:
            W = df_bboxes['W'].unique().item()
        except:
            pdb.set_trace()
        H = df_bboxes['H'].unique().item()

        # Initializing numpy array to zeros
        hm_np = np.zeros((H,W))

        # Bboxes loop
        for idx, row in df_bboxes.iterrows():

            # Current bounding box information
            w = row['w']
            h = row['h']
            w0 = row['w0']
            h0 = row['h0']

            # Creating binary image for current bounding box
            tmp_bimg = np.zeros((H,W))
            tmp_bimg[h0:h0+h, w0:w0+h] = 1

            # Keep adding these images
            hm_np = hm_np + tmp_bimg

        # Normalize heat map to 0 and 1
        hm_np = hm_np/hm_np.max()
        return hm_np
