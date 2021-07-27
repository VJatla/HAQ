import os
import sys
import cv2
import pdb
import wget
import json
import pprint
import tempfile
import numpy as np
import pandas as pd
import aqua
import shutil
import skvideo.io as skvio
from aqua.video_tools import Vid

class Visualize:
    def __init__(self, rdir, act_df):
        """ 
        Methods that help in visualizing activity labels.

        Parameters
        ----------
        rdir: str
            Directory path having activity labels.
        act_df: Pandas DataFrame
            Dataframe containing activity of interest.
        """
        self._rdir = rdir
        self._act_df = act_df

    def to_video_ffmpeg(self, names_df = pd.DataFrame()):
        """ Draws boxes in video using ffmpeg. This is not optimal
        as the video is compressed multiple times. I am writing this
        method due to time crunch.

        Parameters
        ----------
        names_df: Pandas DataFrame, optional
            If provided the video bounding boxes will use pseudonyms instead
            of numerical code.
        """

        # Extract videos to temporary directories
        temp_dir = tempfile.mkdtemp()
        vlist = list(self._act_df['name'].unique())
        for vname in vlist:

            cur_temp_dir = f"{temp_dir}/{os.path.splitext(vname)[0]}"
            os.mkdir(cur_temp_dir)
            
            vpath = f"{self._rdir}/{vname}"
            vid = Vid(vpath)
            print(f"Extracting frames \n\t {vname}")
            vid.extract_all_frames_ffmpeg(cur_temp_dir)

        
        # Loop through each activity instance and put a bounding box
        # around it with pseudonym
        for ridx, row in self._act_df.iterrows():
            pseudonym = names_df[names_df['numeric_code'] == row['person']]['pseudonym'].item()
            cur_temp_dir = f"{temp_dir}/{os.path.splitext(row['name'])[0]}"
            activity = row['activity']
            fs = row['f0']
            fe = row['f0'] + row['f']
            tl = (int(row['w0']), int(row['h0']))
            br = (int(row['w0']) + int(row['w']), int(row['h0']) + int(row['h']))
            for img_idx in range(fs,fe):
                img_path =  f"{cur_temp_dir}/{img_idx}.jpg"
                img = cv2.imread(img_path)
                img = cv2.rectangle(img, tl, br, (0, 255, 0), 2)
                img = cv2.rectangle(img, (int(row['W']/2), 7), (int(row['W']/2)+100, 10+30), (0, 0, 0), -1)
                img = cv2.putText(img, pseudonym, tl, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                img = cv2.putText(img, activity, (int(row['W']/2), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                # cv2.imshow("df", img)
                # cv2.waitKey(0)
                # sys.exit()
                try:
                    cv2.imwrite(img_path, img)
                except:
                    pdb.set_trace()

        # Now creating video from all the images
        vlist = list(self._act_df['name'].unique())
        try:
            for vname in vlist:
                print(f"Writing video\n\t {vname}")
                cur_temp_dir = f"{temp_dir}/{os.path.splitext(vname)[0]}"
                out_path = f"{self._rdir}/algo_labels_{vname}"
                cmd = f"ffmpeg -i '{cur_temp_dir}/%d.jpg' -vf fps=30  {out_path}"
                print(Cmd)
                os.system(cmd)
                    
        except:
            # Cleaning up temporary files if there is exception
            pdb.set_trace()
            shutil.rmtree(temp_dir)
            
        shutil.rmtree(temp_dir)
