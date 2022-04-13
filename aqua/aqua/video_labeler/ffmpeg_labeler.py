import os
import pdb
import pandas as pd
import tempfile
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm


class FFMPEGLabeler:

    act = ""
    """ Activity, {typing, writing} """

    sess_path = ""
    """ Session path """

    act_df = pd.DataFrame()
    """ Data frame having activities  """

    sess_df = pd.DataFrame()
    """ Data frame having session properties """

    def __init__(self, activity_instances, session_properties, act, name_post_fix):
        """
        Creates 
        
        Parameters
        ----------
        activity_instances: Str
            Excel file having activity instances
        session_properties: Str
            CSV having session properties
        act: Str
            Activity we are currently plotting
        name_post_fix: Str
            String added to the end of input video names.
        """
        
        # Properties of current activity instances
        self.act = act
        self.name_post_fix = name_post_fix
        self.sess_path = os.path.dirname(session_properties)
        
        # Creating data frames having session and activity instances information
        self.sess_df = pd.read_csv(session_properties)
        self.act_df = pd.read_excel(activity_instances, sheet_name="Machine readable")
        
        # Create a dictionary having student code as keys and colors as values
        self.colors = self._get_colors_for_students()
        
    def write_labels_to_video(self):
        """ Writes labels to video using xlsx file created from ground truth csv file.
        """
        # Input videos
        in_videos = self.sess_df[0:-1]['name'].tolist()

        # Loop through each video
        for in_video in in_videos:
            
            # Copy input video to output video
            iv_path, ov_path = self._copy_input_video(in_video)
            
            # Get intances for current video from activities dataframe
            vact_df = self.act_df[self.act_df['name'] == in_video].copy()
            
            # if dataframe is empty do not execute the rest of the code
            if vact_df.empty:
                continue

            # Create bounding boxes on the video using ffmpeg
            self._label_video(ov_path, vact_df)

    def _get_colors_for_students(self):
        """ Returns a color dictionary with student codes as keys and colors
        as values. These colors will be used when drawing bounding boxes.
        """
        
        # Color list
        colors = ['red', 'green', 'blue', 'pink', 'magenta', 'orange', 'purple', 'black', 'white', ]
        uniq_students = self.act_df['student_code'].unique().tolist()
        color_dict = {}

        # Loop through each student to create a color dictionary
        for i, sc in enumerate(uniq_students):
            color_dict[sc] = colors[i]
            
        return color_dict

    def _label_video(self, ov_path, df):
        """ Label activity instances on the video.

        ov: Str
            Output video path
        df: DataFrame
            Activity instances in that video
        """

        # Loop through each activity instance
        for i, row in df.iterrows():
            # Draw box
            self._draw_box_fmmpeg(ov_path, row)

        # Label on top left
        self._draw_labels_ffmpeg(ov_path)


    def _draw_box_fmmpeg(self, vpth, row):
        """ Return an ffmpeg command that can draw bounding box

        Parameters
        ----------
        vpth: Str
            Video path
        row: Series
            A row from dataframe having activities
        """
        # Get drawbox string
        drawbox_str = self._get_drawbox_string(row)
        
        tdir = tempfile.gettempdir()

        ffmpeg_cmd = f"ffmpeg -y -i {vpth} -preset veryfast -c:a copy -vf {drawbox_str} {tdir}/temp.mp4"
        os.system(ffmpeg_cmd)

        # Copy temporary file to vpth
        os.system(f"mv {tdir}/temp.mp4 {vpth}")


    def _get_drawbox_string(self, row):
        """
        Parameters
        ----------
        row: Series
            A row from dataframe having activities
        """
        x  = int(row['w0'])
        y  = int(row['h0'])
        w  = int(row['w'])
        h  = int(row['h'])
        f0 = row['f0']
        f1 = row['f0'] + row['f']
        color = self.colors[row['student_code']]

        drawbox_str = (f"\"drawbox=enable='between(n, {f0}, {f1})':x={x}:y={y}:w={w}:h={h}:color={color}@0.8:thickness=3\"")

        return drawbox_str
     
    def _draw_labels_ffmpeg(self, vpth):
        """
        Parameters
        ----------
        vpth: Str
            Video path
        """
        tdir = tempfile.gettempdir()

        text = "[in]"
        x = 5
        y = 5
        for i, k in enumerate(self.colors):
            student_code = k
            color = self.colors[k]
            if i == 0:
                text = f"{text}drawtext:text={student_code}:fontcolor={color}:fontsize=20:x={x}:y={y}:line_spacing=3:borderw=2"
            else:
                text = f"{text},drawtext=text={student_code}:fontcolor={color}:fontsize=20:x={x}:y={y+i*20}:line_spacing=3:borderw=2"
                
        text = f"{text}[out]"

        ffmpeg_cmd = f"ffmpeg -y -i {vpth} -preset veryfast -c:a copy -vf {text} {tdir}/temp.mp4"
        os.system(ffmpeg_cmd)
        os.system(f"mv {tdir}/temp.mp4 {vpth}")
        

                
    def _copy_input_video(self, in_video):
        """ Copy input video to output video.

        Parameters
        ----------
        in_video: Str
            Name of input video
        """
        
        # Input and output video paths
        iv_name_no_ext = os.path.splitext(in_video)[0]
        ov_name = f"{iv_name_no_ext}_{self.name_post_fix}.mp4"
        iv_path = f"{self.sess_path}/{in_video}"
        ov_path = f"{self.sess_path}/{ov_name}"
            
        # Copy input video to output video
        os.system(f"cp {iv_path} {ov_path}")

        return iv_path, ov_path
