"""
IMPORTANT NOTE
--------------
The following class is only used to create session_video.mp4 with a frame
taken every second (30 frames).

The only method used here is, `create_session_video()`.
"""

import pdb
import pandas as pd
import numpy as np
from matplotlib import cm
from tqdm import tqdm

# User defined libraries
import cv2
import pytkit as pk


class ROI:

    rdir = ""
    """ 
    Root directory having csv files with rois extracted from
    matlab video labeler.
    """

    sprops = None
    """
    A dictionary having session properties.
    """

    df_session = None
    """
    A pandas DataFrame having region proposals per second.
    """
    
    def __init__(self, rdir, FPS):
        self.rdir = rdir

        # Session properties
        self.sprops = {"FPS": FPS}

    def create_session_video(self, n=1):
        """ Creating session level video taking `n` frames
        every second.

        Parameters
        ----------
        n : Int, optional
            Number of frames to take from a sec.mp4ond starting from
            frame 0.
        """
        print(f"Saving {self.rdir}/session_video.mp4")
        
        video_files = pk.get_file_paths_with_kws(self.rdir, ['30fps', 'mp4'])
        video_session = pk.Vid(f"{self.rdir}/session_video.mp4", "write")

        for video_file in tqdm(video_files):
            
            video = pk.Vid(video_file, "read")
            
            for f0 in range(0, video.props['num_frames'], video.props['frame_rate']):
                
                frm = video.get_frame(f0)
                frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                video_session.write_frame(frm)
                
            video.close()

        # Closing session video
        video_session.close()


    def create_session_csv(self):
        """ 
        Create a session level csv file that has table rois.

        visualize : bool
            See region proposals on a frame
        """
        # List of csv files that has video level roi exported
        # from matlab
        csv_files = pk.get_file_paths_with_kws(
            self.rdir, ['exported' '.csv'], no_kw_lst=['#']
        )
        
        # Get columns from all the csv files in the directory
        columns = self._get_pseudonyms_for_session(csv_files)
        columns = ['video_name', 'f0'] + columns
        df_session = pd.DataFrame(columns=columns)
        
        # Loop through each video rois and create a lists for session
        # level rois        
        for csv_file in csv_files:

            # Read rois for a video in a session
            df = pd.read_csv(csv_file)

            # Getting rows where there is atleast one entry
            # for roi
            valid_row_idxs = sorted(self._get_valid_row_index(df))
            valid_row_idxs += [len(df)]
            
            # Loop through each valid row_indexes
            sridx = 0
            for ridx in valid_row_idxs:
                
                dft = df[sridx:ridx+1].copy()
                dft = dft.fillna(method='bfill')
                dft = dft.fillna("0-0-0-0")
                
                df_session = pd.concat([df_session, dft])
                df_session = df_session.fillna("0-0-0-0")
                
                sridx = ridx

            # Get every 30th row (1 second rois at 30 FPS)
            df_session = df_session[df_session.index % self.sprops['FPS'] == 0]

        # Save the pandas dataframe
        print(f"Saving {self.rdir}/session_table_rois.csv")
        df_session.to_csv(
            f"{self.rdir}/session_table_rois.csv", index=False
        )
        self.df_session = df_session


    def _get_pseudonyms_for_session(self, csv_files):
        """ Column names, i.e. all the names of persons in the session

        Parameters
        ----------
        csv_files : List of Strings
            Paths to csv files that have rois
        """
        columns = set([])
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            columns_temp = set(df.columns.tolist())
            columns = columns.union(columns_temp)
            
        # Remove `video_name`, `f0`
        columns.remove('f0')
        columns.remove('video_name')
        columns = [x for x in columns]
        return columns
        
        
    def show_rois(self):
        """
        Show ROIs per seocnd.
        """
        df = self.df_session.copy()

        # pseudonyms in the session
        pseudonyms = df.columns.tolist()[2:]
        pseudonyms_tuple = [(i, x) for i, x in enumerate(pseudonyms)]

        # Unique videos
        video_names = df['video_name'].unique().tolist()

        # Video loop
        for video_name in video_names:

            # Data frame having regions for current video
            df_video = df[df['video_name'] == video_name].copy()

            # Load video
            vid = pk.Vid(f"{self.rdir}/{video_name}", "read")

            # Frame level loop
            for idx, row in df_video[df_video.index % 10 == 0].iterrows():

                # Get current frame
                frm = vid.get_frame(row['f0'])

                # Loop through pseudonyms and plot bonding box
                # on the frame
                ccanvas = np.zeros(frm.shape).astype('uint8')
                for i, name in pseudonyms_tuple:
                    color = tuple(
                        [255 * x for x in list(cm.Set3(i))[0:3]])
                    bbox = row[name]
                    ccanvas = self._draw_roi(ccanvas, bbox, name, color)

                # Blend canvas and video frame
                alpha = 0.5
                beta = (1.0 - alpha)
                frm = cv2.addWeighted(frm, alpha, ccanvas, beta, 0.0)

                # Show frame using opencv
                cv2.imshow(f"ROI in {video_name}", frm)
                cv2.waitKey(1)

    def write_rois(self):
        """
        Write ROI to video
        """
        # Session video to write
        vid_out_pth = f"{self.rdir}/session_vid_rois.mp4"
        vid_out = pk.Vid(vid_out_pth, "write")
        
        df = self.df_session.copy()

        # pseudonyms in the session
        pseudonyms = df.columns.tolist()[2:]
        pseudonyms_tuple = [(i, x) for i, x in enumerate(pseudonyms)]

        # Unique videos
        video_names = df['video_name'].unique().tolist()

        # Video loop
        for video_name in video_names:

            # Data frame having regions for current video
            df_video = df[df['video_name'] == video_name].copy()

            # Load video
            vid = pk.Vid(f"{self.rdir}/{video_name}", "read")

            # Frame level loop
            for idx, row in df_video[df_video.index % 10 == 0].iterrows():
                
                # Get current frame
                frm = vid.get_frame(row['f0'])

                # Loop through pseudonyms and plot bonding box
                # on the frame
                ccanvas = np.zeros(frm.shape).astype('uint8')
                for i, name in pseudonyms_tuple:
                    color = tuple(
                        [255 * x for x in list(cm.Set3(i))[0:3]])
                    bbox = row[name]
                    ccanvas = self._draw_roi(ccanvas, bbox, name, color)

                # Blend canvas and video frame
                alpha = 0.5
                beta = (1.0 - alpha)
                frm = cv2.addWeighted(frm, alpha, ccanvas, beta, 0.0)
                frm_rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

                # Write frame to video
                vid_out.write_frame(frm_rgb)

        vid_out.close()


    def _get_valid_row_index(self, df):
        """ Get indexes of rows which have atleaset one entry for ROI

        df : Pandas DataFrame
            A dataframe containing ROIs, that are extracted from MATLAB
        """
        dfp    = self._get_only_pseudonym_cols(df)
        dfp_na = dfp.notna()

        # Or w.r.t columns        
        validity_list = dfp_na.any(axis='columns').tolist()
        valid_row_indexes = [i for i, x in enumerate(validity_list) if x]
        return valid_row_indexes


    def _get_only_pseudonym_cols(self, df):
        """ Remove all columns except for pseudonyms. The dataframe
        is assumed to have following columns,
        (video_name, f0, <Pseudonym 1>, <Pseudonym 2>, ...)

        df : Pandas DataFrame
            A dataframe containing ROIs, that are extracted from MATLAB
        """
        df_temp = df.copy()
        df_temp = df_temp.drop(columns=['video_name','f0'])
        return df_temp

    def _draw_roi(self, ccanvas, bb, name, color):
        """
        Draws a bonding box on frame
        """
        try:
            w0, h0, w, h = [int(x) for x in bb.split("-")]
        except:
            import pdb; pdb.set_trace()
        if w != 0 or h != 0:
            ccanvas = cv2.rectangle(ccanvas, (w0, h0), (w0+w, h0+h), color, 2)

        ccanvas = cv2.putText(
            ccanvas,
            f"{name}",
            (w0+30, h0+30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA
        )

        return ccanvas
        
    def _get_roi_per_person(self, df, p, f0):
        """
        Parameters
        ----------
        df : DataFrame
            DataFrame having video level rois
        p : String
            String having pseudonym
        f0 : List of Int
            List having frame numbers of interest
        """
        return 0
            
        

    def _get_kids_pseudonyms(self, df):
        """ Returns pseudonyms of kids involved in a video.

        Parameters
        ----------
        df : Pandas DataFrame
            DF containing ROI for a video
        """
        pseudonyms = df.columns.tolist()
        pseudonyms = [
            e for e in pseudonyms if e not in ('f0', 'video_name')
        ]
        return pseudonyms
