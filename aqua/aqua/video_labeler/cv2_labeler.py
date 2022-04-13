import json
import os
import pandas as pd
from tqdm import tqdm
import pytkit as pk
import cv2

class CV2Labeler:
    """
    Uses OpenCV tools to write activity labels to videos.

    cfg: Dict
        Configuration dictionary read from json file.
    """
    def __init__(self, cfg):
        """
        Parameters
        ----------
        cfg: Str
            Configuration file having the following information in
            `json` format.

            1. typing instances as excel file
            2. writing instances as excel file
            3. session information as csv file
        """
        with open(cfg) as f:
            self.cfg = json.load(f)

        if not self.cfg['ty'] == "":
            tydf = pd.read_excel(self.cfg['ty'], sheet_name="Machine readable")
            self._actdf = tydf

        if not self.cfg['wr'] == "":
            wrdf = pd.read_excel(self.cfg['wr'], sheet_name="Machine readable")
            self._actdf = wrdf
            
        if not self.cfg['wr'] == "" and not self.cfg['ty'] == "":
            self._actdf = pd.concat([tydf, wrdf])        
        
        self._spdf = pd.read_csv(self.cfg['sprops'])
        self._colors = self._create_color_dict()


    def _create_color_dict(self):
        """ Creates a dictionary with colors corresponding to each student
        """
        colors = [
            (000, 000, 255),  # red
            (255, 000, 255),  # magenta
            (000, 255, 255),  # yellow
            (255, 255, 000),  # aqua
            (128, 128, 000),  # Teal
            (128, 000, 128),  # purple
            (255, 255, 255),  # white
            (128, 128, 128)   # gray
        ]
        student_codes = self._actdf['student_code'].unique().tolist()
        
        color_dict = {}
        for i, student_code in enumerate(student_codes):
            color_dict[student_code] = colors[i]
            
        return color_dict
        

    def write_labels_to_video(self):
        """ Writes typing and writing instances to video.
        """
        
        # Copy videos from input directory to output
        ipaths, opaths = self._copy_videos()

        # Loop over each video
        for vidx in range(0, len(ipaths)):
            
            # Input and output video paths
            opath = opaths[vidx]
            ipath = ipaths[vidx]

            # skip videos with no activity instances
            if self._has_no_act_instances(ipath):
                continue

            # Label activities in the output video,
            print(f"Labeling: {ipath}")
            self._label_video(ipath, opath)

    def _label_video(self, ipath, opath):
        """
        Parameters
        ----------
        ipath: Str
            Input video path
        opath: Str
            output video path
        """
        # Read video using pk library
        iv = pk.Vid(ipath, 'read')
        ov = pk.Vid(opath, 'write', fps=30, shape=(iv.props['width'], iv.props['height']))

        # Dataframe having instances that belong to input video
        iname = os.path.basename(ipath)
        df = self._actdf[self._actdf['name'] == iname].copy()

        # Read video one frame at a time
        for f0 in tqdm(range(0, iv.props['num_frames'])):

            # Read video frame
            frame = iv.get_next_frame()

            # Activity instances at f0
            dff0 = df[f0 >= df['f0']].copy()
            dff0 = dff0[f0 <= dff0['f0'] + dff0['f']].copy()

            # Label activities in the frame
            if not dff0.empty:
                frame = self._label_activities(dff0, frame)

            # Write frame to output video
            ov.write_frame(frame)

        # Close the video
        iv.close()
        ov.close(video_with_audio=ipath)

        # Add audio channel the output video
        
            

    def _label_activities(self, dff0, frame):
        """ Labels activities in `dff0` into frame. We use solid
        boxes to label typing and dashed boxes to label writing.
        Activities done by each person are color coded.

        Parameters
        ----------
        dff0: DataFrame
            DataFrame with activities to be labeled in current frame
        frame: OpenCV mat object
            Frame
        """
        colors = self._colors.copy()
        
        for ridx, row in dff0.iterrows():
            frame = self._label_activity(frame, row, colors)

        return frame
            


    def _label_activity(self, frame, row, colors):
        """
        Labels an activity instance on video. We use solid
        boxes to label typing and dashed boxes to label writing.
        Activities done by each person are color coded.
        """
        x0 = int(row['w0'])
        y0 = int (row['h0'])
        x1 = int(row['w0'] + row['w'])
        y1 = int(row['h0'] + row['h'])
        
        # Bounding box color
        if row['activity'] == "typing":
            bcolor = (255, 0 , 0)
        elif row['activity'] == "writing":
            bcolor = (0, 255, 0)
        else:
            raise Exception(f"Activity, {row['activity']}, is not supported")

        # Text color
        tcolor = colors[row['student_code']]

        # Draw bounding box
        frame = cv2.rectangle(frame, (x0, y0), (x1, y1), bcolor, 2)

        # Creating a black box on which we will write the label
        text_scaling = 0.6
        text_thickness = 1
        text_size, _ = cv2.getTextSize(
            f"{row['student_code']}-{row['activity']}",
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scaling,
            text_thickness
        )
        text_w, text_h = text_size
        frame = cv2.rectangle(
            frame,
            (x0, y0 + 3),
            (x0 + text_w, y0 + text_h + 5),
            (0, 0, 0),
            -1)

        # Label bounding box
        frame = cv2.putText(
            frame,
            f"{row['student_code']}-{row['activity']}",
            (x0, y0 + text_h + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scaling,
            tcolor,
            text_thickness,
            cv2.LINE_AA
        )

        return frame
        

    def _has_no_act_instances(self, ipath):
        """ Returns true if the current video has no activity instances

        Parameters
        ----------
        ipath: Str
            Input video file path
        """
        vname = os.path.basename(ipath)
        if vname in self._actdf['name'].unique().tolist():
            return False
        return True

    def _copy_videos(self):
        """ Copies videos from session propoerties dataframe
        `self._spdf` from input to output directory. It also adds
        a postfix to the name. For example, a.mp4 becomes a_gt.mp4
        """
        idir = self.cfg['idir']
        odir = self.cfg['odir']
        postfix = self.cfg['name_postfix']
        
        inames = self._spdf.iloc[0:-1]['name'].tolist()
        ipaths = [f"{idir}/{x}" for x in inames]
        
        inames_noext = [os.path.splitext(x)[0] for x in inames]
        onames = [f"{x}_{postfix}.mp4" for x in inames_noext]
        opaths = [f"{odir}/{x}" for x in onames]

        for vidx in range(0, len(ipaths)):
            
            ipath = ipaths[vidx]
            opath = opaths[vidx]

            cp_cmd = f"cp {ipath} {opath}"
            os.system(cp_cmd)

        return ipaths, opaths
