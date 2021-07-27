import os
import sys
import cv2
import pdb
from pathlib import Path
import skvideo.io as skvio
import torch


class Vid:

    # public members
    props = dict()
    """
    Properties of video as dictionary with following keys,
    0. islocal
    1. name
    2. extension (mp4, avi, mkv, ...)
    3. dir_loc
    4. frame_rate
    5. duration in seconds
    6. num_frames
    """
    def __init__(self, pth):
        """ Initializes video instance

        Parameters
        ----------
        pth: str
            Full path or url to video
        """

        # Check if Video exists locally
        if not os.path.isfile(pth):
            self.props['islocal'] = False

        # Initialize props dictionary with video properties
        self.props = self._get_video_properties(pth)

    def _get_video_properties(self, vpath):
        """ Returns a dictionary with following video properties,
        1. video_name
        2. video_ext
        3. video_path
        4. frame_rate

        Parameters
        ----------
        vpath: str
            Video file path
        """
        # Get video file name and directory location
        vdir_loc = os.path.dirname(vpath)
        vname, vext = os.path.splitext(os.path.basename(vpath))

        # Read video meta information
        vmeta = skvio.ffprobe(vpath)

        # If it is empty i.e. scikit video cannot read metadata
        # return empty stings and zeros
        if vmeta == {}:
            vprops = {
                'islocal': False,
                'full_path': vpath,
                'name': vname,
                'extension': vext,
                'dir_loc': vdir_loc,
                'frame_rate': 0,
                'duration': 0,
                'num_frames': 0,
                'width': 0,
                'height': 0
            }

            return vprops

        # Calculate average frame rate
        fr_str = vmeta['video']['@avg_frame_rate']
        fr = round(int(fr_str.split("/")[0]) / int(fr_str.split("/")[1]))

        # get duration
        vdur = round(float(vmeta['video']['@duration']))

        # get number of frames
        vnbfrms = int(vmeta['video']['@nb_frames'])

        # video width
        width = int(vmeta['video']['@width'])

        # video height
        height = int(vmeta['video']['@height'])

        # Creating properties dictionary
        vprops = {
            'islocal': True,
            'full_path': vpath,
            'name': vname,
            'extension': vext,
            'dir_loc': vdir_loc,
            'frame_rate': fr,
            'duration': vdur,
            'num_frames': vnbfrms,
            'width': width,
            'height': height
        }

        return vprops

    def extract_all_frames_ffmpeg(self, odir):
        """ Extacts all frames to odir using ffmpeg

        Parameters
        ----------
        odir: output direcotry to which frames are extracted
        """
        cmd = f"ffmpeg -hide_banner -loglevel panic -i {self.props['full_path']} -start_number 0 -q:v 1 {odir}/%d.jpg"
        os.system(cmd)


    def extract_frame(self, frm_num, odir):
        """
        Extracts a frame from video using its frame number

        Parameters
        ----------
        frm_num: int
            Frame number
        odir: str
            Output directory
        """

        # Read video and seek to frame
        vo = cv2.VideoCapture(self.props['full_path'])
        vo.set(cv2.CAP_PROP_POS_FRAMES, frm_num)
        _, frame = vo.read()

        # Saving
        save_loc = f"{odir}/{self.props['name']}_{frm_num}.png"
        print(f"INFO: saving {save_loc}")
        cv2.imwrite(save_loc, frame)

        # Closing video
        vo.release()

    def extract_frame(self, frm_num, odir):
        """
        Returns a frame from video using its frame number

        Parameters
        ----------
        frm_num: int
            Frame number
        """

        # Read video and seek to frame
        vo = cv2.VideoCapture(self.props['full_path'])
        vo.set(cv2.CAP_PROP_POS_FRAMES, frm_num)
        _, frame = vo.read()

        return frame

    def get_frame(self, frm_num):
        """
        Returns a frame from video using its frame number

        Parameters
        ----------
        frm_num: int
            Frame number
        """

        # Read video and seek to frame
        vo = cv2.VideoCapture(self.props['full_path'])
        vo.set(cv2.CAP_PROP_POS_FRAMES, frm_num)
        _, frame = vo.read()

        return frame

    def get_grayscale_frame(self, frm_num):
        """
        Returns a frame from video using its frame number

        Parameters
        ----------
        frm_num: int
            Frame number
        """

        # Read video and seek to frame
        vo = cv2.VideoCapture(self.props['full_path'])
        vo.set(cv2.CAP_PROP_POS_FRAMES, frm_num)
        _, frame = vo.read()

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    def spatiotemporal_trim(self, sfrm, efrm, bbox, opth):
        """
        Create a spatiotemporl trim. The output video name is
        <in_vid>_sfrm_efrm.mp4

        Parameters
        ----------
        sfrm: int
            Frame number of starting frame.
        efrm: int
            Frame number of ending frame.
        bbox: int[arr]
            Bounding box,
            [<width_location>, <height_location>, <width>, <height>]
        opth: str
            Output video path
        """
        # Time stamps from frame numbers
        sts = sfrm / self.props['frame_rate']
        nframes = efrm - sfrm

        # Creating ffmpeg command string
        crop_str = f"{bbox[2]}:{bbox[3]}:{bbox[0]}:{bbox[1]}"
        ffmpeg_cmd = (
            f'ffmpeg -hide_banner -loglevel warning '
            f'-y -ss {sts} -i {self.props["full_path"]} -vf "crop={crop_str}" '
            f'-c:v libx264 -crf 0 -frames:v {nframes} {opth}')
        print(f"INFO: Trimming {opth}")
        os.system(ffmpeg_cmd)

        return opth

    def resize_on_longer_edge(self, vsize, opth):
        """ Resizes videos on longer  edges

        Parameters
        ----------
        vsize: int
            Desired longer edge size
        opth: str
            Output video path
        """
        ow = self.props['width']
        oh = self.props['height']
        if ow >= oh:
            nw = vsize
            nh = round((nw) * (oh / ow))
            nh = nh - (nh % 2)  # nearest multiple of 2
        else:
            nh = vsize
            nw = round((nh) * (ow / oh))
            nw = nw - (nw % 2)  # nearest multiple of 2

        scale_str = f"{int(nw)}:{int(nh)}"

        print(f"INFO: Reshaped to {scale_str}")
        ffmpeg_cmd = (f'ffmpeg -hide_banner -loglevel warning '
                      f'-y -i {self.props["full_path"]} '
                      f' -c:v libx264 -crf 0 -vf scale={scale_str} '
                      f' {opth}')
        os.system(ffmpeg_cmd)

    def load_to_tensor_using_cv2(self, oshape):
        """ Loads a video as tensor using OpenCV

        Parameters
        ----------
        oshape: tuple of ints
            (output width, output height)
        """
        # Initialize torch tensor that can contain video
        frames_torch = torch.FloatTensor(3, self.props['num_frames'],
                                         oshape[1],
                                         oshape[0])

        # Initialize OpenCV video object
        vo = cv2.VideoCapture(self.props['full_path'])

        poc = 0  # picture order count
        while vo.isOpened():
            ret, frame = vo.read()
            if ret:
                frame = cv2.resize(frame, oshape)
                frame_torch = torch.from_numpy(frame)
                frame_torch = frame_torch.permute(
                    2, 0, 1)  # (ht, wd, ch) to (ch, ht, wd)
                frames_torch[:, poc, :, :] = frame_torch
                poc += 1
            else:
                break

        vo.release()
        return frames_torch
