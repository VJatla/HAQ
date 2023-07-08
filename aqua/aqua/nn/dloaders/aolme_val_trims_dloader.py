import os
import cv2
import sys
import pdb
import numpy as np
from aqua.video_tools import Vid
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class AOLMEValTrmsDLoader(Dataset):
    def __init__(self, vdir, trms_lst_file, oshape=(224, 224)):
        """
        DataSet class for loading trimmed aolme videos from activity classification. 
        This is written to work with `mmaction2` way of organizing.

        Parameters
        ----------
        vdir: str
            Directory having videos.
        trms_lst_file: str
            Text file having video relative path (relative to `vdir` argument)
            and activity label.
        oshape: tuple of integers, optional
            (output width, output height) tuple. Defaults to (224, 224)

        Todo
        ----
        - I am using OpenCV to load video files. This is not optimal.
          this should move to ffmpeg/hardware accelerated decoding
          (scikit-video, decord, nvidia-decoder) for faster training.
        """
        self._vdir = vdir
        self._oshape = oshape
        self._trms = self._load_trms_lst_file(trms_lst_file)

    def __len__(self):
        """ Returns total number of trims beign considered.
        """
        return len(self._trms)

    def __getitem__(self, idx):
        vpth = self._trms[idx][0]
        vlabel = np.array(int(self._trms[idx][1]))

        # Video tensor
        vtensor = self._load_video_cv2(vpth)

        # Following book I am returning (label, tensor)
        return (vlabel, vtensor)

    def _load_trms_lst_file(self, fpth):
        """ 
        Read trims list file and create tuple of path and label
        [(<trm path>, <trm label>), ...]

        fpth: str
            Text file having video relative path and activity label.
        """
        if os.path.isfile(fpth):
            with open(fpth, 'r') as f:
                clines = f.readlines()
                lst = [(f"{self._vdir}/{x.split(' ')[0]}",
                        x.split(' ')[1].rstrip()) for x in clines]
        return lst

    def _load_video_cv2(self, vpth):
        """ Load video as tensor using OpenCV video object.
        
        Parameters
        ----------
        vpth: str
            Path to trimmed video
        """
        vid = Vid(vpth)
        frames_rgb_torch = vid.load_to_tensor_using_cv2(oshape=self._oshape, data_aug_flag = False)
        return frames_rgb_torch
