import os
import sys
import pdb
import cv2
import aqua
import math
import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from barbar import Bar
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from aqua.nn.dloaders import AOLMETrmsDLoader
from aqua.nn.dloaders import AOLMEValTrmsDLoader
import pytkit as pk
from aqua.nn.models import DyadicCNN3D
from aqua.nn.models import DyadicCNN3DV2
from torchsummary import summary
import nvidia_smi
nvidia_smi.nvmlInit()
import time

class Typing3:
    """
    Note
    ----
    This is hard coded to 10 frame samples per second or a total of
    30 frames in 3 second video.
    """
    
    tydf = pd.DataFrame()

    tyrp = pd.DataFrame()

    tyrp_roi_only = pd.DataFrame()

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
        """ Spatio temporal typing detection using,

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



    def extract_typing_proposals_using_roi(self, overwrite = True, model_fps = None):
        """Extracts typing region proposals using ROI to a directory (`odir`).

        It write the output to `tyrp_only_roi.csv` adding an extra
        column for names of extracted videos.

        It also creates a text file at the output location of proposals
        in the format our validation dataloader and mmaction2 validation
        dataloader expects called, `proposals_list.txt`

        Parameters
        ----------
        overwrite : Bool, optional
            Overwrite existing typing proposals. Defaults to True.
        model_fps : int, optional
            The input FPS expected by the testing model. The proposals extracted from then
            video have to be sampled at this frame rate. Defaults to the FPS of the session
            video.
        """
        
        # Processing arguments
        odir = pk.check_dir_existance(self.cfg['prop_vid_oloc'])
        if model_fps == None:
            model_fps = self.fps

        # Loop through each video
        prop_name_lst = []
        prop_rel_paths = []
        video_names = self.tyrp_roi_only['name'].unique().tolist()
        for i, video_name in enumerate(video_names):

            # Typing proposals for current dataframe
            print(f"Extracting typing region proposals from: {video_name}")

            # Region proposal per video
            tyrp_video = self.tyrp_roi_only[self.tyrp_roi_only['name'] == video_name].copy()

            # Reading input video
            ivid = pk.Vid(f"{self.cfg['vdir']}/{video_name}", "read")

            # Loop through each instance in the video
            for ii, row in tqdm(tyrp_video.iterrows(), total=tyrp_video.shape[0], desc="Extracting: "):

                # Spatio temporal trim coordinates
                bbox = [row['w0'],row['h0'], row['w'], row['h']]
                sfrm = row['f0']
                efrm = row['f1']

                # Trimming video
                prop_name = f"{ivid.props['name']}_{row['pseudonym']}_{sfrm}_to_{efrm}.mp4"
                prop_name_lst += [prop_name]
                opth_rel = f"proposals/{prop_name}"
                prop_rel_paths += [opth_rel]
                opth = f"{self.cfg['prop_vid_oloc']}/{opth_rel}"

                # Check if the file already exists
                if overwrite:
                    ivid.save_spatiotemporal_trim(sfrm, efrm, bbox, opth, fps=self.cfg['model_fps'])
                else:
                    if not os.path.isfile(opth):
                        ivid.save_spatiotemporal_trim(sfrm, efrm, bbox, opth, fps=self.cfg['model_fps'])

        # Saving the proposal dataframe with new column
        if "proposal_name" in self.tyrp_roi_only.columns:
            self.tyrp_roi_only.drop("proposal_name", inplace=True, axis=1)
            self.tyrp_roi_only['proposal_name'] = prop_name_lst
        else:
            self.tyrp_roi_only['proposal_name'] = prop_name_lst
        tyrp_roi_only_loc = f"{self.cfg['oloc']}/tyrp_only_roi.csv"
        print(f"Rewriting {tyrp_roi_only_loc}")
        self.tyrp_roi_only.to_csv(tyrp_roi_only_loc, index=False)

        # Saving the proposals list text files
        text_file_path = f"{self.cfg['prop_vid_oloc']}/proposals_list.txt"
        print(f"Writing {text_file_path}")
        f = open(text_file_path, "w")
        for i in range(0, len(prop_rel_paths)):
            f.write(f"{prop_rel_paths[i]} 100\n")
        f.close()



    def extract_typing_proposals_using_roi_kbdet(self, overwrite = True, model_fps = None):
        """Extracts typing region proposals using ROI to a directory (`odir`).

        It write the output to `tyrp_only_roi.csv` adding an extra
        column for names of extracted videos.

        It also creates a text file at the output location of proposals
        in the format our validation dataloader and mmaction2 validation
        dataloader expects called, `proposals_list_kbdet.txt`

        Parameters
        ----------
        overwrite : Bool, optional
            Overwrite existing typing proposals. Defaults to True.
        model_fps : int, optional
            The input FPS expected by the testing model. The proposals extracted from then
            video have to be sampled at this frame rate. Defaults to the FPS of the session
            video.
        """
        
        # Processing arguments
        odir = pk.check_dir_existance(self.cfg['prop_vid_oloc'])
        if model_fps == None:
            model_fps = self.fps

        # Loop through each video
        prop_name_lst = []
        prop_rel_paths = []
        video_names = self.tyrp_roi_only['name'].unique().tolist()
        for i, video_name in enumerate(video_names):

            # Loading keyboard detection dataframe
            video_name_no_ext = os.path.splitext(video_name)[0]
            kb_det = pd.read_csv(f"{self.cfg['kb_detdir']}/{video_name_no_ext}_60_det_per_min.csv")

            # Typing proposals for current dataframe
            print(f"Extracting typing region proposals from: {video_name}")

            # Region proposal per video
            tyrp_video = self.tyrp_roi_only[self.tyrp_roi_only['name'] == video_name].copy()

            # Reading input video
            ivid = pk.Vid(f"{self.cfg['vdir']}/{video_name}", "read")

            # Loop through each instance in the video
            for ii, row in tqdm(tyrp_video.iterrows(), total=tyrp_video.shape[0], desc="Extracting: "):

                # Spatio temporal trim coordinates
                bbox = [row['w0'],row['h0'], row['w'], row['h']]
                sfrm = row['f0']
                efrm = row['f1']

                # Get keyboard detection intersection bounding box
                kb_bbox = self._get_kbdet_intersection(kb_det, sfrm, efrm)

                # Did they overlap?
                iflag, icoords = self._get_intersection(bbox, kb_bbox)

                # Trimming video
                if iflag:
                    prop_name = f"{ivid.props['name']}_{row['pseudonym']}_{sfrm}_to_{efrm}.mp4"
                    prop_name_lst += [prop_name]
                    opth_rel = f"proposals_kbdet/{prop_name}"
                    prop_rel_paths += [opth_rel]
                    opth = f"{self.cfg['prop_vid_oloc']}/{opth_rel}"

                    # Check if the file already exists
                    if overwrite:
                        ivid.save_spatiotemporal_trim(sfrm, efrm, bbox, opth, fps=self.cfg['model_fps'])
                    else:
                        if not os.path.isfile(opth):
                            ivid.save_spatiotemporal_trim(sfrm, efrm, bbox, opth, fps=self.cfg['model_fps'])
                else:
                    prop_name_lst += ["dummy_name.mp4"]
                    opth_rel = f"proposals_kbdet/dummy_name.mp4"
                    prop_rel_paths += [opth_rel]
                            

        # Saving the proposal dataframe with new column
        if "proposal_name" in self.tyrp_roi_only.columns:
            self.tyrp_roi_only.drop("proposal_name", inplace=True, axis=1)
            self.tyrp_roi_only['proposal_name'] = prop_name_lst
        else:
            self.tyrp_roi_only['proposal_name'] = prop_name_lst
        tyrp_roi_only_loc = f"{self.cfg['oloc']}/tyrp_only_roi.csv"
        print(f"Rewriting {tyrp_roi_only_loc}")
        self.tyrp_roi_only.to_csv(tyrp_roi_only_loc, index=False)

        # Saving the proposals list text files
        text_file_path = f"{self.cfg['prop_vid_oloc']}/proposals_list_kbdet.txt"
        print(f"Writing {text_file_path}")
        f = open(text_file_path, "w")
        for prop_rel_path in prop_rel_paths:
            if prop_rel_path != "proposals_kbdet/dummy_name.mp4":
                f.write(f"{prop_rel_path} 100\n")
        f.close()
        
    def generate_typing_proposals_using_roi(self, dur=3, fps=30, overwrite = True):
        """ Calculates typing region proposals using ROI and keyboard
        detections.

        It write the output to `tyrp_only_roi.csv`

        Parameters
        ----------
        dur : int, optional
            Duraion of each typing proposal
        fps : Frames per second, optional
            Framerate of 
        """
        tyrp_roi_only_loc = f"{self.cfg['oloc']}/tyrp_only_roi.csv"
        
        # if overwrite == False then we check of existing csv with region proposals
        if not overwrite:
            tyrp_roi_only_exists, self.tyrp_roi_only = self._read_from_disk(tyrp_roi_only_loc)
            if tyrp_roi_only_exists:
                print(f"Reading {tyrp_roi_only_loc}")
                return True

        print(f"Creating {tyrp_roi_only_loc}")
        self.dur = dur
        self.fps = fps
        vid_names = self.cfg['vids']
        
        # Loop over each video every 3 secons
        typrop_lst = []
        for vid_name in vid_names:
            
            # Properties of current Video
            T = int(self.sprop_df[self.sprop_df['name'] == vid_name]['dur'].item())
            W = int(self.sprop_df[self.sprop_df['name'] == vid_name]['width'].item())
            H = int(self.sprop_df[self.sprop_df['name'] == vid_name]['height'].item())
            FPS = int(self.sprop_df[self.sprop_df['name'] == vid_name]['FPS'].item())

            # ROI and Keyboard detection for current video
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

                # skip this 3 seconds?
                skip_flag = self._skip_this_3sec_roi_only(roi_df_3sec)

                # If not skipping 
                if not skip_flag:
                    
                    # Get 3 second proposal regions
                    typrop_3sec = self._get_3sec_proposal_df_roi_only(roi_df_3sec.copy())

                    # Adding to prop_lst
                    for typrop_3sec_i in typrop_3sec:
                        typrop_lst_temp = [vid_name, W, H, FPS, T, f0, f, f1] + typrop_3sec_i
                        typrop_lst += [typrop_lst_temp]

        # Creating typing proposal dataframe
        tyrp = pd.DataFrame(
            typrop_lst,
            columns=['name', 'W', 'H', 'FPS', 'T', 'f0', 'f', 'f1', 'pseudonym', 'w0', 'h0', 'w', 'h']
        )
        self.tyrp_roi_only = tyrp
        self.tyrp_roi_only.to_csv(tyrp_roi_only_loc, index=False)
        return True


    def classify_typing_proposals_roi_fast_approach(self, overwrite=False, batch_size = 4, kb_det=False):
        """Classify each proposed region as typing / no-typing. This method needs the
        dataset in the format validation AOLMETrmsDLoader expects.

        Parameters
        ----------
        overwrite : Bool, optional
            Overwrites existing excel file 
        """

        # Output excel file with predictions
        if kb_det:
            out_file = f"{self.cfg['oloc']}/tynty-roi-ours-{3*self.cfg['model_fps']}-kbdet.csv"
        else:
            out_file = f"{self.cfg['oloc']}/tynty-roi-ours-{3*self.cfg['model_fps']}.csv"

        # Creating default columns for activity, class_idx and class_prob
        self.tyrp_roi_only["activity"] = "notyping"
        self.tyrp_roi_only["class_idx"] = 0
        self.tyrp_roi_only["class_prob"] = 0
        
        # If the file already exists and overwrite argument is false
        # we load the file as dataframe
        if not overwrite:
            if os.path.isfile(out_file):
                print(f"Reading {out_file}")
                tydf = pd.read_csv(out_file)
                self.tydf = tydf
                return tydf

        # creating tydf by copying self.tyrp_only_roi
        self.tydf = self.tyrp_roi_only.copy()
        
        # Loading list of videos from the proposals_list.txt file
        proposal_names = []
        if kb_det:
            proposal_list_file =f"{self.cfg['prop_vid_oloc']}/proposals_list_kbdet.txt"
        else:
            proposal_list_file =f"{self.cfg['prop_vid_oloc']}/proposals_list.txt"
        with open(proposal_list_file) as f:
            lines = f.readlines()
            for line in lines:
                proposal_rel_loc = line.split(" ")[0]
                proposal_name = os.path.basename(proposal_rel_loc)
                proposal_names += [proposal_name]
        
        # Loading the network
        net = self._load_net(self.cfg)


        # Initializing AOLME Validaiton data loader
        tst_data = AOLMEValTrmsDLoader(
            self.cfg['prop_vid_oloc'], proposal_list_file, oshape=(224, 224)
        )
        tst_loader = DataLoader(
            tst_data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=batch_size
        )

        # Resetting maximum memory usage and starting the clock
        torch.cuda.reset_peak_memory_stats(device=0)
        pred_prob_lst = []

        # Starting inference
        start_time = time.time()
        for idx, data in enumerate(Bar(tst_loader)):
            dummy_labels, inputs = (
                data[0].to("cuda:0", non_blocking=True),
                data[1].to("cuda:0", non_blocking=True)
            )

            with torch.no_grad():
                outputs = net(inputs)
                ipred = outputs.data.clone()
                ipred = ipred.to("cpu").numpy().flatten().tolist()
                pred_prob_lst += ipred
        # End of inference

        # Collecting and printing statistics
        end_time = time.time()        
        max_memory_MB = torch.cuda.max_memory_allocated(device=0)/1000000
        print(f"INFO: Total time for batch size of       {batch_size} = {round(end_time - start_time)} sec.")
        print(f"INFO: Max memory usage for batch size of {batch_size} = {round(max_memory_MB, 2)} MB")
        
        # Edit information in the data frame
        for i, proposal_name in enumerate(proposal_names):

            # Calculating class details
            pred = pred_prob_lst[i]
            pred_class_idx = round(pred)
            if pred_class_idx == 1:
                pred_class = "typing"
            else:
                pred_class = "notyping"
            pred_class_prob = round(pred, 2)

            # This is because for 0.5 I am having problems in ROC curve
            if pred_class_prob == 0.5:
                if pred_class_idx == 1:
                    pred_class_prob = 0.51
                else:
                    pred_class_prob = 0.49
                    
            # Adding the class details to to the dataframe
            loc = self.tydf[self.tydf['proposal_name']==proposal_name].index.tolist()
            if len(loc) > 1:
                print(f"Multiple rows having same proposal name! {loc}")
                import pdb; pdb.set_trace()

            self.tydf.loc[loc[0], "activity"] = pred_class
            self.tydf.loc[loc[0], "class_idx"] = pred_class_idx
            self.tydf.loc[loc[0], "class_prob"] = pred_class_prob


        # Saving the csv file
        self.tydf.to_csv(out_file, index=False) 
            
        
        


        

            
    def classify_typing_proposals_roi(self, overwrite=False):
        """Classify each proposed region as typing / no-typing.

        Todo
        ----
        This function evaluates one proposal at a time. This is note
        optimal. I have to redo this to evaluate multiple proposals
        at a time. 
        """
        # if the file is already present load it if overwrite == True
        out_file = f"{self.cfg['oloc']}/tynty-roi-ours-3DCNN_30fps.csv"
        if not overwrite:
            if os.path.isfile(out_file):
                print(f"Reading {out_file}")
                tydf = pd.read_csv(out_file)
                self.tydf = tydf
                return tydf
        
        # Loading neural network into GPU
        print(f"Creating {out_file}")
        net = self._load_net(self.cfg)

        # Loop through each video
        video_names = self.tyrp_roi_only['name'].unique().tolist()
        for i, video_name in enumerate(video_names):

            # Typing proposals for current dataframe
            print(f"Classifying typing in {video_name}")
            tyrp_video = self.tyrp_roi_only[self.tyrp_roi_only['name'] == video_name].copy()
            tyrp_video['activity'] = ""
            tyrp_video['class_idx'] = -1
            tyrp_video['class_prob'] = "-1"
            ivid = pk.Vid(f"{self.cfg['vdir']}/{video_name}", "read")

            # Loop through each instance in the video
            for ii, row in tqdm(tyrp_video.iterrows(), total=tyrp_video.shape[0], desc="INFO: Classifying"):
                
                # Spatio temporal trim coordinates
                bbox = [row['w0'],row['h0'], row['w'], row['h']]
                sfrm = row['f0']
                efrm = row['f1']
                opth = (f"{self.cfg['oloc']}/temp.mp4")

                # Spatio temporal trim and change FPS
                ivid.save_spatiotemporal_trim(sfrm, efrm, bbox, opth, fps=10)

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
                        
                    ipred_class_idx = round(ipred[0])
                    if ipred_class_idx == 1:
                        ipred_class = "typing"
                        ipred_class_prob = round(ipred[0], 2)
                    else:
                        ipred_class = "notyping"
                        ipred_class_prob = 1 - round(ipred[0], 2)
                        
                    # This is because for 0.5 I am having problems in ROC curve
                    if ipred_class_prob == 0.5:
                        if ipred_class_idx == 1:
                            ipred_class_prob = 0.51
                        else:
                            ipred_class_prob = 0.49

                    tyrp_video.at[ii, 'activity'] = ipred_class
                    tyrp_video.at[ii, 'class_prob'] = ipred_class_prob
                    tyrp_video.at[ii, 'class_idx'] = ipred_class_idx


                # Close the vide
                ivid.close()

                        
            # If this is the first time, copy the proposal dataframe to typing dataframe
            # else concatinate
            if i == 0:
                tydf = tyrp_video
            else:
                tydf = pd.concat([tydf, tyrp_video])

        # Save the dataframe
        self.tydf = tydf.copy()
        self.tydf.to_csv(f"{out_file}", index=False)
        return tydf


    def classify_proposals_using_kb_det(self, overwrite=False):
        """ Classify each proposed region as typing / no-typing.
        In this method we use keyboard detection information
        to further improve performance.

        The output is a csv file, "alg2-tynty-ro
        """
        # if the file is already present load it if overwrite == True
        out_file = f"{self.cfg['oloc']}/tynty-roi-ours-3DCNN_kbdet_30fps.csv"
        if not overwrite:
            if os.path.isfile(out_file):
                print(f"Reading {out_file}")
                tydf = pd.read_csv(out_file)
                self.tydf = tydf
                return tydf
        
        # Loading neural network into GPU
        print(f"Creating {out_file}")
        net = self._load_net(self.cfg)

        # Loop through each video
        video_names = self.tyrp_roi_only['name'].unique().tolist()
        for i, video_name in enumerate(video_names):

            # video_name_no_ext
            video_name_no_ext = os.path.splitext(video_name)[0]

            # Loading relavent files
            ivid = pk.Vid(f"{self.cfg['vdir']}/{video_name}", "read")  # Video
            tyrp_video = self.tyrp_roi_only[self.tyrp_roi_only['name'] == video_name].copy()  # Region proposals
            kb_det = pd.read_csv(f"{self.cfg['kb_detdir']}/{video_name_no_ext}_60_det_per_min.csv")


            # Loop through each instance in the video
            print(f"Classifying typing in {video_name}")
            tyrp_video['activity'] = ""
            tyrp_video['class_idx'] = -1
            tyrp_video['class_prob'] = "-1"
            for ii, row in tqdm(tyrp_video.iterrows(), total=tyrp_video.shape[0], desc="INFO: Classifying"):
                
                # Spatio temporal trim coordinates
                bbox = [row['w0'],row['h0'], row['w'], row['h']]
                sfrm = row['f0']
                efrm = row['f1']
                opth = (f"{self.cfg['oloc']}/temp.mp4")

                # Get keyboard detection intersection bounding box
                kb_bbox = self._get_kbdet_intersection(kb_det, sfrm, efrm)

                # Did they overlap?
                iflag, icoords = self._get_intersection(bbox, kb_bbox)

                if not iflag:
                    
                    # If they don't overlap then mark the proposal as notyping
                    # with class probability of 0 <--- Very confident
                    tyrp_video.at[ii, 'activity'] = 'notyping'
                    tyrp_video.at[ii, 'class_prob'] = 0.49
                    tyrp_video.at[ii, 'class_idx'] = 0
                    
                else:
                    
                    # Spatio temporal trim and FPS conversion
                    ivid.save_spatiotemporal_trim(sfrm, efrm, bbox, opth, fps=10)

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

                        ipred_class_idx = round(ipred[0])
                        if ipred_class_idx == 1:
                            ipred_class = "typing"
                            ipred_class_prob = round(ipred[0], 2)
                        else:
                            ipred_class = "notyping"
                            ipred_class_prob = round(ipred[0], 2)

                        # This is because for 0.5 I am having problems in ROC curve
                        if ipred_class_prob == 0.5:
                            if ipred_class_idx == 1:
                                ipred_class_prob = 0.51
                            else:
                                ipred_class_prob = 0.49

                        tyrp_video.at[ii, 'activity'] = ipred_class
                        tyrp_video.at[ii, 'class_prob'] = ipred_class_prob
                        tyrp_video.at[ii, 'class_idx'] = ipred_class_idx
                        


                # Close the vide
                ivid.close()

                        
            # If this is the first time, copy the proposal dataframe to typing dataframe
            # else concatinate
            if i == 0:
                tydf = tyrp_video
            else:
                tydf = pd.concat([tydf, tyrp_video])

        # Save the dataframe
        self.tydf = tydf.copy()
        self.tydf.to_csv(f"{out_file}", index=False)
        return tydf

    def _get_kbdet_intersection(self, kb_det, sfrm, efrm):
        """Determines keyboard detection intersectin bouhnding box between
        sfrm and efrm"""

        # Snipping keyboard detection dataframe between sfrm and efrm
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
    
    def _read_from_disk(self, tyrp_csv):
        """Load the file if it exists"""
        if os.path.isfile(tyrp_csv):
            tyrp = pd.read_csv(tyrp_csv)
            return True, tyrp
        else:
            return False, None

    def _get_3sec_proposal_df(self, roi_df, kdf):
        """Returns a dataframe with typing region proposals using
        1. Table ROI
        2. Keyboard detections

        Parameters
        ----------
        roi_df : Pandas Dataframe
            Table ROI for 3 seconds
        kdf : Pandas Dataframe
            Keyboard detection for 3 seconds
        """

        # ROI column names (persons sitting around the table)
        roidf_temp = roi_df.copy()
        roidf_temp = roidf_temp.drop(['Time', 'f0', 'f', 'video_names'], axis=1)
        persons_list = roidf_temp.columns.tolist()
        
        # Loop over each person ROI and check for keyboard detection
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
        """Returns a dataframe with typing region proposals using
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
        
        # Loop over each person ROI and check for keyboard detection
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
            Keyboard detection dataframe for current duration.
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
        1. We do not have table region of interest or keyboard detection
        2. Keyboard detections should be available for more than half
           of the duration.
        3. ROI should be available for more than half of the duration.

        Parameters
        ----------
        roidf : Pandas DataFrame instance
            ROI dataframe
        detdf : Pandas DataFrame instance
            Keyboard detection instances

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

        # Return True if there is no keyboard detection. If we don't
        # detect keyboard we keep widht and heigh to 0.
        w_sum = detdf_temp['w'].sum()
        if w_sum <= 0:
            return True

        # There should be atleast two keyboard detections, otherwise
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
        




                    
    def _get_union_of_keyboard_detections(self, df, f0, f):
        """ Returns keyboard detection regions using union of all the detections
        in an interval.
        
        Parameters
        ----------
        df : DataFrame
            A DataFrame having keyboard detections.

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
        oclass = "keyboard"
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

    
    def _remove_outside_keyboard_detections(self, df, th=0.5):
        """ Removes all the keyboard detections that are less that are
        50% not inside the table boundary.

        Parameters
        ----------
        df : DataFrame
            DataFrame having keyboard detections with `roi-overlap-ratio`
            column.
        th : Float
            Detectons which are < th are removed.
        """
        for ridx, row in df.iterrows():
            if row['roi-overlap-ratio'] < th:
                df.drop([ridx], inplace = True)
        return df

    
    def _get_roi_overlap_ratio(self, hdf, table_boundary):
        """ Adds a column to keyboard detections, roi-overlap-ratio.
            
            - Table boundary = T
            - Keyboard detection = H
            - Overlap (O) = Intersection(T, H)
                overlap-ratio = Area(O) / Area(H)

        Parameters
        ----------
        hdf : DataFrame
            Dataframe having keyboard detections

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

        # Loop over each keyboard detection
        o_area_ratio_lst = []
        for ridx, row in hdf.iterrows():

            # keyboard detection is loaded into proper variables
            [hw0, hh0, hw, hh] = [row['w0'], row['h0'], row['w'], row['h']]
            h_area = hw*hh

            # Creating a binary image with keyboard detection as 1
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

            # Drop the keyboard detection if the overlap area is less than
            # 50% of keyboard area
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
            A data frame having roi entries for `self.tydur`.
        """

        # Dropping unnecessary columns
        df.drop(['Time', 'f0', 'video_names', 'f'], axis = 1, inplace=True)

        # Looping over each column and if atleast one column contain 2 valid entries
        # return true
        for col in df.columns.tolist():
            
            valid_bboxes = 0
            for i in range(0, self.tydur):

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
            A data frame having roi entries for `self.tydur`.
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
        # net = DyadicCNN3DV2(depth, [3, 30, 224, 224])
        net = DyadicCNN3DV2(depth, cfg['input_shape'].copy())
        net.to("cuda:0")

        # Print summary of network.
        summary(net, tuple(cfg['input_shape'].copy()))

        # Loading the net with trained weights to cuda device 0
        ckpt_weights = torch.load(ckpt)
        net.load_state_dict(ckpt_weights['model_state_dict'])
        net.eval()

        return net
        
        

    def _check_for_typing(self, proposal_df):
        """ Checks for typing in the proposal data frame.

        Parameters
        ----------
        proposal_df: DataFrame
            Proposal dataframe having keyboards bounding boxes.
        Todo
        ----
        1. Here I am trimming -> typing to hdd -> loading. This is not
           recommended for speed. Please try to improve.
        """
        import pdb; pdb.set_trace()
        # Loop over proposal dataframe
        for i, row in proposal_df.iterrows():

            # if w or h == 0 then there is no keyboards
            if row['w'] == 0 or row['h'] == 0:
                proposal_df.at[i, 'typing'] = -1
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
                        proposal_df.at[i, 'typing'] = round(ipred[0])
                        
        return proposal_df



    def _get_proposal_df(self, bboxes, tydur):
        """
        OBSOLETE SHOULD BE DELETED IN CLEANUP PHASE.
        Creates a data frame with each row representing 3 seconds.

        Parameters
        ----------
        bboxes: str
            path to file having keyboards bounding boxes
        tydur: int, optional
            Each typing instance duration considered in seconds. 
            Defaults to 3.
        """
        # Video properties
        num_frames = self._vid.props['num_frames']
        fps = self._vid.props['frame_rate']

        # Creating f0 and f lists
        num_trims = math.floor(num_frames/(tydur*fps))
        f0 = [x*(tydur*fps) for x in range(0, num_trims)]
        f = [tydur*fps]*num_trims

        # Creating W, H and FPS lists
        W = [self._vid.props['width']]*num_trims
        H = [self._vid.props['height']]*num_trims
        fps_lst = [fps]*num_trims

        # Get bounding boxes
        w0, h0, w, h = self._get_proposal_bboxes(bboxes, f0, f)

        # Intializing all typing instances are marked nan(numpy)
        typing_lst = [np.nan]*(num_trims)

        # Creating data frame with all the lists
        df = pd.DataFrame(list(zip(W, H, fps_lst, f0, f, w0, h0, w, h, typing_lst)),
                          columns=["W","H", "FPS", "f0", "f", "w0", "h0", "w", "h", "typing"])
        return df


    
    def write_to_csv(self):
        """ Writes typing instances to a csv file. The name of the file is `<video name>_wr_using_alg.csv` and has
        following columns,

            1. f0      : poc of starting frame
            2. f       : number of frames
            3. W, H    : Video width and height
            4. w0, h0  : Bounding box top left corner
            5. w, h    : width and height of bounding box
            6. FPS     : Frames per second
            7. typing : {-1, 0, 1}.
                -1 => Keyboards not found
                0  => notyping
                1  => typing

        """
        # Update typing instances in typing dataframe by processing
        # valid instances to 0 or 1
        self.tydf = self._check_for_typing(self._proposal_df.copy())
        
        vname = self._vid.props['name']
        vloc = self._vid.props['dir_loc']
        csv_pth = f"{vloc}/{vname}_wr_using_alg.csv"
        self.tydf.to_csv(csv_pth)
        

        
    def _get_proposal_bboxes(self, bboxes, f0_lst, f_lst):
        """ Creates proposal bounding boxes. Trims from these bounding boxes are
        later processed via typing detection algorithm.

        If there are multiple bounding boxes in the duration we consider the
        union of bounding boxes.

        Parameters
        ----------
        bboxes: str
            Path to file having keyboards detection bounding boxes
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
