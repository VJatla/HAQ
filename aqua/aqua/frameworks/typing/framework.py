import os
import pdb
import cv2
import aqua
import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from aqua.nn.dloaders import AOLMETrmsDLoader


class Typing:
    tydf = pd.DataFrame()
    """ Data frame having information on typing instances. 
    It has following columns,
    ```
    1. f0     : poc of starting frame
    2. f      : number of frames
    3. W, H   : Video width and height
    4. w0, h0 : Bounding box top left corner
    5. w, h   : width and height of bounding box
    6. FPS    : Frames per second
    7. typing : {-1, 0, 1}.
                -1 => Keyboard not found
                0 => notyping
                1 => typing
    ```
    """
    
    def __init__(self, video, bboxes, net, ckpt, tydur=3):
        """ Spatio temporal typing detection using,
        - Keyboard bounding boxes
        - Activity recognition model check point trained in pytorch.

        We classify each trim created taking union of keyboard detections for
        every `tydur`seconds. It uses GPU 0.

        Parameters
        ----------
        video: str
            Video path
        bboxes: str
            Path to CSV file having keyboard bounding boxes
        net: Custom Network Instance
            Instance of pytorch custom network we will be 
            training
        ckpt: str
            Path to check point file that is trained on typing/notyping
        tydur: int, optional
            Each typing instance duration considered in seconds. 
            Defaults to 3.

        This code expects the bounding box csv file to have following
        columns,
        ```
        1. f0
        2. W, H
        3. w0, h0
        4. w, h
        5. FPS
        ```
        
        """
        # Check if all the files exist
        aqua.fd_ops.files_exist([video, bboxes, ckpt])

        # Loading nn weights and pushing it to GPU memory
        self._net = net
        self._net.to("cuda:0")
        checkpoint = torch.load(ckpt)
        self._net.load_state_dict(checkpoint['model_state_dict'])
        self._net.eval()

        # Loading video
        self._vid = aqua.video_tools.Vid(video)

        # Proposal dataframe having keyboard bounding boxes.
        self._proposal_df = self._get_proposal_df(bboxes, tydur)



    def _check_for_typing(self, proposal_df):
        """ Checks for typing in the proposal data frame.

        Parameters
        ----------
        proposal_df: DataFrame
            Proposal dataframe having keyboard bounding boxes.
        Todo
        ----
        1. Here I am trimming -> writing to hdd -> loading. This is not
           recommended for speed. Please try to improve.
        """
        import pdb; pdb.set_trace()
        # Loop over proposal dataframe
        for i, row in proposal_df.iterrows():

            # if w or h == 0 then there is no keyboard
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
        """ Creates a data frame with each row representing 3 seconds.

        Parameters
        ----------
        bboxes: str
            path to file having keyboard bounding boxes
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
        """ Writes typing instances to a csv file.

        The name of the file is <video name>_ty_using_alg.csv and has
        following columns,
        ```
        1. f0     : poc of starting frame
        2. f      : number of frames
        3. W, H   : Video width and height
        4. w0, h0 : Bounding box top left corner
        5. w, h   : width and height of bounding box
        6. FPS    : Frames per second
        7. typing : {-1, 0, 1}.
                    -1 => Keyboard not found
                     0 => notyping
                     1 => typing
        ```
        """
        # Update typing instances in typing dataframe by processing
        # valid instances to 0 or 1
        self.tydf = self._check_for_typing(self._proposal_df.copy())
        
        vname = self._vid.props['name']
        vloc = self._vid.props['dir_loc']
        csv_pth = f"{vloc}/{vname}_ty_using_alg.csv"
        self.tydf.to_csv(csv_pth)
        

    def _get_proposal_bboxes(self, bboxes, f0_lst, f_lst):
        """ Creates proposal bounding boxes. Trims from these bounding boxes are
        later processed via typing detection algorithm.

        If there are multiple bounding boxes in the duration we consider the
        union of bounding boxes.

        Parameters
        ----------
        bboxes: str
            Path to file having keyboard detection bounding boxes
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
