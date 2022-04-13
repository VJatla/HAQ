import os
import pdb
import cv2
import aqua
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from aqua.nn.dloaders import AOLMETrmsDLoader
import pytkit as pk
from aqua.nn.models import DyadicCNN3D
from torchsummary import summary


class Writing:
    wrdf = pd.DataFrame()
    """ Data frame having information on writing instances. 
    It has following columns,
    ```
    1. f0     : poc of starting frame
    2. f      : number of frames
    3. W, H   : Video width and height
    4. w0, h0 : Bounding box top left corner
    5. w, h   : width and height of bounding box
    6. FPS    : Frames per second
    7. writing : {-1, 0, 1}.
                -1 => Hands not found
                0 => nowriting
                1 => writing
    ```
    """

    cfg = {}
    """ Configuration dictionary """

    ivid = None
    """ Input video instance. """

    wrdur = 0
    """ Duration of spati-temporal trims to classify. """
    
    
    
    def __init__(self, cfg, wrdur=3):
        """ Spatio temporal writing detection using,

        Parameters
        ----------
        cfg : Str
        Configuration file. The configuration has the following entries.
        
            1. "video": Video path for which we have hand detections.
            2. "hands": Hand detections as CSV file
            3. "table_roi": CSV file having Table ROI marked
            4. "depth": Depth of 3D-CNN that provided the best performance,
            5. "ckpt": Checkpoint of 3D-CNN
        
        """

        # Configuration dictionary
        self.cfg = cfg
        
        # wrdur
        self.wrdur = wrdur
        
        # Checking configuration file
        self._check_cfg(cfg)

        # Loading video
        self.ivid = pk.Vid(cfg['video'], 'read')
        
        
    def get_writing_proposals(self):
        """ Generates hand proposal regions by using hand detections
        and manually annotated table region of interest.

        Parameters
        ----------
        hands_csv_path : str
            Path of configuration file
        table_roi_path : str
            Path of csv file containing manually annotated table regions.
        """
        
        # Creating an empty writing region proposal dataframe
        wrp = pd.DataFrame()

        # Load hands and table roi csvs
        hands_df = pd.read_csv(self.cfg['hands'])
        troi_df = pd.read_csv(self.cfg['table_roi'])
        troi_df = troi_df[troi_df['video_names'] == f"{self.ivid.props['name']}{self.ivid.props['extension']}"].copy()
        
        # Calcuating hand region proposals every 3 seconds
        fps = hands_df['FPS'].unique().item()
        new_rows = []
        for i in tqdm(range(0, troi_df['f0'].max(), 3*fps), desc="INFO: Writing proposals"):

            # Get hands and table regions for 3 seconds
            hdf = hands_df[hands_df['f0'].between(i, i + 3*fps - 1)].copy()
            tdf = troi_df[troi_df['f0'].between(i, i + 3*fps - 1)].copy()

            # Skip,
            # 1. if the current 3 seconds do not have sufficient regions of interest.
            # 2. if the hands dataframe is empty
            if (not self._have_sufficient_rois(tdf.copy())) or (len(hdf) == 0):
                continue

            # Calculating table boundary from regions of interest
            table_boundary = self._get_table_boundary(tdf.copy())

            # Calculating overlap ratio between table bondary and hand detections
            hdf = self._get_roi_overlap_ratio(hdf, table_boundary)

            # Remove hand regions that do not corss a overlap ratio threshold
            hdf = self._remove_outside_hand_detections(hdf.copy())

            # Skip to next 3 seconds if all the hand regions lie outside
            # the table boundary
            if len(hdf) == 0:
                continue

            # Calculate hand region proposal rows for 3 seconds by tabking union
            # of all the valid hand detections
            new_rows += self._get_union_of_hand_detections(hdf.copy(), i, 3*fps)


        wrp = pd.DataFrame(new_rows, columns=[
                'W', 'H', 'FPS', 'f0', 'f', 'class', 'table_boundary',
                'w0', 'h0', 'w', 'h'
            ])
            
        return wrp

    def classify_writing_proposals(self, wrp, debug=True):
        """ Classify each proposed region as writing / no-writing.

        Todo
        ----
        This function evaluates one proposal at a time. This is note
        optimal. I have to redo this to evaluate multiple proposals
        at a time. 

        Parameters
        ----------
        wrp : Pandas DataFrame instance
            DataFrame having writing region proposals
        debug : Bool
            When True we will save the class probability and trimmed
            videos.

        Return
        ------
        A DataFrame with `writing` column with labels 1 and 0 for
        writing and no-writing respectively.
        """
        
        # Output location to save trims and writing csv file
        oloc = self.ivid.props['dir_loc']
        
        # Creating another column in writing region proposal dataframe
        wrp['writing'] = -1
        if debug:
            print(f"INFO: RUNNING IN DEBUG MODE!")
            wrp['p'] = -1
            wrp['trim_path'] = ""
            os.system(f"rm -r {oloc}/trims")
            os.system(f"mkdir -p {oloc}/trims/0")
            os.system(f"mkdir -p {oloc}/trims/1")
        
        # Loading neural network into GPU
        net = self._load_net(self.cfg)

        # Loop through writing proposal
        for i, row in tqdm(wrp.iterrows(), total=wrp.shape[0], desc="INFO: Classifying"):

            # Spatio temporal trim coordinates
            bbox = [row['w0'],row['h0'], row['w'], row['h']]
            sfrm = row['f0']
            efrm = sfrm + row['f']
            opth = (f"{oloc}/temp.mp4")

            # Spatio temporal trim
            self.ivid.save_spatiotemporal_trim(sfrm, efrm, bbox, opth)

            # Creating a temporary text file `temp.txt` having
            # temp.mp4 and a dummy label (100)
            with open(f"{oloc}/temp.txt", "w") as f:
                f.write("temp.mp4 100")

                    

            # Intialize AOLME data loader instance
            tst_data = AOLMETrmsDLoader(
                oloc, f"{oloc}/temp.txt", oshape=(224, 224)
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
                    wrp.at[i, 'writing'] = round(ipred[0])
                    if debug:
                        trim_pth = f"{oloc}/trims/{round(ipred[0])}/trim_{i}.mp4"
                        os.system(f"cp {oloc}/temp.mp4 {trim_pth}")
                        wrp.at[i, 'p'] = ipred[0]
                        wrp.at[i, 'trim_path'] = f"{round(ipred[0])}/trim_{i}.mp4"
        return wrp

                    
    def _get_union_of_hand_detections(self, df, f0, f):
        """ Returns hand detection regions using union of all the detections
        in an interval.
        
        Parameters
        ----------
        df : DataFrame
            A DataFrame having hand detections.

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
        oclass = df['class'].unique().item()
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

            

            


        


    def _remove_outside_hand_detections(self, df, th=0.5):
        """ Removes all the hand detections that are less that are
        50% not inside the table boundary.

        Parameters
        ----------
        df : DataFrame
            DataFrame having hand detections with `roi-overlap-ratio`
            column.
        th : Float
            Detectons which are < th are removed.
        """
        for ridx, row in df.iterrows():
            if row['roi-overlap-ratio'] < th:
                df.drop([ridx], inplace = True)
        return df
            
    def _get_roi_overlap_ratio(self, hdf, table_boundary):
        """ Adds a column to hand detections, roi-overlap-ratio.
            
            - Table boundary = T
            - Hand detection = H
            - Overlap (O) = Intersection(T, H)
                overlap-ratio = Area(O) / Area(H)

        Parameters
        ----------
        hdf : DataFrame
            Dataframe having hand detections

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

        # Loop over each hand detection
        o_area_ratio_lst = []
        for ridx, row in hdf.iterrows():

            # hand detection is loaded into proper variables
            [hw0, hh0, hw, hh] = [row['w0'], row['h0'], row['w'], row['h']]
            h_area = hw*hh

            # Creating a binary image with hand detection as 1
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

            # Drop the hand detection if the overlap area is less than
            # 50% of hand area
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
            A data frame having roi entries for `self.wrdur`.
        """

        # Dropping unnecessary columns
        df.drop(['Time', 'f0', 'video_names', 'f'], axis = 1, inplace=True)

        # Looping over each column and if atleast one column contain 2 valid entries
        # return true
        for col in df.columns.tolist():
            
            valid_bboxes = 0
            for i in range(0, self.wrdur):

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
            A data frame having roi entries for `self.wrdur`.
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
        

    
    def _check_cfg(self, cfg):
        """ Validates the configuration file."""

        # Check files
        pk.check_file_existance(cfg['video'])
        pk.check_file_existance(cfg['hands'])
        pk.check_file_existance(cfg['table_roi'])
        pk.check_file_existance(cfg['ckpt'])
        
        # Depth should be between 2 and 4
        if cfg['depth'] < 2 or cfg['depth'] > 4:
            raise Exception(
                "USER EXCEPTION: The dyadic depth is not valid."
                f"\n\t Depth: {cfg['depth']}"
            )


        
    def _load_net(self, cfg):
        """ Load neural network to GPU. """

        print("INFO: Loading Trained network to GPU ...")

        # Checkpoint and depth from cfg file.
        ckpt = cfg['ckpt']
        depth = cfg['depth']

        # Creating an instance of Dyadic 3D-CNN
        net = DyadicCNN3D(depth, [3, 90, 224, 224])
        net.to("cuda:0")

        # Print summary of network.
        # summary(net, (3, 90, 224, 224))

        # Loading the net with trained weights to cuda device 0
        ckpt_weights = torch.load(ckpt)
        net.load_state_dict(ckpt_weights['model_state_dict'])
        net.eval()

        return net
        
        

    def _check_for_writing(self, proposal_df):
        """ Checks for writing in the proposal data frame.

        Parameters
        ----------
        proposal_df: DataFrame
            Proposal dataframe having hands bounding boxes.
        Todo
        ----
        1. Here I am trimming -> writing to hdd -> loading. This is not
           recommended for speed. Please try to improve.
        """
        import pdb; pdb.set_trace()
        # Loop over proposal dataframe
        for i, row in proposal_df.iterrows():

            # if w or h == 0 then there is no hands
            if row['w'] == 0 or row['h'] == 0:
                proposal_df.at[i, 'writing'] = -1
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
                        proposal_df.at[i, 'writing'] = round(ipred[0])
                        
        return proposal_df



    def _get_proposal_df(self, bboxes, wrdur):
        """
        OBSOLETE SHOULD BE DELETED IN CLEANUP PHASE.
        Creates a data frame with each row representing 3 seconds.

        Parameters
        ----------
        bboxes: str
            path to file having hands bounding boxes
        wrdur: int, optional
            Each writing instance duration considered in seconds. 
            Defaults to 3.
        """
        # Video properties
        num_frames = self._vid.props['num_frames']
        fps = self._vid.props['frame_rate']

        # Creating f0 and f lists
        num_trims = math.floor(num_frames/(wrdur*fps))
        f0 = [x*(wrdur*fps) for x in range(0, num_trims)]
        f = [wrdur*fps]*num_trims

        # Creating W, H and FPS lists
        W = [self._vid.props['width']]*num_trims
        H = [self._vid.props['height']]*num_trims
        fps_lst = [fps]*num_trims

        # Get bounding boxes
        w0, h0, w, h = self._get_proposal_bboxes(bboxes, f0, f)

        # Intializing all writing instances are marked nan(numpy)
        writing_lst = [np.nan]*(num_trims)

        # Creating data frame with all the lists
        df = pd.DataFrame(list(zip(W, H, fps_lst, f0, f, w0, h0, w, h, writing_lst)),
                          columns=["W","H", "FPS", "f0", "f", "w0", "h0", "w", "h", "writing"])
        return df


    
    def write_to_csv(self):
        """ Writes writing instances to a csv file. The name of the file is `<video name>_wr_using_alg.csv` and has
        following columns,

            1. f0      : poc of starting frame
            2. f       : number of frames
            3. W, H    : Video width and height
            4. w0, h0  : Bounding box top left corner
            5. w, h    : width and height of bounding box
            6. FPS     : Frames per second
            7. writing : {-1, 0, 1}.
                -1 => Hands not found
                0  => nowriting
                1  => writing

        """
        # Update writing instances in writing dataframe by processing
        # valid instances to 0 or 1
        self.wrdf = self._check_for_writing(self._proposal_df.copy())
        
        vname = self._vid.props['name']
        vloc = self._vid.props['dir_loc']
        csv_pth = f"{vloc}/{vname}_wr_using_alg.csv"
        self.wrdf.to_csv(csv_pth)
        

        
    def _get_proposal_bboxes(self, bboxes, f0_lst, f_lst):
        """ Creates proposal bounding boxes. Trims from these bounding boxes are
        later processed via writing detection algorithm.

        If there are multiple bounding boxes in the duration we consider the
        union of bounding boxes.

        Parameters
        ----------
        bboxes: str
            Path to file having hands detection bounding boxes
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
