import os
import pdb
import aqua
import random
import shutil
import numpy as np
import pandas as pd
from aqua.data_tools.aolme.trimmed_videos.tsn import TSNDataTools
from aqua.data_tools.aolme.trimmed_videos.data_splitter import DSplitter


class AOLMETrimmedVideos:
    _paths = []

    def __init__(self, rdir, vext):
        """ Methods that operate on trimmed activity videos.

        Parameters
        ----------
        rdir: str
            Root directory that has trimmed videos. The instance assumes
            a directory strcuture. For example a directory that has
            three activities with three video samples per activity has
            following directory structure:
            ```bash
            ├── act1
            │   ├── v1.mp4
            │   ├── v2.mp4
            │   └── v3.mp4
            ├── act2
            │   ├── v1.mp4
            │   ├── v2.mp4
            │   └── v3.mp4
            └── act3
                ├── v1.mp4
                ├── v2.mp4
                └── v3.mp4
            ```
        vext: str
            Extension of trimmed videos being processed
        """
        self._paths = aqua.fd_ops.get_file_paths_with_kws(rdir, vext)

    def extract_images_for_classification(self,
                                          odir,
                                          class_label,
                                          split_info_file,
                                          imgs_per_vid=1):
        """ Extract images for classification and stores in
        corresponding split (trn, tst, val).

        Parameters
        ----------
        odir: str
            output directory
        class_label: str
            Class label of the frame we are extracting.
        split_info_file: str
            Path to file having trn, val and tst splits information
        imgs_per_vid: int, optional
            Number of images to extract from each video        
        """

        # Loop through each video and extract frames
        for tvpath in self._paths:

            # Creating video instance
            tv = aqua.video_tools.Vid(tvpath)

            # get split label for current video
            split_label = self._get_split_label(tv.props['name'],
                                                split_info_file)

            # image saving directory and path
            save_dir = (f"{odir}/{split_label}/{class_label}")
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            # Choosing frames that are 10 frames apart starting from
            # 10th frame (maximum = 8 frames for 90 frame video)
            frm_numbers = list(range(10, int(tv.props['num_frames']),
                                     10))[0:imgs_per_vid]

            for frm_number in frm_numbers:
                save_pth = (f"{save_dir}/{tv.props['name']}_{frm_number}.png")
                tv.extract_frame(frm_number, save_dir)

    def _get_split_label(self, vname, split_info_file):
        """ Get trimmed video split label
        
        Parameters
        ----------
        split_info_file: str
            Path to file having trn, val and tst splits information
        """

        # Load split information as data frame
        split_info_df = pd.read_csv(split_info_file)

        # Extract information from name
        info_from_name = aqua.data_tools.aolme.name_parser.parse_video_name(
            vname)

        # Creating search string to search in split information data frame
        group_str = (f"C{info_from_name['cohort']}"
                     f"L{info_from_name['level']}"
                     f"{info_from_name['school']}"
                     f"-{info_from_name['group']}")

        # Select corresponding row from split information data frame
        cvid_split_info = split_info_df.loc[
            (split_info_df['group'] == group_str)
            & (split_info_df['date'] == info_from_name['date'])].copy()

        # If no hits or more than 1 hit throw error
        if not len(cvid_split_info) == 1:
            print(cvid_split_info)
            raise Exception(f"{vname}")

        # return label
        return cvid_split_info['label'].item()

    def create_tsn_tvt_lists(self, rfdir, odir, labels_dict, split_info_file):
        """
        Creates training, validation and testing lists from trimmed
        videos in the format TSN understands. 

        It specifically creates the following text files,
        1. trn_videos.txt
        2. trn_rawframes.txt
        3. val_videos.txt
        4. val_rawframes.txt
        5. tst_videos.txt
        6. tst_rawframes.txt

        *_videos.txt file have following entries,
        ```
        <video directory>/<video name> <numerical label>
        ```
        *_rawframes.txt file has following entries,
        ```
        <video directory>/<video name> <num frames -1> <numerical label>
        ```

        Parameters
        ----------
        rfdir: str
            Path to directory having raw frames (extracted using TSN tools)
        odir: str
            Output directory path
        labels_dict: Dict
            A dictionary having activity and corresponding label
        split_info_file: str
            Path to file having trn, val and tst splits information
        """
        # Creating tsn data tools instance
        tsn_dtools = TSNDataTools(self._paths.copy(), labels_dict,
                                  split_info_file)

        # Creating video list files
        tsn_dtools._create_video_list_files(odir)

        # Creating rawframe list files
        tsn_dtools._create_rawframes_list_files(rfdir, odir)

    def create_video_tvt_lists(self, odir, labels_dict, split_info_file):
        """
        Creates training, validation and testing lists from trimmed
        videos in the format `mmaction2` understands. 

        It specifically creates the following text files,
        1. trn_videos.txt
        3. val_videos.txt
        5. tst_videos.txt

        *_videos.txt file have following entries,
        ```
        <video directory>/<video name> <numerical label>
        ```

        Parameters
        ----------
        odir: str
            Output directory path
        labels_dict: Dict
            A dictionary having activity and corresponding label
        split_info_file: str
            Path to file having trn, val and tst splits information
        """
        # Creating tsn data tools instance
        tsn_dtools = TSNDataTools(self._paths.copy(), labels_dict,
                                  split_info_file)

        # Creating video list files
        tsn_dtools._create_video_list_files(odir)

    def create_subsampled_tvt_lists(self, odir, labels_dict, split_info_file,
                                    samples_per_act):
        """
        Creates training, validation and testing subsampled lists from trimmed
        videos in the format `mmaction2` understands. 

        It specifically creates the following text files,
        1. trn_videos.txt
        3. val_videos.txt
        5. tst_videos.txt

        *_videos.txt file have following entries,
        ```
        <video directory>/<video name> <numerical label>
        ```

        Parameters
        ----------
        odir: str
            Output directory path
        labels_dict: Dict
            A dictionary having activity and corresponding label
        split_info_file: str
            Path to file having trn, val and tst splits information
        samples_per_act: Tuple of ints
            (<no. trn samples>, <no. val samples>, <no. tst samples>). To
            use all samples give (-1,-1,-1)
        """
        # Creating tsn data tools instance
        tsn_dtools = TSNDataTools(self._paths.copy(), labels_dict,
                                  split_info_file)

        # Creating video list files
        tsn_dtools._create_subsampled_video_list_files(odir, samples_per_act)




    def check_videos(self):
        """ checks all the videos 
        """
        invalid_videos = []
        for pth in self._paths:

            # Create video object
            vid = aqua.video_tools.Vid(pth)

            # List of invalid videos
            if vid.props['num_frames'] == 0:
                invalid_videos += [pth]

        # Print properties of invalid videos
        for pth in invalid_videos:

            # Create video object
            vid = aqua.video_tools.Vid(pth)

            # Printing properties
            print(vid.props)

        print(f"Invalid videos:\n{invalid_videos}")

    def resize(self, vsize, odir):
        """
        This method resizes all trimmed videos to `visze` on the
        longer edge.

        Parameters
        ----------
        vsize: int
            Long edge size required
        odir: str
            Output directory
        """
        # if output directory does not exist make it
        if not os.path.isdir(odir):
            os.makedirs(odir)

        for pth in self._paths:
            vid = aqua.video_tools.Vid(pth)

            # Output full path
            opth = (
                f"{odir}/"
                f"{vid.props['name']}_rzto_{vsize}{vid.props['extension']}")
            vid.resize_on_longer_edge(vsize, opth)


    def create_leave_onegroup_out_lists(self, groups, odir, labels_dict, split_info_file):
        """
        Creates `(trn, val).txt` files having samples under `odir/<group>` directory. The
        `trn.txt` file has trimmed videos from all the groups except for `<group>`, 
        while `val.txt` has trimmed videos from `<group>`.

        Parameters
        ----------
        groups: list of str
            Groups that are considered for training. If it is empty all groups
            are considered.
        odir: str
            Output directory path
        labels_dict: Dict
            A dictionary having activity and corresponding label
        split_info_file: str
            Path to file having trn, val and tst splits information
        """
        # If output directory does not exist crete it
        if not os.path.isdir(odir):
            print(f"INFO: Creating {odir}")
            os.makedirs(odir)
        else:
            print(f"INFO: Removing directory")
            shutil.rmtree(odir)
            print(f"INFO: Creating {odir}")
            os.makedirs(odir)
            
        # Creating data splitting instance
        splitter = DSplitter(self._paths.copy(), groups, labels_dict,
                          split_info_file)
        splitter._create_group_leave_one_out(odir)
