""" 
DEPRICATED

NOTE: THIS SHALL BE DEPRICATED WITH, `TrnValTstSplits` class
"""
import re
import os
import pdb
import math
import aqua
import random
import pandas as pd


class TSNDataTools:

    _tvdf = pd.DataFrame()
    """ List of tuples having information of trimmed videos 
    (<activity label>, <video name>, <activity numeric label>, <split label>)
    """
    def __init__(self, vpaths, ldict, split_info_file):
        """ 
        Methods to prepare AOLME trimmed videos in the format TSN
        requires.

        Parameters
        ----------
        vpaths: array of strings
            List of paths having trimmed videos location
        ldict: Dict
            A dictionary having activity and corresponding label
        split_info_file: str
            Path to file having trn, val and tst splits information
        """
        self._splitdf = pd.read_csv(split_info_file)

        # Get video labels array
        self._tvdf = self._get_video_names_and_labels(vpaths, ldict)

        # Shuffle
        self._tvdf.sample(frac=1).reset_index(drop=True)

    def _get_video_names_and_labels(self, paths, ldict):
        """ Returns a list having tuples with following information
        (<activity label>, <video name>, <activity numeric label>, <split label>)
        
        Parameters
        ----------
        ldict: Dict
            A dictionary having activity and corresponding label
        paths: list of strings
            List having trimmed video paths.
        """
        vnames = [os.path.basename(x) for x in paths]
        vlabels = [os.path.split(os.path.dirname(x))[1] for x in paths]
        vnlabels = [ldict[os.path.split(os.path.dirname(x))[1]] for x in paths]

        # Split label, {"trn", "val", "tst"}
        split_label = [self._get_split_label(x) for x in vnames]

        # session+person label, ex: C1L1P-A-Feb27-Kid11
        splabel = [self._get_session_and_person_label(x) for x in vnames]

        # Zipping
        tvlist_of_tuples = list(
            zip(vlabels, vnames, vnlabels, split_label, splabel))

        # Converting each tuple to list
        tvlist = [list(a) for a in tvlist_of_tuples]

        # Convert to dataframe
        columns = ['activity', 'vname', 'num_label', 'split', 'splabel']
        df = pd.DataFrame(tvlist, columns=columns)

        return df

    def _get_session_and_person_label(self, vname):
        """ Returns a label that has information about 
        person and session of current activty instance
        from video name. For example, C1L1P-A-Feb27-Kid11.
        tv_0_G-C2L1W-Feb27-B-Issac_q2_01-04_30fps_4828_4918_Kidx_rzto_224.mp4

        Todo
        ----
        - Support for multiple cameras is available.
        """
        vname = vname.replace("_", "-")
        vsplit = vname.split("-")

        splabel = f"{vsplit[3]}-{vsplit[4]}-{vsplit[5]}-{vsplit[13]}"

        return splabel

    def _get_split_label(self, vname):
        """ Get trimmed video split label
        
        Parameters
        ----------
        vname: str
            Trimmed video name
        """

        # Extract information from name
        info_from_name = aqua.data_tools.aolme.name_parser.parse_video_name(
            vname)

        # Creating search string to search in split information data frame
        group_str = (f"C{info_from_name['cohort']}"
                     f"L{info_from_name['level']}"
                     f"{info_from_name['school']}"
                     f"-{info_from_name['group']}")

        # Select corresponding row from split information data frame
        cvid_split_info = self._splitdf.loc[
            (self._splitdf['group'] == group_str)
            & (self._splitdf['date'] == info_from_name['date'])].copy()

        # If no hits or more than 1 hit throw error
        if not len(cvid_split_info) == 1:
            print(cvid_split_info)
            raise Exception(f"{vname}")

        # return label
        return cvid_split_info['label'].item()

    def _create_rawframes_list_files(self, rfdir, odir, oflow=True):
        """
        NOTE: NEEDS TO BE UPDATED

        creates a text files having entries in the following format,
        ```
        <video directory>/<video name> <num frames> <numerical label>
        ```
        It creates the following files, {trn, val, tst}_rawframes.txt

        Parameters
        ----------
        rfdir: str
            Path to directory having raw frames (extracted using TSN tools)
        opath: str
            Output text file path
        oflow: bool
            Do we use optical flow?
        """
        print(f"INFO: Creating {odir}/trn_rawframes.txt")
        print(f"INFO: Creating {odir}/val_rawframes.txt")
        print(f"INFO: Creating {odir}/tst_rawframes.txt")

        trn_f = open(f"{odir}/trn_rawframes.txt", "w")
        val_f = open(f"{odir}/val_rawframes.txt", "w")
        tst_f = open(f"{odir}/tst_rawframes.txt", "w")

        for tvtuple in self._tvtuples:

            vlabel, vname, vnlabel, slabel = tvtuple

            # Calculate number of freames for current video
            rfpath = f"{rfdir}/{vlabel}/{vname}"
            if not os.path.isdir(rfpath):
                raise Exception(f"{rfpath} does not exist.")

            num_frames = len(
                aqua.fd_ops.get_file_paths_with_kws(rfpath, [".jpg"]))
            if num_frames == 0:
                raise Exception(f"{rfpath} has no images")

            if oflow:
                num_frames = int(num_frames / 3)

            # Create line to write
            line = f"{vlabel}/{vname} {num_frames} {vnlabel}\n"

            if slabel == "trn":
                trn_f.write(line)

            elif slabel == "val":
                val_f.write(line)

            elif slabel == "tst":
                tst_f.write(line)

            else:
                raise Exception(f"ERROR: Incorrect split label {slabel}")

        trn_f.close()
        val_f.close()
        tst_f.close()

    def _create_video_list_files(self, odir):
        """
        NOTE: NEEDS TO BE UPDATED

        creates a text files having entries in the following format,
        ```
        <video directory>/<video name> <numerical label>
        ```
        It creates the following files, {trn, val, tst}_videos.txt

        Parameters
        ----------
        opath: str
            Output text file path
        """
        print(f"INFO: Creating {odir}/trn_videos.txt")
        print(f"INFO: Creating {odir}/val_videos.txt")
        print(f"INFO: Creating {odir}/tst_videos.txt")

        trn_f = open(f"{odir}/trn_videos.txt", "w")
        val_f = open(f"{odir}/val_videos.txt", "w")
        tst_f = open(f"{odir}/tst_videos.txt", "w")

        for tvtuple in self._tvtuples:

            vlabel, vname, vnlabel, slabel = tvtuple
            line = f"{vlabel}/{vname} {vnlabel}\n"

            if slabel == "trn":
                trn_f.write(line)

            elif slabel == "val":
                val_f.write(line)

            elif slabel == "tst":
                tst_f.write(line)

            else:
                raise Exception(f"ERROR: Incorrect split label {slabel}")

        trn_f.close()
        val_f.close()
        tst_f.close()

    def _create_subsampled_video_list_files(self, odir, num_samples):
        """
        creates a text files having entries in the following format,
        ```
        <video directory>/<video name> <numerical label>
        ```
        It creates the following files, {trn, val, tst}_videos.txt

        Parameters
        ----------
        opath: str
            Output text file path
        num_samples: Tuple of ints 
            Representing number of samples per activity.
            (<no. trn samples>, <no. val samples>, <no. tst samples>)
            To extract all samples give (-1, -1, -1)
        """
        num_trn_samples = num_samples[0]
        num_val_samples = num_samples[1]
        num_tst_samples = num_samples[2]

        # if -ve number of samples become infinite
        if num_trn_samples < 0:
            num_trn_samples = math.inf
            trn_fname = f"trn_videos_all.txt"
        else:
            trn_fname = f"trn_videos_{num_trn_samples}per_act.txt"
        if num_val_samples < 0:
            num_val_samples = math.inf
            val_fname = f"val_videos_all.txt"
        else:
            val_fname = f"val_videos_{num_val_samples}per_act.txt"
        if num_tst_samples < 0:
            num_tst_samples = math.inf
            tst_fname = f"tst_videos_all.txt"
        else:
            tst_fname = f"tst_videos_{num_tst_samples}per_act.txt"

        print(f"INFO: Creating {odir}/{trn_fname}")
        print(f"INFO: Creating {odir}/{val_fname}")
        print(f"INFO: Creating {odir}/{tst_fname}")

        trn_f = open(f"{odir}/{trn_fname}", "w")
        val_f = open(f"{odir}/{val_fname}", "w")
        tst_f = open(f"{odir}/{tst_fname}", "w")

        # Activities
        activities = self._tvdf['activity'].unique().tolist()

        # Activity loop
        for act in activities:
            df_act = self._tvdf[self._tvdf['activity'] == act].copy()

            # Create list of data frames grouped by session and person
            dftrn, dfval, dftst = self._create_df_list(df_act, "splabel")

            # Creating list for training samples
            self._create_txt_list(trn_f, dftrn, num_trn_samples)

            # Creating list for validation samples
            self._create_txt_list(val_f, dfval, num_val_samples)

            # Creating list for testing samples
            self._create_txt_list(tst_f, dftst, num_tst_samples)

        trn_f.close()
        val_f.close()
        tst_f.close()

    def _create_txt_list(self, f, dflst, n):
        """ Writes each trimmed video sample into the file
        while maintaining diversity.

        Parameters
        ----------
        f: File pointer
            File pointer opened in write mode, 
            Ex. `open("temp.txt", "w")`
        dflst: List of DataFrames
            One entry represents activity instances of a person
            during a session.
        n: int
            Number of samples required from list of dataframes
        """
        extracted_samples = 0
        while (extracted_samples < n):
            empty_df_count = 0
            for df_idx, df in enumerate(dflst):

                if len(df) > 0:
                    ridx = random.randint(0, len(df) - 1)
                    rrow = df.iloc[ridx]
                    df = df.drop(df.index[[ridx]])
                    dflst[df_idx] = df

                    # Writing line
                    line = (f"{rrow['activity']}/{rrow['vname']} "
                            f"{rrow['num_label']}\n")
                    f.write(line)
                    extracted_samples += 1

                    # Break if we have extracted enough samples break from for
                    if extracted_samples >= n:
                        break
                else:
                    empty_df_count += 1

            # Break from while if
            # 1. If we extracted enough samples
            # 2. All the dataframes in the list are empty
            if extracted_samples >= n:
                break

            # If all data frames in dflst are empty break from while
            if empty_df_count >= len(dflst):
                print(f"All samples are extracted, {extracted_samples}")
                break

    def _create_df_list(self, df, col_name):
        """ Creates list of dataframes split based on `col_name`

        Parameters
        ----------
        col_name: str
            Column name based on which list of dataframes are created.
        """

        df_trn = []
        df_val = []
        df_tst = []

        splits = ["trn", "val", "tst"]

        for csplit in splits:
            df_split = df[df['split'] == csplit].copy()

            # Printing number of samples available
            # print(f"INFO: Number of {csplit} samples: {len(df_split)}")

            # Unique session + person instances
            uniq_col_values = df_split[col_name].unique().tolist()

            # Loop over splits
            for col_val in uniq_col_values:

                df_tmp = df[df[col_name] == col_val].copy()

                if csplit == "trn":
                    df_trn = df_trn + [df_tmp.reset_index(drop=True)]
                elif csplit == "val":
                    df_val = df_val + [df_tmp.reset_index(drop=True)]
                elif csplit == "tst":
                    df_tst = df_tst + [df_tmp.reset_index(drop=True)]
                else:
                    raise Exception(f"Split not found {csplit}")

        return df_trn, df_val, df_tst
