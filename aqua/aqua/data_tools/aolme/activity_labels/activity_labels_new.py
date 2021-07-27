import os
import sys
import pdb
import aqua
import math
import datetime
import numpy as np
import pandas as pd
import skvideo.io as skvio
from aqua.data_tools.aolme.activity_labels.standardize import Standardize
from aqua.data_tools.aolme.activity_labels.process import Process
from aqua.data_tools.aolme.activity_labels.summarize import Summarize
from aqua.data_tools.aolme.activity_labels.visualize import Visualize

class AOLMEActivityLabels:
    def __init__(self, rdir, labels_fname):
        """ Methods to (1). Standardize, (2). Analyze
        and (3). Process activity labels.

        Parameters
        ----------
        rdir: str
            Directory path having activity labels.
        labels_fname: str
            Name of the file that has activity labels.
        """
        self._rdir = rdir
        self._fname = labels_fname

    def create_labeled_videos(self, names_csv, activity):
        """
        The following method overlays labeles on video.

        Assumptions
        -----------
        1. The videos are present in the same directory as csv file
           containing the labels.

        Parameters
        ----------
        names_csv: str
            Path to csv file having numeric names and pseudonyms
        activity: str
            Activity that is being processed
        """
        # Loading data from csv files to dataframes
        df = pd.read_csv(f"{self._rdir}/{self._fname}")
        tydf = df[df['activity'] == activity].copy()
        
        # Loading dataframe that contains pseudonyms
        ndf = pd.read_csv(f"{names_csv}")
        
        # Initializing visualization instance
        viz = Visualize(self._rdir, tydf)
        viz.to_video_ffmpeg(names_df = ndf)
        

    def create_xlsx(self, names_csv, out_xlsx, activity):
        """
        The following code parses ground truth csv file to xlsx sheets.
        This is done to provide easy access to education researchers.

        NOTE
        ----
        This method assumes the existance of `properties_session.csv` file.
        This file contains information related every video in the session.
        

        Parameters
        ----------
        names_csv: str
            Path to csv file having numeric names and pseudonyms
        out_xlsx: str
            Path to output xlsx file
        activity: str
            Activity that is being processed
        """
        # Session properties
        sprops = pd.read_csv(f"{self._rdir}/properties_session.csv")
        sdur_sec = sprops[sprops['name'] == 'total']['dur'].item()

        # Loading dataframe that contains pseudonyms
        ndf = pd.read_csv(f"{names_csv}")

        # Load dataframe that has typing instances
        df = pd.read_csv(f"{self._rdir}/{self._fname}")
        tydf = df[df['activity'] == activity].copy()
        persons = tydf['person'].unique()

        # Loop over all typing instances and collect information
        hrlist = [] # Human readable
        summary_dict = dict.fromkeys(persons, 0)
        for ridx, row in tydf.iterrows():

            # Information of tydf
            name = row['name']
            num_code = row['person']
            stime = math.ceil(row['f0']/row['FPS'])
            stime_str = str(datetime.timedelta(seconds=stime))
            etime = math.floor(stime + (row['f']/row['FPS']))
            etime_str = str(datetime.timedelta(seconds=etime))
            dur_sec = etime - stime
            dur_min = round(dur_sec/60.0, 2)
            dur_str = str(datetime.timedelta(seconds=dur_sec))
            w0 = row['w0']
            h0 = row['h0']
            w = row['w']
            h = row['h']
            

            # Pseudonym
            try:
                pseudonym = ndf[ndf['numeric_code'] == num_code]['pseudonym'].item()
            except:
                pdb.set_trace()

            # Creating list
            hrlist += [[name, num_code, pseudonym,
                        stime_str, etime_str, dur_str,
                        w0, h0, w, h]]

            # Adding information to summary
            summary_dict[num_code] += dur_min

        # Summary dataframe
        summary_df = pd.DataFrame(columns=['Numeric code', 'Pseudonym', 'minutes'])
        for num_code in summary_dict:
            summary_df.loc[len(summary_df.index)] = [
                num_code,
                ndf[ndf['numeric_code'] == num_code]['pseudonym'].item(),
                summary_dict[num_code]
            ]
        summary_df.loc[len(summary_df.index)] = ['Total', 'typing',
                                                 sum(summary_dict.values())] 
        summary_df.loc[len(summary_df.index)] = ['Total', 'notyping',
                                                 int(sdur_sec/60) - sum(summary_dict.values())] 
        
        # Output dataframe
        odf = pd.DataFrame(hrlist, columns=['Video name', 'Numeric code', 'Pseudonym',
                                            'Start time', 'End time', 'Duration',
                                            'w0', 'h0', 'w', 'h'])

        # Export to excel
        print(f"INFO: Writing {out_xlsx}")
        writer = pd.ExcelWriter(out_xlsx, engine='xlsxwriter')
        odf.to_excel(writer, sheet_name="Human readable", index=False)
        tydf.to_excel(writer, sheet_name="Machine readable", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        writer.save()
        

    def save_summary_to_json(self):
        """ Method that summarizes activity labels to a text
        file. The file is saved as `summary.json` in the root
        directory.
        """
        flist = aqua.get_file_paths_with_kws(self._rdir, [self._fname])

        # Create a data frame from all the activity labels
        df = self._load_all_activity_labels(flist)

        # Add time column to data frame
        df = self._add_time(df)

        # Creating sumarize instance
        summary = Summarize(df)

        # Sumamry file path
        opth = f"{self._rdir}/summary.json"

        # Summarize to text file
        summary.to_json(opth)

    def hist_of_activity_labels(self, title):
        """ The following method saves histograms of
        width, height and duration of bounding boxes.
        """
        flist = aqua.get_file_paths_with_kws(self._rdir, [self._fname])

        # Create a data frame from all the activity labels
        df = self._load_all_activity_labels(flist)

        # Add time duration in seconds to data frame
        df = self._add_time(df)

        # Write code from here
        pdb.set_trace()

    def standardize_activity_labels(self, fr=30, overwrite=False):
        """ Standardizes activity labels.

        Parameters
        ----------
        fr: int, optional
            Required frame rate.
        overwrite: bool, optional
            Overwrites. Defaults to False
        """
        # Standardizing activity labels
        std = Standardize(self._rdir, self._fname, fr, overwrite)
        std.create_activity_labels_at_fr()

    def standardize_videos(self, vdb_path, fr=30, overwrite=False):
        """ Standardizes video frame rate in videos.

        Parameters
        ----------
        vdb_path: str
            Video data base path having download link to videos.
        fr: int, optional
            Required frame rate.
        overwrite: bool, optional
            Overwrites. Defaults to False
        """
        # Standardizing videos
        std = Standardize(self._rdir, self._fname, fr, overwrite)
        std.transcode_videos_at_fr(vdb_path)

    def create_spatiotemporal_trims(self,
                                    odir,
                                    trims_per_instance=1,
                                    dur=3,
                                    overwrite=False):
        """ Creates one spatiotemporal trim per one instance of activity
        label.

        Parameters
        ----------
        odir: str
            Path of direcotry to store trims.
        trims_per_instance: int,
            Number of trims to be extracted from one instance of
            activity label. -1 implies that we trim completely.
        dur: int, optional
            Duration of each trim in seconds. Defaults to 3.
        overwrite: bool, optional
            Overwrites. Defaults to False
        """
        flist = aqua.get_file_paths_with_kws(self._rdir, [self._fname])

        # Create a data frame from all the activity labels
        df = self._load_all_activity_labels(flist)

        # Add time column to data frame
        df = self._add_time(df)

        proc = Process(self._rdir, self._fname)
        if trims_per_instance == 1:
            proc.one_trim_per_instance(odir, dur, overwrite)
        elif trims_per_instance == -1:
            proc.trim_instances(odir, overwrite)
        else:
            raise Exception("Invalid trims per instance, "
                            f"{trims_per_instance}")

    def _load_all_activity_labels(self, flist):
        """ Loads all activity labels present under rood directory inot
        one dataframe.

        Parameters
        ----------
        flist: list of str
            List of csv file paths having activity labels.
        """
        dflst = []
        for f in flist:
            dflst += [pd.read_csv(f)]

        return pd.concat(dflst, ignore_index=True)

    def _add_time(self, df):
        """ Adds time column to data frame
        
        Parameters
        ----------
        df: DataFrame
            DataFrame having activity label instances
        """
        fps = df['FPS'].to_numpy()
        f = df['f'].to_numpy()
        t = np.round(f / fps, 2)

        df['t'] = t

        return df
