import os
import sys
import pdb
import aqua
import math
import datetime
import numpy as np
import pandas as pd
import pretty_errors
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



    def get_person_codes(self, ndf, row, person_code, person_code_type):
        """ Returns person information.
        """
        
        if person_code_type == 'numeric_code':
            numeric_code = row[person_code]
            pseudonym = ndf[ndf[person_code_type] == numeric_code]['pseudonym'].item()
            student_code = ndf[ndf[person_code_type] == numeric_code]['student_code'].item()
            
        elif person_code_type == 'student_code':
            student_code = row[person_code]
            pseudonym = ndf[ndf[person_code_type] == student_code]['pseudonym'].item()
            numeric_code = ndf[ndf[person_code_type] == student_code]['numeric_code'].item()
                        
        elif person_code_type == 'pseudonym':
            pseudonym = row[person_code]
            numeric_code = ndf[ndf[person_code_type] == pseudonym]['numeric_code'].item()
            student_code = ndf[ndf[person_code_type] == pseudonym]['student_code'].item()
            
        else:
            print(f"Does not support {person_code_type}")

        
        return numeric_code, pseudonym, student_code

    def create_xlsx(self, names_csv, out_xlsx, activity, person_code, person_code_type):
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
        person_code: str
            Column name having student identification
        person_code_type: str
            Type of student identification used, {numeric_code, student_code, pseudonym}
        """
        # Session properties
        sprops = pd.read_csv(f"{self._rdir}/properties_session.csv")
        sdur_sec = sprops[sprops['name'] == 'total']['dur'].item()

        # Loading dataframe that contains pseudonyms
        ndf = pd.read_csv(f"{names_csv}")

        # Load dataframe that has typing instances
        df = pd.read_csv(f"{self._rdir}/{self._fname}")
        tydf = df[df['activity'] == activity].copy()
        persons = tydf[person_code].unique()

        # Loop over all typing instances and collect information
        hrlist = [] # Human readable
        for ridx, row in tydf.iterrows():

            # Information of tydf
            name = row['name']
            stime = math.ceil(row['f0']/row['FPS'])
            stime_str = str(datetime.timedelta(seconds=stime))
            etime = math.floor(stime + (row['f']/row['FPS']))
            etime_str = str(datetime.timedelta(seconds=etime))
            dur_sec = etime - stime
            dur_min = round(dur_sec/60.0, 2)
            dur_str = str(datetime.timedelta(seconds=dur_sec))
            w0 = math.floor(row['w0'])
            h0 = math.floor(row['h0'])
            w = math.floor(row['w'])
            h = math.floor(row['h'])
            
            # Getting person name information
            numeric_code, pseudonym, student_code = self.get_person_codes(ndf, row, person_code, person_code_type)

            # Creating list
            hrlist += [[name, numeric_code, pseudonym, student_code,
                        stime_str, etime_str, dur_str,
                        w0, h0, w, h]]




            
        
        # Output dataframe
        try:
            odf = pd.DataFrame(hrlist, columns=['Video name', 'Numeric code', 'Pseudonym', 'Student code',
                                            'Start time', 'End time', 'Duration',
                                            'w0', 'h0', 'w', 'h'])
        except:
            import pdb; pdb.set_trace()

        # Export to excel
        
        print(f"INFO: Writing {out_xlsx}")
        writer = pd.ExcelWriter(out_xlsx, engine='xlsxwriter')
        odf.to_excel(writer, sheet_name="Human readable", index=False)

        # Adding pseudonym, numeric_code, student_code to tydf
        tydf['pseudonym'] = [row[2] for row in hrlist]
        tydf['numeric_code'] = [row[1] for row in hrlist]
        tydf['student_code'] = [row[3] for row in hrlist]
        tydf.to_excel(writer, sheet_name="Machine readable", index=False)
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



    def _get_dirs(self, dirloc):
        """Returns unique groups in the root directory. It is assumed
        that all directories under root directory are groups.

        Parameters
        ----------
        dirloc : Str
            Directory location
        """
        list_dir = os.listdir(dirloc)
        unique_groups = []
        for file_or_dir in list_dir:
            if os.path.isdir(f"{dirloc}/{file_or_dir}"):
                unique_groups += [file_or_dir]

        return unique_groups
        
    def _get_sessions(self):
        """Returns two lists of same size having groups and
        dates. A session is groups[i]/dates[i].
        """

        # Unique groups
        groups = self._get_dirs(self._rdir)

        # Group loop
        groups_ = []
        sessions_ = []
        for group in groups:
            group_path = f"{self._rdir}/{group}"
            dates = self._get_dirs(group_path)
            groups_ += [group]*len(dates)
            sessions_ += dates

        return groups_, sessions_

    def _get_num_act_noact_inst(self, groups, dates, activity):
        """Returns two lists containing number of activity and
        noactivity instances.

        Parameters
        ----------
        groups : Lis[Str]
            A list of strings having groups directory names

        dates : List[Str]
            A list of strings having dates.
        activities : List[Str]
            List of strings having activity name. For example
            it can be ["typing", "notyping"]
        """
        # Creating a list having session full paths
        sess_paths = [
            f"{self._rdir}/{groups[x]}/{dates[x]}" for x in range(0, len(groups))
        ]

        # Loop over each session
        num_act_inst = []
        num_noact_inst = []
        for sess_path in sess_paths:
            sess_gt = pd.read_csv(f"{sess_path}/{self._fname}")
            
            sess_num_act_inst = sum(sess_gt["activity"] == activity)
            sess_num_noact_inst = sum(sess_gt['activity'] == f"no{activity}")
            
            num_act_inst += [sess_num_act_inst]
            num_noact_inst += [sess_num_noact_inst]

        return num_act_inst, num_noact_inst


    def _get_dur_act_noact_inst(self, groups, dates, activity):
        """Returns two lists containing duration of activity and
        noactivity instances.

        Parameters
        ----------
        groups : Lis[Str]
            A list of strings having groups directory names

        dates : List[Str]
            A list of strings having dates.
        activities : List[Str]
            List of strings having activity name. For example
            it can be ["typing", "notyping"]
        """
        # Creating a list having session full paths
        sess_paths = [
            f"{self._rdir}/{groups[x]}/{dates[x]}" for x in range(0, len(groups))
        ]

        # Loop over each session
        dur_act_inst = []
        dur_noact_inst = []
        for sess_path in sess_paths:

            sess_gt = pd.read_csv(f"{sess_path}/{self._fname}")
            fps = sess_gt['FPS'].unique().item()
            sess_gt['dur'] = sess_gt['f']/fps
            sess_gt_act = sess_gt[sess_gt['activity'] == activity].copy()
            sess_gt_noact = sess_gt[sess_gt['activity'] == f"no{activity}"].copy()
            
            sess_dur_act_inst = math.floor(sess_gt_act['dur'].sum())
            sess_dur_noact_inst = math.floor(sess_gt_noact['dur'].sum())
            
            dur_act_inst += [sess_dur_act_inst]
            dur_noact_inst += [sess_dur_noact_inst]

        return dur_act_inst, dur_noact_inst

    
    def save_summary_per_session(self, activity, prev_data_split):
        """Summarizes activity instance ground truth per session.

        Parameters
        ----------
        activities : Str
            activity name. For example it can be "typing".
        prev_data_split : Str
            Path to the csv file containing labels of previous data split.
        """

        # Get session detials.
        # A session is identified by group and corresponding date
        groups, dates = self._get_sessions()

        # number of  <activity> and no<activity> instances per session
        num_act_inst, num_noact_inst = self._get_num_act_noact_inst(groups, dates, activity)

        # Duration (in seconds) of <activity> and no<activity> instances per session
        dur_act_inst, dur_noact_inst = self._get_dur_act_noact_inst(groups, dates, activity)

        # List of previous data split labels

        summary_df = pd.DataFrame(
            list(zip(groups, dates, num_act_inst, num_noact_inst, dur_act_inst, dur_noact_inst)),
            columns=[
                "group", "date_full",
                "num_act_inst", "num_noact_inst",
                "dur_act_inst", "dur_noact_inst"
            ])

        # Summary dataframe get group summary
        summary_df_copy = summary_df.copy()
        summary_df_copy.drop(["date_full"], axis=1)
        agg_functions = {
            'num_act_inst': 'sum',
            'num_noact_inst': 'sum',
            'dur_act_inst': 'sum',
            'dur_noact_inst': 'sum'
        }
        summary_df_group = summary_df_copy.groupby(summary_df_copy['group']).aggregate(agg_functions)
        summary_df_group = summary_df_group.sort_values(by=['group'])
        summary_df_group.loc["Total"] = summary_df_group.sum()

        dur_act_inst_readable = []
        dur_noact_inst_readable = []
        for i, row in summary_df_group.iterrows():
            dur_act_inst_readable += [str(datetime.timedelta(seconds=int(row['dur_act_inst'])))]
            dur_noact_inst_readable += [str(datetime.timedelta(seconds=int(row['dur_noact_inst'])))]


        summary_df_group['dur_act_inst_readable'] = dur_act_inst_readable
        summary_df_group['dur_noact_inst_readable'] = dur_noact_inst_readable
        

        print(f"{self._rdir}/gt_summary_per_session.csv")
        print(f"{self._rdir}/gt_summary_dissertation_table.xlsx")
        summary_df = summary_df.sort_values(by=['group'])
        summary_df.to_csv(f"{self._rdir}/gt_summary_per_session.csv", index=False)
        writer = pd.ExcelWriter(f"{self._rdir}/gt_summary_dissertation_table.xlsx", engine = 'xlsxwriter')
        summary_df.to_excel(writer, sheet_name="Session", index=False)
        summary_df_group.to_excel(writer, sheet_name="Group")
        writer.close()
        
        

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
        """ Loads all activity labels present under root directory to a dataframe.

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
