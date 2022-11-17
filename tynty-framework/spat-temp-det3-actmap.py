"""
DESCRIPTION
-----------
    Creates three activity maps,
        1. gt.html
        2. <alg name>.html
        3. <alg name>_vs_gt.html

USAGE
-----
python spat-temp-det3-actmap.py /home/vj/Dropbox/AOLME_Activity_Maps/Typing/C2L1P-B/20180223/C2L1P-B_Feb23_actmap.json

"""
import os
import json
import argparse
import pdb
import pandas as pd
import numpy as np
import pytkit as pk
import math
import plotly.graph_objects as go


class ActMaps:


    cfg = {}
    """ Configuration dictionary """

    gtdf = pd.DataFrame()
    """ Ground truth data frame """

    algdf = pd.DataFrame()
    """ Ground truth data frame """

    sessdf = pd.DataFrame()
    """ Session properties dataframe """

    namesdf = pd.DataFrame()
    """ Session properties dataframe """

    _collist = ["green", "red", "blue", "black",
               "magenta", "orange", "pink"]
    """ List of colors for each student """

    
    def __init__(self, cfg_path):
        """
        Parameters
        ----------
        cfg_path : Str
            Configuration path
        """

        # Loading the configuration
        with open(cfg_path) as f:
            cfg = json.load(f)
            self.cfg = cfg

        # Loading ground truth dataframe
        self.gtdf = pd.read_excel(cfg['gt_xl'], sheet_name="Machine readable")

        # Loading the algorithm detection dataframe
        self.algdf = pd.read_excel(cfg['alg_xl'])

        # Loading the session dataframe and sorting
        self.sessdf = pd.read_csv(cfg['sess_prop_csv'])
        self.sessdf = self.sessdf.sort_values(by=['name'])

        # Student names dataframe
        self.namesdf = pd.read_csv(cfg['student_names_csv'])


    def create_tygt_map(self):
        """Create ground truth activity map"""

        # Copying grouhd truth dataframe
        gtdf = self.gtdf.copy()

        # Initilize plotly figure
        fig = go.Figure()

        # Person names as used by education department
        gtdf, student_codes = self._get_student_codes(gtdf.copy())

        # video properties
        vnames = self.sessdf['name'].tolist()[0:-1]
        vdurs  = self.sessdf['dur'].tolist()[0:-1]
        vframes = [30*x for x in vdurs]
        x_ = np.arange(0, sum(vdurs), 1).tolist()

        # Loop through each person
        for sidx, student_code in enumerate(student_codes):

            # Initialize y values to np.nan
            y_ = [np.nan]*len(x_)

            # Dataframe for current student
            stu_gtdf = gtdf[gtdf['student_code'] == student_code].copy()

            # Loop through each video
            first_act_per_stu = True
            for vidx, vname in enumerate(vnames):

                # Dataframe instances of current video
                stu_vid_gtdf = stu_gtdf[stu_gtdf['name'] == vname].copy()
                stu_vid_gtdf = stu_vid_gtdf.reset_index()

                # if empty continue to next video
                if stu_vid_gtdf.empty:
                    continue

                # Previous number of frames
                prev_nframes = sum(vframes[0:vidx])

                # loop through each instance of activity
                first_act_per_vid = True
                for aidx, arow in stu_vid_gtdf.iterrows():

                    # starting frame, duration w.r.t. video and session
                    fs_vid = arow['f0']
                    fs_sess = prev_nframes + fs_vid
                    dur_start_vid = math.floor(fs_vid/30)
                    dur_start_sess = math.floor(fs_sess/30)

                    # ending frame, duration w.r.t. video and session
                    fe_vid = fs_vid + arow['f']
                    fe_sess = fs_sess + arow['f']
                    dur_end_vid = math.floor(fe_vid/30)
                    dur_end_sess = math.floor(fe_sess/30)

                    # Changing y values
                    y_[dur_start_sess:dur_end_sess] = [sidx + 1]*(dur_end_sess - dur_start_sess)

                    # Trace
                    if first_act_per_stu:
                        fig.add_trace(go.Scatter(
                            x=x_,
                            y=y_,
                            mode="lines",
                            line=dict(color=self._collist[sidx], width=20),
                            name=f"{student_code}",
                            showlegend=True,
                            legendgroup=f"{student_code}"))
                        first_act_per_stu = False
                    else:
                        fig.add_trace(go.Scatter(
                            x=x_,
                            y=y_,
                            mode="lines",
                            line=dict(color=self._collist[sidx], width=20),
                            name=f"{student_code}",
                            showlegend=False,
                            legendgroup=f"{student_code}"))

                    if first_act_per_vid:
                        # Initilize with new video activity time stamps
                        ann_dur_start = [dur_start_vid, dur_start_sess]
                        ann_dur_end = [dur_end_vid, dur_end_sess]
                        first_act_per_vid = False
                    else:
                        # Annotate previous activity instance if the current activity starts
                        # after 60 seconds of previous activity ends.
                        if dur_start_vid - ann_dur_end[0]  > 60:
                            fig = self._annotate(fig, vname, sidx, ann_dur_start, ann_dur_end)
                            ann_dur_start = [dur_start_vid, dur_start_sess]
                            ann_dur_end = [dur_end_vid, dur_end_sess]
                        else:
                            ann_dur_end = [dur_end_vid, dur_end_sess]

                    # If we hit last activity annotate it and reset lists
                    if aidx == len(stu_vid_gtdf) - 1:
                        fig = self._annotate(fig, vname, sidx, ann_dur_start, ann_dur_end)
                            
        # Updating y axis
        student_codes_vals = np.arange(1, len(student_codes) + 1)
        fig.update_yaxes(
            title_font={"size": 32},
            tickvals=student_codes_vals,
            ticktext=student_codes,
            tickfont=dict(size=20),
        )
        fig.update(layout_yaxis_range = [0,len(student_codes)+1])

        # Updating x axis
        x_vals = np.arange(0, sum(vdurs), 60 * 5).tolist()
        x_labels = [f"{int(x/60)} min" for x in x_vals]
        fig.update_xaxes(
            title_font={"size": 32},
            tickvals=x_vals,
            ticktext=x_labels,
            tickfont=dict(size=20),
        )
        
        # Return figure
        return fig


    def _prepare_algdf(self):
        """ Prepares algorithm dataframe for activity map. Preparation includes,
        1. Remove notyping
        2. Join typing activity that are next to each other

        Returns
        -------
        algdf : Pandas DataFrame
            Algorithm dataframe prepared for plotting
        """

        # Removing notyping instances
        algdf = self.algdf.copy()
        algdf.drop(algdf.index[algdf['activity'] == 'notyping'], inplace=True)

        # Join typing instances that are next to each other.
        algdf_ = self._join_algdf_activities(algdf.copy())

        return algdf_

    
    def _join_algdf_activities(self, df, fth = 1):
        """Join typing activities that are very close to each other.
        Here I define very close with numner of frames threshold, `fth`.
        """

        # New dataframe that should contain the continuous activity instances
        df_ = pd.DataFrame(columns=df.columns)

        # Loop over each student
        pseudonyms = df['pseudonym'].unique().tolist()
        for pseudonym in pseudonyms:

            # Get activities from current student
            dfp = df[df['pseudonym'] == pseudonym].copy()

            # Loop over each video
            videos = df['name'].unique().tolist()

            for video in videos:
                
                # Get activities from current video
                dfv = dfp[dfp['name'] == video].copy()
                dfv = dfv.sort_values(by=['f0'])
                dfv.reset_index(inplace=True)

                # Loop through each activity instance
                nrows_dfv = len(dfv)
                i = 0
                while (i < nrows_dfv):

                    # Previous row
                    irow = dfv.iloc[i]
                    df_.loc[len(df_)] = irow

                    # Loop through remaining rows to check if they are
                    # withing `fth` threshold
                    j = i + 1
                    while(j < nrows_dfv):
                        jrow = dfv.iloc[j]

                        # If the activities are next to each other
                        # combine
                        i_f1 = irow['f1']
                        j_f0 = jrow['f0']
                        
                        if j_f0 - i_f1 <= fth:

                            # Update activity instamce in df_
                            df_.at[len(df_) - 1, 'f1'] = jrow['f1']
                            df_.at[len(df_) - 1, 'f'] += 90   # 3 sec x 30 FPS

                            # Update  previous row
                            irow = df_.iloc[len(df_) - 1]

                            # Move to next activity instance
                            j += 1

                        else:
                            break
                        
                    # Set i value to the row that is not in the activity group
                    i = j
                    
        return df_

    def create_tyalg_map(self):
        """Create algorithm activity map"""

        # Copying grouhd truth dataframe
        algdf = self._prepare_algdf()
        namesdf = self.namesdf.copy()

        # Initilize plotly figure
        fig = go.Figure()

        # Person names as used by education department
        algdf, student_codes = self._get_student_codes(algdf.copy(), namesdf = namesdf)

        # video properties
        vnames = self.sessdf['name'].tolist()[0:-1]
        vdurs  = self.sessdf['dur'].tolist()[0:-1]
        vframes = [30*x for x in vdurs]
        x_ = np.arange(0, sum(vdurs), 1).tolist()

        # Loop through each person
        for sidx, student_code in enumerate(student_codes):

            # Initialize y values to np.nan
            y_ = [np.nan]*len(x_)

            # Dataframe for current student
            stu_algdf = algdf[algdf['student_code'] == student_code].copy()

            # Loop through each video
            first_act_per_stu = True
            for vidx, vname in enumerate(vnames):

                # Dataframe instances of current video
                stu_vid_algdf = stu_algdf[stu_algdf['name'] == vname].copy()
                stu_vid_algdf = stu_vid_algdf.reset_index()

                # if empty continue to next video
                if stu_vid_algdf.empty:
                    continue

                # Previous number of frames
                prev_nframes = sum(vframes[0:vidx])

                # loop through each instance of activity
                first_act_per_vid = True
                for aidx, arow in stu_vid_algdf.iterrows():

                    # starting frame, duration w.r.t. video and session
                    fs_vid = arow['f0']
                    fs_sess = prev_nframes + fs_vid
                    dur_start_vid = math.floor(fs_vid/30)
                    dur_start_sess = math.floor(fs_sess/30)

                    # ending frame, duration w.r.t. video and session
                    fe_vid = fs_vid + arow['f']
                    fe_sess = fs_sess + arow['f']
                    dur_end_vid = math.floor(fe_vid/30)
                    dur_end_sess = math.floor(fe_sess/30)

                    # Changing y values
                    y_[dur_start_sess:dur_end_sess] = [sidx + 1]*(dur_end_sess - dur_start_sess)

                    # Trace
                    if first_act_per_stu:
                        fig.add_trace(go.Scatter(
                            x=x_,
                            y=y_,
                            mode="lines",
                            line=dict(color=self._collist[sidx], width=20),
                            name=f"{student_code}",
                            showlegend=True,
                            legendgroup=f"{student_code}"))
                        first_act_per_stu = False
                    else:
                        fig.add_trace(go.Scatter(
                            x=x_,
                            y=y_,
                            mode="lines",
                            line=dict(color=self._collist[sidx], width=20),
                            name=f"{student_code}",
                            showlegend=False,
                            legendgroup=f"{student_code}"))

                    if first_act_per_vid:
                        # Initilize with new video activity time stamps
                        ann_dur_start = [dur_start_vid, dur_start_sess]
                        ann_dur_end = [dur_end_vid, dur_end_sess]
                        first_act_per_vid = False
                    else:
                        # Annotate previous activity instance if the current activity starts
                        # after 60 seconds of previous activity ends.
                        if dur_start_vid - ann_dur_end[0]  > 60:
                            # fig = self._annotate(fig, vname, sidx, ann_dur_start, ann_dur_end)
                            ann_dur_start = [dur_start_vid, dur_start_sess]
                            ann_dur_end = [dur_end_vid, dur_end_sess]
                        else:
                            ann_dur_end = [dur_end_vid, dur_end_sess]

                    # If we hit last activity annotate it and reset lists
                    if aidx == len(stu_vid_algdf) - 1:
                        fig = self._annotate(fig, vname, sidx, ann_dur_start, ann_dur_end)
                            
        # Updating y axis
        student_codes_vals = np.arange(1, len(student_codes) + 1)
        fig.update_yaxes(
            title_font={"size": 32},
            tickvals=student_codes_vals,
            ticktext=student_codes,
            tickfont=dict(size=20),
        )
        fig.update(layout_yaxis_range = [0,len(student_codes)+1])

        # Updating x axis
        x_vals = np.arange(0, sum(vdurs), 60 * 5).tolist()
        x_labels = [f"{int(x/60)} min" for x in x_vals]
        fig.update_xaxes(
            title_font={"size": 32},
            tickvals=x_vals,
            ticktext=x_labels,
            tickfont=dict(size=20),
        )
        
        # Return figure
        return fig

            


    def _annotate(self, fig, vname, sidx, sdur, edur):
        """ Annotate activity map creating link to original video.

        Parameters
        ----------
        fig : Plotly figure instance
            Plotly figure instance where we are adding annotation
        vname : str
            Video name        link = (
            f"{link}?vloc={self.vloc}&vname={vname_gt}&start_time={start_sec}&end_time={end_sec}"
        )
        sidx : int
            Student index (or) y value
        sdur : List[int]
            Starting duration, [video level, session level]
        edur : List[int]
            Ending duration, [video level, session level]
        """

        # Website link
        vname_link = os.path.splitext(vname)[0] + "_gt.mp4"
        link = "https://aolme.unm.edu/researcher/activity_maps/video.php"
        link = (
            f"{link}?vloc={self.cfg['vloc']}&vname={vname_link}&start_time={sdur[0]}&end_time={edur[0]}"
        )

        # Time string
        time_str = f"{math.floor(sdur[0]/60)} min. {sdur[0]%60} sec. to {math.floor(edur[0]/60)} min. {edur[0]%60} sec."
        
        # On hover show the following information
        hover_tempalte = (
            f"<b>Video:</b> {vname}<br>" f"<b>Time:</b> {time_str}"
        )

        # Adding annotation
        fig.add_annotation(
            x=sdur[1],
            y=sidx + 1,
            hovertext=hover_tempalte,
            width=5,
            text=f"<a href='{link}'>*</a>",
            showarrow=True,
        )
        fig.add_annotation(
            x=edur[1]-1,
            y=sidx + 1,
            hovertext=hover_tempalte,
            width=5,
            text=f"<a href='{link}'>#</a>",
            showarrow=True,
        )
        return fig
        
        
        

    def _get_student_codes(self, df, namesdf = pd.DataFrame()):
        """ Returns person codes that education department
        is using in their research.

        df : DataFrame
            Groundtruth or Algorithm dataframe
        """

        # If student_code is alread in the column names
        if "student_code" in df.columns:
            return df, df['student_code'].unique().tolist()

        # Student pseudonyms
        pseudonyms = df['pseudonym'].unique().tolist()
        df['student_code'] = ""
        student_codes = []
        for pseudonym in pseudonyms:

            # Get student code and add to dataframe
            student_code = namesdf[namesdf['pseudonym'] == pseudonym]['student_code'].item()
            student_codes += [student_code]
            df['student_code'][df['pseudonym'] == pseudonym] = student_code

        return df, student_codes
        
        
        
def _arguments():
    
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=("Plots activity maps"),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Adding arguments
    args_inst.add_argument(
        "cfg_path",
        type=str,
        help=("Configuration path")
    )
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {'cfg_path': args.cfg_path}

    # Return arguments as dictionary
    return args_dict


# Calling as script
if __name__ == "__main__":
    
    # Input argumetns
    args = _arguments()

    # Initalize activity maps instance
    act_maps = ActMaps(args['cfg_path'])

    # Create ground truth activity map
    # gtfig = act_maps.create_tygt_map()
    # gtfig.write_html(f"{act_maps.cfg['odir']}/gt.html")

    # Save the activity map
    algfig = act_maps.create_tyalg_map()
    algfig.write_html(f"{act_maps.cfg['odir']}/alg.html")
    


    # Initialize 
    print(args)
