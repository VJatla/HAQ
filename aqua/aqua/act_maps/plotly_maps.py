import pdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import json
import os

class ActMapsPlotly:

    tydf_ps = pd.DataFrame()
    """ Data frame having typing activities per second  """

    wrdf_ps = pd.DataFrame()
    """ Data frame having writing activities per second  """

    sess_df = pd.DataFrame()
    """ Data frame having session properties """

    vloc = ""
    """Video location w.r.t the root of AOLME Data server"""

    odir = ""
    """ Output path, it should be a .html file """

    student_y_val = {}
    """ A dictionary having having student codes. The key is sutdent
    code and the value is the integer used to represent the sutdent
    in activity map."""

    def __init__(self, cfg):
        """

        Parameters
        ----------
        cfg: Str
            Configuration file having the following information in
            `json` format.

            1. typing instances as excel file
            2. writing instances as excel file
            3. session information as csv file
        """
        with open(cfg) as f:
            self.cfg = json.load(f)

        # Loading activity instances and session properties as dataframe
        self.odir = self.cfg["odir"]
        self.vloc = self.cfg["vloc"]
        self.sess_df = pd.read_csv(self.cfg["sprops"])

        # Load the ativity dataframes with 1 second resolution if they aready
        # exist, else create them
        if not self.cfg["ty"] == "":
            tydf = pd.read_excel(self.cfg["ty"], sheet_name="Machine readable")
            tydf_ps_path = f"{self.odir}/tydf_per_sec.csv"
            if os.path.isfile(tydf_ps_path):
                self.tydf_ps = pd.read_csv(tydf_ps_path)
            else:
                self.tydf_ps = self._get_activities_per_sec(tydf, "typing")

        if not self.cfg["wr"] == "":
            wrdf = pd.read_excel(self.cfg["wr"], sheet_name="Machine readable")
            wrdf_ps_path = f"{self.odir}/wrdf_per_sec.csv"
            if os.path.isfile(wrdf_ps_path):
                self.wrdf_ps = pd.read_csv(wrdf_ps_path)
            else:
                self.wrdf_ps = self._get_activities_per_sec(wrdf, "writing")

        self.student_y_val = self._get_student_value(self.tydf_ps.copy(), self.wrdf_ps.copy())
            





            

    def _get_activities_per_sec(self, df, act):
        """Creates a data frame with each row representing one second.
        Extra columns are added to support multiple instances instances
        that happen in the same time.
        """

        # Duration column
        dur_col = self._get_dur_column()

        # video_name column
        video_name_col = self._get_video_name_column()

        # W, H, FPS columns
        W, H, FPS = self._get_W_H_FPS()
        W_col = [W] * len(dur_col)
        H_col = [H] * len(dur_col)
        FPS_col = [FPS] * len(dur_col)

        # f0 and f
        f0_col = self._get_f0_column(FPS)
        f_col = [FPS] * len(dur_col)

        # f0_sess
        f0_sess_col = np.arange(0, len(dur_col) * FPS, FPS).tolist()

        # Get student_codes
        student_codes, student_codes_col = self._get_student_codes(df, dur_col)

        # Ground truth columns
        act_col, bbox_col, fn_col = self._get_act_columns(
            df.copy(), video_name_col, f0_col, f_col, student_codes_col, act
        )

        # Creating an empty data frame with required columns
        df_per_sec = pd.DataFrame(
            list(
                zip(
                    video_name_col,
                    dur_col,
                    W_col,
                    H_col,
                    FPS_col,
                    f0_col,
                    f_col,
                    f0_sess_col,
                    student_codes_col,
                    bbox_col,
                    act_col,
                    fn_col,
                )
            ),
            columns=[
                "video_name",
                "dur",
                "W",
                "H",
                "FPS",
                "f0",
                "f",
                "f0_sess",
                "student_codes",
                "bbox",
                "act",
                "fn",
            ],
        )
        return df_per_sec

    def _get_student_codes(self, df, dur_col):
        """Returns student_codes by considering students from both
        ground truth and algorithm.
        """

        sc = df["student_code"].unique()
        student_code_str = ":".join(sc)

        return sc, [student_code_str] * len(dur_col)

    def _get_act_columns(
        self, df, video_name_col, f0_col, f_col, student_codes_col, act
    ):
        """Checks every second and labels it as activity or
        no-activity. If labeled as activity it returns bounding box
        with hyphen separated values, 'w0-h0-w-h'.
        """
        act_col = []
        bbox_col = []
        fn_col = []

        # Loop every second
        for i, f0 in tqdm(enumerate(f0_col)):

            f = f_col[i]
            video_name = video_name_col[i]
            student_codes = student_codes_col[i]

            act_, bbox, fn_gt = self._get_act_label(
                df.copy(), video_name, f0, f, student_codes, act
            )

            act_col += [act_]
            bbox_col += [bbox]
            fn_col += [fn_gt]

        return act_col, bbox_col, fn_col

    def _get_act_label(self, df, vname, f0, f, student_codes, cur_act):
        """If there are >= 1/2*frames between f0 and f0+f are
        classified as 'activity' we return <self.act>, else
        we return no_<self.act>
        """
        # Get student codes from string
        student_codes = [x for x in student_codes.split(":")]

        # Current video instances
        df = df[df["name"] == vname].copy()

        # Filtering events that happened before or after current
        # frames, (f0,f)
        df["f1"] = df["f0"] + df["f"]
        df = df[f0 <= df["f1"]]
        df = df[f0 + f >= df["f0"]]

        # Loop through each student
        bboxs = ""
        activities = ""
        fns = ""
        for i, student_code in enumerate(student_codes):

            # Data frame having entries for current student
            dfs = df[df["student_code"] == student_code].copy()

            # Calculating number of frames in time interval
            fn = 0
            w0 = 0
            h0 = 0
            w = 0
            h = 0
            for j, row in dfs.iterrows():
                fn += min(f0 + f, row["f1"]) - max(f0, row["f0"])
                w0 += row["w0"]
                h0 += row["h0"]
                w += row["w"]
                h += row["h"]

            # Bounding box
            if not dfs.empty:
                bbox = f"{w0/len(dfs)}-{h0/len(dfs)}-{w/len(dfs)}-{h/len(dfs)}"
            else:
                bbox = "0-0-0-0"

            # If we have activity for more than 50% of frames
            # in current time interval, we label it as self.act
            if fn >= 0.5 * (f):
                act = cur_act
            else:
                act = f"no_{cur_act}"

            # Creating a string for bounding boxes, student_code and
            # activities
            if i == 0:
                bboxs = f"{bbox}"
                activities = f"{act}"
                fns = f"{fn}"
            else:
                bboxs += f" :{bbox}"
                activities += f" :{act}"
                fns += f" :{fn}"

        return activities, bboxs, fns

    def _get_f0_column(self, FPS):
        """Returns f0 column"""

        # Copy session data frame and drop last row
        sdf = self.sess_df.copy()
        sdf = sdf[:-1]

        # Loop over each row of session dataframe
        f0_col = []
        for ridx, row in sdf.iterrows():
            f0_col += np.arange(0, FPS * row["dur"], FPS).tolist()
        return f0_col

    def _get_W_H_FPS(self):
        """Returns Width, Height and FPS"""
        W = self.sess_df["width"].unique()[0].astype("int")
        H = self.sess_df["height"].unique()[0].astype("int")
        FPS = self.sess_df["FPS"].unique()[0].astype("int")

        return W, H, FPS

    def _get_video_name_column(self):
        """Make a column with video names"""

        # Copy session data frame and drop last row
        sdf = self.sess_df.copy()
        sdf = sdf[:-1]

        # Loop over each row of session dataframe
        video_name_col = []
        for ridx, row in sdf.iterrows():
            video_name_col += [row["name"]] * int(row["dur"])

        return video_name_col

    def _get_dur_column(self):
        """Get session duration"""

        sess_df = self.sess_df.copy()
        total_row = sess_df[sess_df["name"] == "total"]
        total_dur = total_row["dur"].item()
        dur_col = np.arange(0, total_dur).tolist()

        return dur_col

    def _group_cur_student_act(self, start_end, th):
        """ Group activities into one if the start of next activity is withing `th`
        seconds of end of previous activity.

        Parameters
        ----------
        start_end : List of tuples
            Start and end of activities
        th : int
            Threshold to group activities
        """
        # If input list is empty return empty list
        if len(start_end) == 0:
            return []
        
        g_start_end = []
        for i in range(0, len(start_end)):
            if i == 0:
                g_start, g_end = start_end[i]
            else:
                start, end = start_end[i]
                if start - g_end <= th:
                    g_end = end
                else:
                    g_start_end += [(g_start, g_end)]
                    # Starting new group
                    g_start = start
                    g_end = end
        
        # Case where group list is empty
        if len(g_start_end) == 0:
            g_start_end += [(g_start, g_end)]
            
        # In case we missed the last entry            
        if g_start_end[-1][1] != start_end[-1][1]:
            g_start_end += [(g_start, g_end)]

        return g_start_end

    def _get_cur_student_act_start_end(self, cur_act_labels, act):
        """Return a list of tuples with starting and ending index of activity.

        Example
        -------
        [(50, 60), (300, 800)]
        """
        act_list = [1 * (x == act) for x in cur_act_labels]

        act_start_end_list = []
        start_idx_flag = True
        for i, v in enumerate(act_list):

            if v == 0 and start_idx_flag:
                continue
            elif v == 1 and not (start_idx_flag):
                continue

            elif v == 1 and start_idx_flag:
                start_idx = i
                start_idx_flag = False

            elif v == 0 and not (start_idx_flag):
                end_idx = i - 1
                act_start_end_list += [(start_idx, end_idx)]
                start_idx_flag = True

            else:
                pdb.set_trace()
                raise Exception(f"Haha I am not sure what is happening")

        return act_start_end_list

    def _get_time_in_min(self, f0):
        """Returns time as 'MM:SS' format"""
        FPS = self.sess_df["FPS"].unique()[0].astype("int")
        # Seconds
        sec = int(f0 / FPS)

        # Minutes
        min_ = int(sec // 60)

        # seconds
        sec_ = int(sec % 60)

        return sec, f"{min_:02d}:{sec_:02d}"

    def _annotate(self, df, fig, y_val, start, end, vname):

        # Time w.r.t the video
        f0_start = df.iloc[start]["f0"]
        f0_end = df.iloc[end - 1]["f0"] + df.iloc[end -1]["f"]
        start_sec, start_mm_ss = self._get_time_in_min(f0_start)
        end_sec, end_mm_ss = self._get_time_in_min(f0_end)
        time_str = f"{start_mm_ss} to {end_mm_ss}"

        # Update link with video index and start and end time
        link = "https://aolme.unm.edu/researcher/activity_maps/video.php"
        vname_gt = os.path.splitext(vname)[0] + "_gt.mp4"
        link = (
            f"{link}?vloc={self.vloc}&vname={vname_gt}&start_time={start_sec}&end_time={end_sec}"
        )

        # On hover show the following information
        hover_tempalte = (
            f"<b>Video:</b> {vname}<br>" f"<b>Time:</b> {time_str}"
        )

        x_val = end

        fig.add_annotation(
            x=x_val,
            y=y_val,
            hovertext=hover_tempalte,
            width=5,
            text=f"<a href='{link}'>*</a>",
            showarrow=True,
        )
        return fig

    def _get_student_value(self, tydf, wrdf):
        """ 
        """
        df = pd.concat([tydf, wrdf])
        ty_students = set()
        wr_students = set()
        
        if not tydf.empty:
            ty_students = set(tydf.iloc[0]['student_codes'].split(":"))
        if not wrdf.empty:
            wr_students = set(wrdf.iloc[0]['student_codes'].split(":"))
            
        all_students = sorted(ty_students.union(wr_students))
        student_y_val = {}
        for i in range(0, len(all_students)):
            student_y_val[all_students[i]] = i + 1
            
        return student_y_val
            
            

    def _create_act_map(
            self, fig, df, act, annotation_delta_time=60, color="red"
    ):
        """Creates ground truth activity map

        Parameters
        ----------
        fig: Plotly Figure object
             Plotly figure object on which we need to trace
        
        annotation_delta_time : Int
            Two activities that end withing `annotation_delata_time` are not
            required to be marked.
        """

        # Get student codes and activity labels from per second dataframe
        student_codes = df["student_codes"].tolist()
        act_labels = df[f"act"].tolist()
        import pdb; pdb.set_trace()

        # Creating a 2D matrix
        student_codes = [x.split(":") for x in student_codes]
        act_labels = [x.split(":") for x in act_labels]

        # Loop over each student
        for sidx in range(0, len(student_codes[0])):

            # Current student confusion matrix labels
            cur_act_labels = [x[sidx].strip() for x in act_labels]
            cur_student = student_codes[0][sidx]

            # Plot typing instances or current student
            x_vals = np.arange(0, len(cur_act_labels)).tolist()
            y_val = self.student_y_val[cur_student]
            y_vals = [y_val * (x == act) for x in cur_act_labels]
            y_vals = [np.nan if x == 0 else x for x in y_vals]

            if sidx == 0:
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines",
                        line=dict(color=color, width=20),
                        name=f"{act}",
                        showlegend=True,
                        legendgroup=f"{act}"
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines",
                        line=dict(color=color, width=20),
                        name=f"{act}",
                        showlegend=False,
                        legendgroup=f"{act}"
                    )
                )

            # Get starting and ending indexes of each event for
            # current student. The time duration is at session level.
            start_end = self._get_cur_student_act_start_end(cur_act_labels, act)

            # Annotate event. Make sure that each annotation
            start_end_grouped = self._group_cur_student_act(start_end, annotation_delta_time)


            prev_end_time = 0
            for cur_act_start_end in start_end_grouped:

                # Current start and end time at session level
                start, end = cur_act_start_end

                start_vname = df.iloc[start]["video_name"]
                end_vname = df.iloc[end]["video_name"]

                if(start_vname != end_vname):
                    new_end = self.sess_df[self.sess_df['name'] == start_vname]['dur'].item()
                    fig = self._annotate(df, fig, y_val, start, new_end, start_vname)

                    new_start = new_end # 30 minutes (all AOLME videos are under 25min)
                    fig = self._annotate(df, fig, y_val, new_start, end, end_vname)

                else:
                    fig = self._annotate(df, fig, y_val, start, end, start_vname)

        # Updating y_ticks
        y_vals = np.arange(1, len(self.student_y_val) + 1)
        y_labels = [*self.student_y_val]
        fig.update_yaxes(
            title_font={"size": 32},
            tickvals=y_vals,
            ticktext=y_labels,
            tickfont=dict(size=20),
        )
        fig.update(layout_yaxis_range = [0,len(self.student_y_val)+1])

        # Updating x_ticks: we will be having a label every
        # 3 minutes
        x_vals = np.arange(0, len(act_labels), 60 * 5).tolist()
        x_labels = [f"{int(x/60)} min" for x in x_vals]
        fig.update_xaxes(
            title_font={"size": 32},
            tickvals=x_vals,
            ticktext=x_labels,
            tickfont=dict(size=20),
        )

        # Legend font
        fig.update_layout(legend=dict(font=dict(size=20)))

        return fig

    def write_activity_map(self, act_map_type, annotation_delta_time=30, visualize=True):
        """Writes maps to a directory

        Parameters
        ----------
        act_map_type: Str
            Activity map type. It is added as prefix before saving the html files.
        annotation_delta_time: Int
            Two activities that end withing `annotation_delata_time` are not
            required to be marked.
        visualize: Bool
            Visualize activity map
        """
        # Create activity map
        ty_fig = go.Figure()
        wr_fig = go.Figure()
        fig = go.Figure()

        if not self.tydf_ps.empty:
            ty_fig = self._create_act_map(
                ty_fig, self.tydf_ps, "typing", annotation_delta_time=60, color="blue"
            )
            if visualize:
                ty_fig.show()
            ty_fig.write_html(f"{self.odir}/{act_map_type}_ty.html")

        if not self.wrdf_ps.empty:
            wr_fig = self._create_act_map(
                wr_fig, self.wrdf_ps, "writing", annotation_delta_time=60, color="green"
            )
            if visualize:
                wr_fig.show()
            wr_fig.write_html(f"{self.odir}/{act_map_type}_wr.html")

        if not self.tydf_ps.empty and not self.wrdf_ps.empty:
            fig = ty_fig
            fig = self._create_act_map(
                fig, self.wrdf_ps, "writing", annotation_delta_time=60, color="green"
            )
            if visualize:
                fig.show()
            fig.write_html(f"{self.odir}/{act_map_type}.html")
