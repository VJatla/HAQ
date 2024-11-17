
import argparse
import os
import pdb
import pandas as pd
import pytkit as pk
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder


class ActMapsAndPerf():

    act = ""
    """ Activity, {typing, writing} """

    alg_df = pd.DataFrame()
    """ Data frame having activity localization using algorithm """

    gt_df = pd.DataFrame()
    """ Data frame having activity localization from ground turth """

    sess_df = pd.DataFrame()
    """ Data frame having session properties """

    perf_df = pd.DataFrame()
    """ Data frame having current session performance calculated
    every second. """

    vdb_df = pd.DataFrame()
    """ Data frame containing group videos data-base. """


    def __init__(self, idir, odir, vdb, act):
        """ Methods and techniques to evaluate spatio-temporal
        activity detection performance. This class also contain
        methods that help us visualize the performance by using
        activity maps.

        Parameters
        ----------
        idir: Str
            Input directory that has required files. They are,
                1. gt-ty-30fps.xlsx       : Excel file having typing instances from ground truth.
                2. alg-ty-30fps.xlsx      : Excel file having typing instances from algorithm.
                3. properties_session.csv : Session properties
        odir: Str
            Output directory to store performance and visualizations
            (activity maps).
        vdb: Str
            Path to csv file containing group video database. This file is
            exported from AOLME website Cpanel.
        act: Str
            Activity, {typing, writing}
        """
        idir = pk.check_dir_existance(idir)
        odir = pk.check_dir_existance(odir)

        # Check if required files exist
        alg_pth  = pk.check_file_existance(f"{idir}/alg-ty-30fps.xlsx")
        gt_pth   = pk.check_file_existance(f"{idir}/gt-ty-30fps.xlsx")
        prop_pth = pk.check_file_existance(f"{idir}/properties_session.csv")

        # Load as data frames
        self.alg_df  = pd.read_excel(alg_pth, sheet_name="Machine readable")
        self.gt_df   = pd.read_excel(gt_pth, sheet_name="Machine readable")
        self.sess_df = pd.read_csv(prop_pth)
        self.vdb_df  = pd.read_csv(vdb)

        # Activity we are interested in
        self.act = act

        # Check for performance csv, if not available create one.
        if not os.path.isfile(f"{idir}/perf_per_sec.csv"):
            self.perf_df = self._calc_perf_per_sec()
            self.perf_df.to_csv(f"{idir}/perf_per_sec.csv", index=False)
        else:
            self.perf_df = pd.read_csv(f"{idir}/perf_per_sec.csv")


    def _calc_perf_per_sec(self):
        """ Evaluates algorithm activity instances against ground
        truth. It produces a data frame with each row representing
        a second.

        Parameters
        ----------
        """

        # Duration column
        dur_col = self._get_dur_column()

        # video_name column
        video_name_col = self._get_video_name_column()

        # W, H, FPS columns
        W, H, FPS = self._get_W_H_FPS()
        W_col   = [W]*len(dur_col)
        H_col   = [H]*len(dur_col)
        FPS_col = [FPS]*len(dur_col)

        # f0 and f
        f0_col   = self._get_f0_column(FPS)
        f_col    = [FPS]*len(dur_col)

        # f0_sess
        f0_sess_col = np.arange(0, len(dur_col)*FPS, FPS).tolist()

        # Get student_codes, union of GT and Alg.
        student_codes, student_codes_col = self._get_student_codes(dur_col)

        # Ground truth columns
        act_gt_col, bbox_gt_col, fn_gt_col = self._get_act_columns(
            self.gt_df.copy(), video_name_col, f0_col, f_col, student_codes_col
        )

        # Algorithm column
        act_alg_col, bbox_alg_col, fn_alg_col = self._get_act_columns(
            self.alg_df.copy(), video_name_col, f0_col, f_col, student_codes_col
        )

        # Applying median filter for algorithm column
        act_alg_col = self._apply_median_filter_on_labels(act_alg_col, window_size=10)
        
        # Confusion matrix label
        cf_label_col = self._get_cf_mat_labels(act_gt_col, act_alg_col)
        
        # IoU <--- ??? Should do after meeting

        # Creating an empty data frame with required columns
        df = pd.DataFrame(
            list(zip(video_name_col, dur_col, W_col, H_col, FPS_col,
                     f0_col, f_col, f0_sess_col, student_codes_col,
                     bbox_gt_col, act_gt_col, fn_gt_col,
                     bbox_alg_col, act_alg_col, fn_alg_col,
                     cf_label_col)),
            columns=[
                'video_name', 'dur', 'W', 'H', 'FPS',
                'f0', 'f', 'f0_sess', 'student_codes',
                'bbox_gt','act_gt','fn_gt',
                'bbox_alg','act_alg', 'fn_alg',
                'cf_label'
        ])

        return df



    def _get_cf_mat_labels(self, gt, alg):
        """
        Creates confusion matrix labels for each student. It is
        in the form of "TP:TN:FP:FN".
        """

        cf_col = [""]*len(gt)
        for i in range(0,len(gt)):

            gti  = [x for x in gt[i].split(":")]
            algi = [x for x in alg[i].split(":")]
            cfi = []

            for j in range(0,len(gti)):

                gtij = gti[j].strip()
                algij = algi[j].strip()

                if gtij == f"no_{self.act}" and algij == f"no_{self.act}":
                    cfi += ["TN"]

                elif gtij == f"{self.act}" and algij == f"{self.act}":
                    cfi += ["TP"]

                elif gtij == f"no_{self.act}" and algij == f"{self.act}":
                    cfi += ["FP"]

                elif gtij == f"{self.act}" and algij == f"no_{self.act}":
                    cfi += ["FN"]

                else:
                    import pdb; pdb.set_trace()
                    raise Exception(f"Something is wrong")

            cfi = ":".join(cfi)
            cf_col[i] = cfi

        return cf_col




    def _get_student_codes(self, dur_col):
        """ Returns student_codes by considering students from both
        ground truth and algorithm.
        """
        gt_sc = self.gt_df['student_code'].unique()
        alg_sc = self.alg_df['student_code'].unique()
        union_sc = list(set().union(gt_sc, alg_sc))

        for i,x in enumerate(union_sc):
            if i == 0:
                student_code_str = f"{x}"
            else:
                student_code_str += f":{x}"

        return union_sc, [student_code_str]*len(dur_col)


    def _get_act_columns(self, df, video_name_col, f0_col, f_col, student_codes_col):
        """ Checks every second and labels it as activity or
        no-activity. If labeled as activity it returns bounding box
        with hyphen separated values, 'w0-h0-w-h'.
        """
        act_col = []
        bbox_col = []
        fn_col = []

        # Loop every second
        for i, f0 in enumerate(f0_col):

            f = f_col[i]
            video_name = video_name_col[i]
            student_codes = student_codes_col[i]

            act_, bbox, fn_gt= self._get_act_label(
                df.copy(), video_name, f0, f, student_codes
            )

            act_col += [act_]
            bbox_col += [bbox]
            fn_col += [fn_gt]

        return act_col, bbox_col, fn_col



    def _get_act_label(self, df, vname, f0, f, student_codes):
        """ If there are >= 1/2*frames between f0 and f0+f are
        classified as 'activity' we return <self.act>, else
        we return no_<self.act>
        """
        # Get student codes from string
        student_codes = [x for x in student_codes.split(":")]

        # Current video instances
        df = df[df['name'] == vname].copy()

        # Filtering events that happened before or after current
        # frames, (f0,f)
        df['f1'] = df['f0'] + df['f']
        df = df[ f0 <= df['f1'] ]
        df = df[ f0 + f >= df['f0'] ]

        # Loop through each student
        bboxs = ""
        activities = ""
        fns = ""
        for i,student_code in enumerate(student_codes):

            # Data frame having entries for current student
            dfs = df[df['student_code'] == student_code].copy()


            # Calculating number of frames in time interval
            fn = 0
            w0 = 0; h0 = 0; w = 0; h = 0
            for j,row in dfs.iterrows():
                fn += min(f0+f, row['f1']) - max(f0,row['f0'])
                w0 += row['w0']
                h0 += row['h0']
                w += row['w']
                h += row['h']

            # Bounding box
            if not dfs.empty:
                bbox = (
                    f"{w0/len(dfs)}-{h0/len(dfs)}-{w/len(dfs)}-{h/len(dfs)}"
                )
            else:
                bbox = "0-0-0-0"


            # If we have activity for more than 50% of frames
            # in current time interval, we label it as self.act
            if fn >= 0.5*(f):
                act = self.act
            else:
                act = f"no_{self.act}"

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
        """ Returns f0 column """

        # Copy session data frame and drop last row
        sdf = self.sess_df.copy()
        sdf = sdf[:-1]

        # Loop over each row of session dataframe
        f0_col = []
        for ridx, row in sdf.iterrows():
          f0_col += np.arange(0, FPS*row['dur'], FPS).tolist()
        return f0_col


    def _get_W_H_FPS(self):
        """ Returns Width, Height and FPS """
        W   = self.sess_df['width'].unique()[0].astype('int')
        H   = self.sess_df['height'].unique()[0].astype('int')
        FPS = self.sess_df['FPS'].unique()[0].astype('int')

        return W, H, FPS

    def _get_video_name_column(self):
        """ Make a column with video names """

        # Copy session data frame and drop last row
        sdf = self.sess_df.copy()
        sdf = sdf[:-1]

        # Loop over each row of session dataframe
        video_name_col = []
        for ridx, row in sdf.iterrows():
            video_name_col += [row['name']]*int(row['dur'])

        return video_name_col






    def _get_dur_column(self):
        """ Get session duration """

        sess_df   = self.sess_df.copy()
        total_row = sess_df[sess_df['name'] == 'total']
        total_dur = total_row['dur'].item()
        dur_col   = np.arange(0,total_dur).tolist()

        return dur_col

    def _plot_cf_labels(self, fig, sidx, cf_labels, label, labels_traced):
        """ Plots confusion matrix labels on `fig`.
        """
        # Show legend flag
        show_legend = False

        # X values
        x_vals = np.arange(0, len(cf_labels)).tolist()

        # Calculating y value and color
        if label == "TP":
            # y_val = (sidx + 1) - 0.1
            y_val = (sidx + 1)
            color = "green"
            if labels_traced['TP'] == False:
                show_legend=True
                labels_traced['TP'] = True
        elif label == "TN":
            y_val = (sidx + 1)
            color = "rgba(0, 125, 0, 0.5)"
            if labels_traced['TN'] == False:
                show_legend=True
                labels_traced['TN'] = True
        elif label == "FN":
            y_val = (sidx + 1)
            color = "red"
            if labels_traced['FN'] == False:
                show_legend=True
                labels_traced['FN'] = True
        elif label == "FP":
            # y_val = (sidx + 1) + 0.1
            y_val = (sidx + 1)
            color = "orange"
            if labels_traced['FP'] == False:
                show_legend=True
                labels_traced['FP'] = True
        else:
            raise Exception(f"The label, {label}, is not supported")

        # Creating y values list
        y_vals = [y_val*(x == label) for x in cf_labels]
        y_vals = [np.nan if x==0 else x for x in y_vals]

        # Tracing using plotly
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines',
                line=dict(color=color, width=20),
                legendgroup=label,
                name=f"{label}",
                showlegend=show_legend
            ))


        return fig, labels_traced


    def _get_x_labels(self, x_vals):
        """
        Get x labels of the format, {video idx}, {#min}min.

        For example at 3 minutes of <video name>_01_05_30fps.mp4 will
        be labeled, 01, 3min.
        """
        df = self.perf_df.copy()
        FPS = self.sess_df['FPS'].unique()[0].astype('int')

        x_labels = []
        for x in x_vals:
            name = df[df['f0_sess'] == FPS*x]['video_name'].item()
            f0 = df[df['f0_sess'] == FPS*x]['f0'].item()
            vidx = name.split('-')[4].split('_')[-1]
            x_labels += [f"{int(f0/(FPS*60))} min"]

        return x_labels

    def _apply_median_filter_on_labels(self, cur_cf_labels, window_size=10):
        """Applying median filter"""
        list_of_labels = cur_cf_labels

        # Converting labels to integers
        encoder = LabelEncoder()
        int_labels = encoder.fit_transform(list_of_labels)

        # Convert to pandas Series
        int_labels_series = pd.Series(int_labels)

        # Apply rolling window median
        median_labels = int_labels_series.rolling(window=window_size, center=True).median()
        median_labels = median_labels.fillna(method='bfill').fillna(method='ffill')
        median_labels_list = median_labels.tolist()

        # Convert back
        final_labels = encoder.inverse_transform([int(label) for label in median_labels_list]).tolist()

        return final_labels
        

    def _create_cf_mat_map(self, fig):
        """ Creates activity map.
        """
        student_codes = self.perf_df['student_codes'].tolist()
        cf_labels = self.perf_df['cf_label'].tolist()

        # Creating a 2D matrix
        student_codes = [x.split(":") for x in student_codes]
        cf_labels = [x.split(":") for x in cf_labels]

        # Labels dicitonary to keep track of labels that are already
        # traced. This helps in not having multiple legends with
        # same name.
        labels_traced = {"TP":False, "TN": False, "FP": False, "FN": False}

        # Loop over each student
        for sidx in range(0, len(student_codes[0])):

            # Current student confusion matrix labels
            cur_cf_labels = [x[sidx] for x in cf_labels]
            
            # Plot TP, FP, FN
            fig, labels_traced = self._plot_cf_labels(fig, sidx, cur_cf_labels, "TP", labels_traced)
            fig, labels_traced = self._plot_cf_labels(fig, sidx, cur_cf_labels, "FP", labels_traced)
            fig, labels_traced = self._plot_cf_labels(fig,  sidx, cur_cf_labels, "FN", labels_traced)
            # fig, labels_traced = self._plot_cf_labels(fig,  sidx, cur_cf_labels, "TN", labels_traced)

        # Updating y_ticks
        y_vals = np.arange(1, len(student_codes) + 1)
        y_labels = student_codes[0]
        fig.update_yaxes(
        title_font={"size": 32},
        tickvals=y_vals,
        ticktext=y_labels,
        tickfont=dict(size=20))

        # Updating x_ticks: we will be having a label every
        # 3 minutes
        x_vals = np.arange(0, len(cf_labels), 60*20).tolist()
        x_labels = [f"{int(x/60)} min." for x in x_vals]
        fig.update_xaxes(
        title_font={"size": 32},
        tickvals=x_vals,
        ticktext=x_labels,
        tickfont=dict(size=20))


        # Legend font
        fig.update_layout(
            legend = dict(font = dict(size = 20))
        )


        return fig


    def _get_cur_student_act_start_end(self, cur_act_labels):
        """ Returns a list of tuples with starting and ending index of
        activity.

        Example
        -------
        [(50, 60), (300, 800)]
        """
        act_list = [1*(x == self.act) for x in cur_act_labels]

        act_start_end_list = []
        start_idx_flag = True
        for i, v in enumerate(act_list):

            if v == 0 and start_idx_flag:
                continue
            elif v == 1 and not(start_idx_flag):
                continue

            elif v == 1 and start_idx_flag:
                start_idx = i
                start_idx_flag = False

            elif v == 0 and not(start_idx_flag):
                end_idx = i - 1
                act_start_end_list += [(start_idx, end_idx)]
                start_idx_flag = True

            else:
                pdb.set_trace()
                raise Exception(f"Haha I am not sure what is happening")

        return act_start_end_list


    def _get_time_in_min(self, f0):
        """ Returns time as 'MM:SS' format
        """
        FPS = self.sess_df['FPS'].unique()[0].astype('int')
        # Seconds
        sec = int(f0/FPS)

        # Minutes
        min_ = int(sec//60)

        # seconds
        sec_ = int(sec%60)

        return sec, f"{min_:02d}:{sec_:02d}"

    def _get_vidx_from_groups_db(self, name):
        """
        Returns video index from groups database file.

        Parameters
        ----------
        name : str
            Video name
        """
        # removing _30fps from name
        name = name.replace("_30fps","")
        df = self.vdb_df.copy()
        vidx = df[df['video_name'] == name]['idx'].item()
        return vidx


    def _create_act_map(self, fig, method, annotation_delta_time=1, apply_median=False):
        """ Creates ground truth activity map
        """

        # Get student codes and activity labels from performance dataframe
        student_codes = self.perf_df['student_codes'].tolist()
        act_labels = self.perf_df[f'act_{method}'].tolist()


        # Creating a 2D matrix
        student_codes = [x.split(":") for x in student_codes]
        act_labels = [x.split(":") for x in act_labels]

        # Loop over each student
        for sidx in range(0, len(student_codes[0])):

            # Current student confusion matrix labels
            cur_act_labels = [x[sidx].strip() for x in act_labels]

            # Apply median for algorithm
            if apply_median:
                cur_act_labels = self._apply_median_filter_on_labels(cur_act_labels, window_size=10)

            # Current student
            cur_student = student_codes[0][sidx]

            # Plot typing instances or current student
            x_vals = np.arange(0,len(cur_act_labels)).tolist()
            y_val = (sidx + 1)
            y_vals = [y_val*(x == self.act) for x in cur_act_labels]
            y_vals = [np.nan if x==0 else x for x in y_vals]

            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines',
                    line=dict(color="green", width=20),
                    name=f"{cur_student}",
                    showlegend=False
                )
            )

            # Get starting and ending indexes of each event for
            # current student. The time duration is at session level.
            start_end = self._get_cur_student_act_start_end(cur_act_labels)

            # Annotate event. Make sure that each annotation is
            # atleast 1 minute apart
            prev_end_time = 0
            for cur_act_start_end in start_end:

                # Current start and end time at session level
                start, end = cur_act_start_end

                # Set annotation flag based on prev_end_time and
                # current end time
                delta_time = end - prev_end_time
                if delta_time/60 > annotation_delta_time:
                    annotate_falg = True
                    prev_end_time = end
                else:
                    annotate_falg = False

                # Annotate only if annotation flag is true
                if annotate_falg:

                    start_vname = self.perf_df.iloc[start]['video_name']
                    end_vname = self.perf_df.iloc[end]['video_name']
                    if start_vname != end_vname:
                        # At this point I do not know how to address
                        # this. So I am skipping it throwing a warning
                        print(f"Warning: start and end videos are different")

                    # Time w.r.t the video
                    f0_start               = self.perf_df.iloc[start]['f0']
                    f0_end                 = self.perf_df.iloc[end]['f0']
                    start_sec, start_mm_ss = self._get_time_in_min(f0_start)
                    end_sec, end_mm_ss     = self._get_time_in_min(f0_end)
                    time_str               = f"{start_mm_ss} to {end_mm_ss}"

                    # Update link with video index and start and end time
                    link = "https://aolme.unm.edu/researcher/activity_maps/video.php"
                    vidx = self._get_vidx_from_groups_db(start_vname)
                    link = f"{link}?vidx={vidx}&start_time={start_sec}&end_time={end_sec}"

                    # On hover show the following information
                    hover_tempalte = (
                        f"<b>Video:</b> {start_vname}<br>"
                        f"<b>Time:</b> {time_str}"
                    )

                    fig.add_annotation(
                        x=(start+end)/2,
                        y=y_val,
                        hovertext=hover_tempalte,
                        width=5,
                        text=f"<a href='{link}'>*</a>",
                        showarrow=True
                    )


        # Updating y_ticks
        y_vals = np.arange(1, len(student_codes) + 1)
        y_labels = student_codes[0]
        fig.update_yaxes(
        title_font={"size": 32},
        tickvals=y_vals,
        ticktext=y_labels,
        tickfont=dict(size=20))

        # Updating x_ticks: we will be having a label every
        # 3 minutes
        x_vals = np.arange(0, len(act_labels), 60*20).tolist()
        # x_labels = self._get_x_labels(x_vals)
        x_labels = [f"{int(x/60)} min." for x in x_vals]
        fig.update_xaxes(
        title_font={"size": 32},
        tickvals=x_vals,
        ticktext=x_labels,
        tickfont=dict(size=20))


        # Legend font
        fig.update_layout(
            legend = dict(font = dict(size = 20))
        )


        return fig

    def visualize_maps(self):
        """ Helps in visualizing confusion matrix for activity
        detection as activity map.
        """

        # Initialize figure handle
        cf_map_fig = go.Figure()
        gt_fig     = go.Figure()
        alg_fig    = go.Figure()

        # Create maps
        cf_map_fig = self._create_cf_mat_map(cf_map_fig)
        gt_fig     = self._create_act_map(gt_fig, "gt")
        alg_fig    = self._create_act_map(alg_fig, "alg", annotation_delta_time=2, apply_median=True)


        # Show maps
        cf_map_fig.show()
        gt_fig.show()
        alg_fig.show()

    def write_maps(self, out_path):
        """ Writes maps to a directory
        """
        # Initialize figure handle
        cf_map_fig = go.Figure()
        gt_fig     = go.Figure()
        alg_fig    = go.Figure()

        # Create maps
        cf_map_fig = self._create_cf_mat_map(cf_map_fig)
        gt_fig     = self._create_act_map(gt_fig, "gt")
        alg_fig    = self._create_act_map(alg_fig, "alg", annotation_delta_time=2)


        # Write maps

        cf_map_fig.write_html(f"{out_path}/gt_vs_alg.html")
        gt_fig.write_html(f"{out_path}/gt.html")
        alg_fig.write_html(f"{out_path}/alg.html")


    def save_cf_mat_per_person(self, out_path):
        """ Prints confusion matrix per person
        """
        df = self.perf_df.copy()

        # Student codes
        student_codes = [x.split(':') for x in df['student_codes'].tolist()][0]

        # confusion matrix labels
        cf_labels = [x.split(':') for x in df['cf_label'].tolist()]

        # Loop through each student
        rows = []
        for i in range(0,len(student_codes)):

            # Getting current student information
            student_i = student_codes[i]
            cf_labels_i = [x[i] for x in cf_labels]

            # TP, TN, FP, FN
            TP_i = cf_labels_i.count("TP")
            TN_i = cf_labels_i.count("TN")
            FP_i = cf_labels_i.count("FP")
            FN_i = cf_labels_i.count("FN")

            rows += [[student_i, TP_i, TN_i, FP_i, FN_i]]

        # Save the csv file
        df_out = pd.DataFrame(
            rows, columns=['student_code','TP', 'TN', 'FP', 'FN']
        )
        df_out.to_csv(out_path, index=False)
