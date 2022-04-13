"""
Todo
----
1. Add IoU column in to the performacnce data frame

Example to run:
python alg_perf.py ~/Dropbox/typing-notyping/C1L1P-C/20170413 ~/Dropbox/AOLME_Activity_Maps/Typing/C1L1P-C/20170413/ typing
"""


import argparse
"""
Todo
----
1. Add IoU column in to the performacnce data frame
"""
import argparse
import os
import pdb
import pandas as pd
import pytkit as pk
import numpy as np
import plotly.graph_objects as go

class ActDetPerf():

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


    def __init__(self, idir, odir, act):
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

        # Activity we are interested in
        self.act = act

        # Check for performance csv, if not available create one.
        if not os.path.isfile(f"{idir}/perf_per_sec_temp_in.csv"):
            raise Exception(f"{idir}/perf_per_sec_temp_in.csv should be there!")
        else:
            self.perf_df = pd.read_csv(f"{idir}/perf_per_sec_temp_in.csv")
        

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
                    cfi += ["TN"]
                    
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
            y_val = (sidx + 1) - 0.1
            color = "green"
            if labels_traced['TP'] == False:
                show_legend=True
                labels_traced['TP'] = True
        elif label == "FN":
            y_val = (sidx + 1)
            color = "red"
            if labels_traced['FN'] == False:
                show_legend=True
                labels_traced['FN'] = True
        elif label == "FP":
            y_val = (sidx + 1) + 0.1
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
            cur_student = student_codes[0][sidx]

            # Plot TP, FP, FN
            fig, labels_traced = self._plot_cf_labels(fig, sidx, cur_cf_labels, "TP", labels_traced)
            fig, labels_traced = self._plot_cf_labels(fig, sidx, cur_cf_labels, "FP", labels_traced)
            fig, labels_traced = self._plot_cf_labels(fig,  sidx, cur_cf_labels, "FN", labels_traced)

        # Updating y_ticks
        y_vals = np.arange(1, len(student_codes) + 1)
        y_labels = student_codes[0]
        fig.update_yaxes(
        title="Student codes",
        title_font={"size": 32},
        tickvals=y_vals,
        ticktext=y_labels,
        tickfont=dict(size=20))

        # Updating x_ticks
        y_vals = np.arange(1, len(student_codes) + 1)
        y_labels = student_codes[0]
        fig.update_yaxes(
        title="Student codes",
        title_font={"size": 32},
        tickvals=y_vals,
        ticktext=y_labels,
        tickfont=dict(size=20))

        # Legend font
        fig.update_layout(
            legend = dict(font = dict(size = 20))
        )

        return fig
    
    def visualize_cf_mat_map(self):
        """ Helps in visualizing confusion matrix for activity
        detection as activity map.
        """

        # Initialize figure handle
        fig = go.Figure()

        # Create and show map
        fig = self._create_cf_mat_map(fig)
        fig.show()

    def write_cf_mat_map(self, out_path):
        """ Helps in visualizing confusion matrix for activity
        detection as activity map.
        """

        # Initialize figure handle
        fig = go.Figure()

        # Create and show map
        fig = self._create_cf_mat_map(fig)

        # Write
        fig.write_html(out_path)

    

    def bb_intersection_over_union(self, boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou  

    def get_bbox_for_iou(self, df, cf_list, i):

        for each_idx in cf_list:

            boxA = []
            boxA.append(int(float(df.iloc[each_idx]['gt_'+i].split("-")[0])))
            boxA.append(int(float(df.iloc[each_idx]['gt_'+ i].split("-")[1])))
            boxA.append(int(float(df.iloc[each_idx]['gt_' + i].split("-")[2])) + int(float(df.iloc[each_idx]['gt_'+i].split("-")[0])))
            boxA.append(int(float(df.iloc[each_idx]['gt_'+ i].split("-")[3])) + int(float(df.iloc[each_idx]['gt_'+ i].split("-")[1])))

            boxB = []
            boxB.append(int(float(df.iloc[each_idx]['alg_'+ i].split("-")[0])))
            boxB.append(int(float(df.iloc[each_idx]['alg_'+ i].split("-")[1])))
            boxB.append(int(float(df.iloc[each_idx]['alg_'+ i].split("-")[2])) + int(float(df.iloc[each_idx]['gt_'+ i].split("-")[0])))
            boxB.append(int(float(df.iloc[each_idx]['alg_'+ i].split("-")[3])) + int(float(df.iloc[each_idx]['gt_'+ i].split("-")[1])))

            iou = self.bb_intersection_over_union(boxA, boxB)
            df.loc[each_idx, 'iou_'+ i] = iou

            

    def add_IoU_column(self):
        """ 
        """
        df = self.perf_df

        num_cols = len(pd.DataFrame(df.cf_label.str.split(":",expand = True)).columns)

        for i in range(num_cols):
            df['cf_'+ str(i)] =  pd.DataFrame(df.cf_label.str.split(":",expand = True))[i]
            df['gt_'+ str(i)] =  pd.DataFrame(df.bbox_gt.str.split(":",expand = True))[i]
            df['alg_'+ str(i)] = pd.DataFrame(df.bbox_alg.str.split(":",expand = True))[i]
            df['iou_'+ str(i)] = 0
            
            col_name = 'cf_'+ str(i)
            cf_idx_list = df.query(f'{col_name} == "TP"').index.tolist()
            
            #confusion matrix
            print("confusion matrix for person: ", i)
            print(df['cf_'+ str(i)].value_counts())
            
            self.get_bbox_for_iou(df, cf_idx_list, str(i))

            df = df.drop(columns= ['cf_'+ str(i),'gt_'+ str(i),'alg_'+ str(i)])

        df["IOU"] =  df.loc[:,df.columns.str.startswith("iou")].astype(str).apply(lambda x: ': '.join(x), axis = 1)
        df = df.drop(columns = df.loc[:,df.columns.str.startswith("iou")].columns)
        pdb.set_trace()    




        








            
def _arguments():
    """ Parses input arguments """
    
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=("""
        Evaluates performance and viusalize it via activity maps.
        The input directory should have the following files,
            1. gt-ty-30fps.xlsx       : Excel file having typing instances from ground truth.
            2. alg-ty-30fps.xlsx      : Excel file having typing instances from algorithm.
            3. properties_session.csv : Session properties
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args_inst.add_argument("idir", type=str,help="Input directory having necessary files")
    args_inst.add_argument("odir", type=str,help="Output directory")
    args_inst.add_argument("act", type=str,help="Activity name")
    args = args_inst.parse_args()

    args_dict = {
        'idir': args.idir,
        'odir': args.odir,
        'act' : args.act
    }
    return args_dict


# Execution starts from here
if __name__ == "__main__":
    
    # Initializing Activityperf with input and output directories
    args = _arguments()
    ty_perf = ActDetPerf(**args)

    # Visualize confusion matrix as activity map
    #ty_perf.visualize_cf_mat_map()

    # Write the map produced by confusion matrix
    #output_pth = f"{args['odir']}/cf_mat_alg_vs_gt_sravani.html"
    #ty_perf.write_cf_mat_map(output_pth)

    # Creating IoU
    ty_perf.add_IoU_column()


