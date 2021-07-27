import os
import glob
from tqdm import tqdm
import cv2
import math
import statistics as stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import pdb
import sys
import shutil
import tempfile
from aqua.video_tools import Vid


class TypingPerfEval():

    def __init__(self, gt_csv, alg_csv, prop_csv):
        """ Methods and techiniques to evaluate typing
        framework performance.

        Parameters:
        gt_csv: str
            Ground truth csv file.
        alg_csv: str
            CSV file produced by typing framework.
        prop_csv: str
            CSV file having session properties
        """
        # Check if the files exist
        if not os.path.isfile(gt_csv):
            raise Exception(f"{gt_csv} does not exist.")
        if not os.path.isfile(alg_csv):
            raise Exception(f"{alg_csv} does not exist.")
        if not os.path.isfile(prop_csv):
            raise Exception(f"{prop_csv} does not exist.")

        # Loading ground truth for typing (I remove notyping instances).
        self._gtdf = pd.read_csv(gt_csv)
        self._gtdf = self._gtdf[self._gtdf['activity'] == "typing"]

        self._algdf = pd.read_csv(alg_csv)
        self._algdf = self._algdf[self._algdf['activity'] == "typing"]

        self._propdf = pd.read_csv(prop_csv).sort_values('name')
        self._propdf = self._propdf[self._propdf.name != 'total']

    def write_spatio_tempo_perf_tocsv(self, tdur, out_csv):
        """
        Temporal:
            Creates a csv file with each `t` second video segment from automation algorithm classified as
            tp, tn, fp or fn (Confusion matirx). A `t` second instance from ground truth is considered to
            belong to typing if >=50% of frames (>= 15 frames if `t` == 1 sec) belong to typing.The output
            is a CSV file classifying every `t` second video segment in the session as tp, tn, fp, fn.

        Spatial:
            IoU + bounding box coordinates (GT + Automation) are documented. Color code:
            1. tp = Green
            2. tn = Green
            3. fp = Yello (we are ok with fp)
            4. fn = Red (we don't like fn)
        """
        rows = []
        gt_lst = []
        iou_lst = []
        pred_lst =  []
        for vidx, vrow in self._propdf.iterrows():
            vdur = vrow['dur']
            vfps = vrow['FPS']
            cur_vname = vrow['name'] + ".mp4"
            print(f"Processing:    {cur_vname}")

            # >= 50% of frames
            frm_th_for_act = 0.5*(tdur*vfps)

            cur_gtdf = self._gtdf[self._gtdf['name'] == cur_vname].copy()
            cur_algdf = self._algdf[self._algdf['name'] == cur_vname].copy()

            t0_lst = list(range(0, vdur, tdur))
            f0_lst = [int(vfps)*x for x in t0_lst]

            for i, f0 in enumerate(f0_lst):

                t0 = t0_lst[i]
                t0_hr = self._get_hr_time(t0)

                f1 = int(f0 + vfps*tdur)

                gt_ty_flag = self._is_typing((f0, f1), cur_gtdf, frm_th_for_act)
                alg_ty_flag = self._is_typing((f0, f1), cur_algdf, frm_th_for_act)

                # Collecting labels in an array to print metrics
                gt_lst += [int(gt_ty_flag)]
                pred_lst += [int(alg_ty_flag)]

                # Labeling tp, tn, fp, fn

                if gt_ty_flag and alg_ty_flag:
                    # Get bounding box coordinates
                    gt_ty_bbox, gt_ty_bbox_dict = self._get_union_bbox((f0,f1), cur_gtdf)
                    alg_ty_bbox, alg_ty_bbox_dict = self._get_union_bbox((f0,f1), cur_algdf)
                    iou = self._get_iou(gt_ty_bbox_dict, alg_ty_bbox_dict)

                    row = [cur_vname, vdur, vfps, f0, f1-f0, t0_hr, tdur, gt_ty_bbox, alg_ty_bbox]
                    row += ['tp']
                elif not(gt_ty_flag) and not(alg_ty_flag):
                    coord_dict = {'x1':0, 'y1':0, 'x2':0, 'y2':0}
                    gt_ty_bbox, gt_ty_bbox_dict = ([0,0,0,0], coord_dict)
                    alg_ty_bbox, alg_ty_bbox_dict = ([0,0,0,0], coord_dict)

                    row = [cur_vname, vdur, vfps, f0, f1-f0, t0_hr, tdur, gt_ty_bbox, alg_ty_bbox]
                    row += ['tn']
                    iou = 1
                elif not(gt_ty_flag) and alg_ty_flag:
                    alg_ty_bbox, alg_ty_bbox_dict = self._get_union_bbox((f0,f1), cur_algdf)
                    coord_dict = {'x1':0, 'y1':0, 'x2':0, 'y2':0}
                    gt_ty_bbox, gt_ty_bbox_dict = ([0,0,0,0], coord_dict)
                    row = [cur_vname, vdur, vfps, f0, f1-f0, t0_hr, tdur, gt_ty_bbox, alg_ty_bbox]
                    row += ['fp']
                    iou = 0
                elif gt_ty_flag and not(alg_ty_flag):
                    gt_ty_bbox, gt_ty_bbox_dict = self._get_union_bbox((f0,f1), cur_gtdf)
                    alg_ty_bbox, alg_ty_bbox_dict = ([0,0,0,0], coord_dict)
                    row = [cur_vname, vdur, vfps, f0, f1-f0, t0_hr, tdur, gt_ty_bbox, alg_ty_bbox]
                    row += ['fn']
                    iou = 0
                else:
                    raise Exception("Something is fishy")
                iou_lst += [iou]
                row += [iou]
                rows += [row]

        # Write to pandas dataframe
        columns = ['name', 'T', 'FPS', 'f0', 'f', 't0', 't', 'gtbbox', 'algbbox', 'conf_labels','iou']
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(out_csv)

        # Output properties
        tn, fp, fn, tp = confusion_matrix(gt_lst, pred_lst).ravel()
        accuracy = (tn+tp)/(tp+tn+fp+fn)
        precision = (tp)/(tp+fp)
        recall = (tp)/(tp+fn)
        sensitivity = (tp)/(tp+fn)
        specificity = (tn)/(tn+fp)
        print("---- Performance stats ----")
        print(f"tp:{tp}, tn:{tn}, fp:{fp}, fn:{fn}")
        print(f"Accuracy   = {round(accuracy,2)}")
        print(f"Precision  = {round(precision,2)}")
        print(f"Recall     = {round(recall,2)}")
        print(f"Sensitivit = {round(sensitivity,2)}")
        print(f"Specificity = {round(specificity,2)}")
        print(f"IoU = {round(stats.mean(iou_lst), 2)}")
        print("---------------------------")


    def _is_typing(self, frms, df, frm_th):

        """ Returns `True` or `False` of current interval to belong to typing.
        Number of frames that belong to typing are counted between `frms`. If
        that number is greater than or equal to `frm_th` we return `True`.

        Parameters
        ----------
        frms: tuple of int
            Tuple of integers having starting and ending frame numbers. The
            indexing starts from 0.
        df: Pandas DataFrame
            DataFrame containing typing instances
        frm_th: int
            Frames thereshold

        Returns
        -------
        int
            Number of frames that has typing between frames frms tuple, (f0, f1)

        Assumption
        ----------
        - The `df` will only contain typing instances.
        """
        # Current activity interval
        cf0, cf1 = frms

        # Creating f1 column in df
        df['f1'] = df['f0'] + df['f']

        # Filter all rows f1 >= cf0
        # Activities that end after current instance began
        df_fil1 = df[df['f1'] >= cf0].copy()

        # Filter all remaining rows with f0 <= cf1
        # Activities that begun begun before current instance ends
        df_fil2 = df_fil1[df_fil1['f0'] <= cf1].copy()

        # Loop through all the overlapping instances(df_fil2)
        nfrms = 0
        for ridx,row in df_fil2.iterrows():
            f0 = row['f0']
            f1 = row['f1']

            if0 = max(f0, cf0)
            if1 = min(f1, cf1)

            nfrms = if1 - if0 + 1 # +1 as indexing starts from 0(??? correct)

        if nfrms >= frm_th:
            return True
        else:
            return False

    def _get_union_bbox(self,frms, df):
        """
        Returns bounding box coordinates.

        Parameters
        ----------
        frm: int
            Frame number(frame number starts from 0)
        df: Pandas DataFrame
            Dataframe having bounding box coordinates
        """
        w0_lst = []
        h0_lst = []
        w1_lst = []
        h1_lst = []
        for frm in range(frms[0], frms[1]):

            bbox_flag, cdict = self._get_bbox(frm, df)
            if bbox_flag:
                w0_lst += [cdict['x1']]
                h0_lst += [cdict['y1']]
                w1_lst += [cdict['x2']]
                h1_lst += [cdict['y2']]
        if len(w0_lst) != 0:
            w0_min = min(w0_lst)
            h0_min = min(h0_lst)
            w1_max = max(w1_lst)
            h1_max = max(h1_lst)
            bbox = [w0_min, h0_min, w1_max - w0_min, h1_max - h0_min]
            coord_dict = {
                    'x1':w0_min,
                    'y1':h0_min,
                    'x2':w1_max,
                    'y2':h1_max
                }
        else:
            coord_dict = {
                'x1':0,
                'y1':0,
                'x2':0,
                'y2':0
            }
            bbox = [0, 0, 0, 0]


        return bbox, coord_dict


    def _get_bbox(self, frm, df):
        """
        Returns bounding box coordinates.

        Parameters
        ----------
        frm: int
            Frame number(frame number starts from 0)
        df: Pandas DataFrame
            Dataframe having bounding box coordinates

        Returns
        -------
        (Flag, coordinate dictionary). The coordiante dictionary,
        A dictionary with Keys: {'x1', 'x2', 'y1', 'y2'}, The (x1, y1) position
        is at the top left corner, the (x2, y2) position is at the bottom right corner
        """
        df_frm = df[frm >= df['f0']].copy()
        df_frm = df_frm[frm < df_frm['f0'] + df_frm['f']]
        if len(df_frm) == 0:
            coord_dict = {'x1':0, 'y1':0, 'x2':0, 'y2':0}
            return (False, coord_dict)
        elif len(df_frm) > 1:
            raise Exception(f"Multiple typing instances at same time not supported.")
        else:
            row = df_frm.iloc[0]
            coord_dict = {
                'x1':row['w0'],
                'y1':row['h0'],
                'x2':row['w0'] + row['w'],
                'y2':row['h0'] + row['h']
            }
            return (True, coord_dict)




    def _get_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
        # Converting bb
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    def _get_hr_time(self, t):
        """ Get human readable time
        """
        tmin, tsec = divmod(t, 60)
        tmin = str(tmin).zfill(2)
        tsec = str(tsec).zfill(2)
        return f"{tmin}:{tsec}"


class TypingPerfViz():
    def __init__(self, perf_csv):
        """ Contains methods that aid in visualizing typing performance.

        Assumptions
        ----------
        1. Videos are present at the same location as `perf_csv`.

        Parameters
        ----------
        perf_csv: str
            CSV file having performance data.
        """
        if not os.path.isfile(perf_csv):
            raise Exception(f"{perf_csv} not found")

        self._df = pd.read_csv(perf_csv)
        self._df['gtbbox'] = self._df['gtbbox'].map(lambda x: x.replace("[",""))
        self._df['gtbbox'] = self._df['gtbbox'].map(lambda x: x.replace("]",""))
        self._df['algbbox'] = self._df['algbbox'].map(lambda x: x.replace("[",""))
        self._df['algbbox'] = self._df['algbbox'].map(lambda x: x.replace("]",""))
        self._rdir = os.path.dirname(perf_csv)

    def write_to_video(self):
        """ Create video with performance.
        """
        df = self._df.copy()


        # Extract videos to temporary directories
        if True:
            temp_dir = tempfile.mkdtemp()
            print(f"INFO: Extracting videos to frames at {temp_dir}")
            vlist = list(df['name'].unique())
            for vname in tqdm(vlist):
                cur_temp_dir = f"{temp_dir}/{os.path.splitext(vname)[0]}"
                os.mkdir(cur_temp_dir)
                vpath = f"{self._rdir}/{vname}"
                vid = Vid(vpath)
                vid.extract_all_frames_ffmpeg(cur_temp_dir)


        # Loop through each activity instance and put a bounding box
        print("INFO: Writing bboxes")
        for ridx, row in tqdm(df.iterrows(), total=df.shape[0]):
            cur_temp_dir = f"{temp_dir}/{os.path.splitext(row['name'])[0]}"
            num_files = len(glob.glob1(cur_temp_dir,"*.jpg"))
            conf_label = row['conf_labels']

            # Draw and write for each frame
            fs = int(row['f0'])
            fe = fs + int(row['f'])

            fe = min(num_files, fe)
            if conf_label != 'tn' :
                for img_idx in range(fs,fe):
                    # Load image
                    img_path =  f"{cur_temp_dir}/{img_idx}.jpg"
                    img = cv2.imread(img_path)
                    H, W, C = img.shape

                    # Box and text based on confusion matrix label
                    if conf_label == 'tp':
                        c_alg = (0, 255, 0) # Green
                        c_gt = (0, 125, 0)
                        gt_bbox = [float(x) for x in row['gtbbox'].split(",")]
                        gt_tl = (int(gt_bbox[0]), int(gt_bbox[1]))
                        gt_br = (int(gt_bbox[0]) + int(gt_bbox[2]), int(gt_bbox[1]) + int(gt_bbox[3]))
                        img = cv2.rectangle(img, gt_tl, gt_br, c_gt, 2)
                        alg_bbox = [float(x) for x in row['algbbox'].split(",")]
                        alg_tl = (int(alg_bbox[0]), int(alg_bbox[1]))
                        alg_br = (int(alg_bbox[0]) + int(alg_bbox[2]), int(alg_bbox[1]) + int(alg_bbox[3]))
                        img = cv2.rectangle(img, alg_tl, alg_br, c_alg, 2)
                        iou = round(row['iou'], 2)
                        text = f"TP - {iou}"
                        img = cv2.putText(img, text, alg_tl, cv2.FONT_HERSHEY_SIMPLEX, 1, c_alg, 2, cv2.LINE_AA)

                    elif conf_label == 'fp':
                        c = (0, 255, 255)  # yellow

                        alg_bbox = [float(x) for x in row['algbbox'].split(",")]
                        alg_tl = (int(alg_bbox[0]), int(alg_bbox[1]))
                        alg_br = (int(alg_bbox[0]) + int(alg_bbox[2]), int(alg_bbox[1]) + int(alg_bbox[3]))
                        img = cv2.rectangle(img, alg_tl, alg_br, c, 2)

                        text = "FP"
                        img = cv2.putText(img, text, alg_tl, cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)

                    elif conf_label == 'fn':
                        c = (0, 0, 255)  # Red
                        gt_bbox = [float(x) for x in row['gtbbox'].split(",")]
                        gt_tl = (int(gt_bbox[0]), int(gt_bbox[1]))
                        gt_br = (int(gt_bbox[0]) + int(gt_bbox[2]), int(gt_bbox[1]) + int(gt_bbox[3]))
                        img = cv2.rectangle(img, gt_tl, gt_br, c, 2)
                        text = "FN"
                        img = cv2.putText(img, text, gt_tl, cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)

                    else:
                        raise Exception(f"Confusion matrix albel {conf_label} is not supported")

                    cv2.imwrite(img_path, img)
                    # cv2.imshow("test", img)
                    # cv2.waitKey(1)

        # Now writing video
        vlist = list(df['name'].unique())
        try:
            print("INFO: Writing videos with bounding boxes.")
            for vname in tqdm(vlist):
                cur_temp_dir = f"{temp_dir}/{os.path.splitext(vname)[0]}"
                out_path = f"{self._rdir}/perf_{vname}"
                cmd = f"ffmpeg -y -hide_banner -loglevel panic -i '{cur_temp_dir}/%d.jpg' -vf fps=30 -pix_fmt yuv420p {out_path}"
                os.system(cmd)
        except:
            # Cleaning up temporary files if there is exception
            pdb.set_trace()
            shutil.rmtree(temp_dir)


        shutil.rmtree(temp_dir)
