# Acesss all the files with name gTruth
# Inside each of them calculate
import pdb
import numpy as np
import aqua
import pandas as pd
import matplotlib.pyplot as plt

class TrimStat:
    "The following class contains methods to anaylse trim video statistics"

    def __init__(self, rdir, labels_fname, act_name):
        """
        Parameters
        ----------
        rdir: str
            Directory path having activity labels.
        labels_fname: str
            Name of the file that has activity labels.
        """
        self._rdir = rdir
        self._fname = labels_fname
        self._actname = act_name

    def calculate_act_dur(self):
        "This method calculate duration of activity and no-activity"
        files = aqua.get_file_paths_with_kws(self._rdir, [self._fname])
        df_full_act = pd.DataFrame()
        df_full_noact = pd.DataFrame()
        for ifile in files:
            file = ifile
            df = pd.read_csv(file)
            for index, row in df.iterrows():
                if row.activity == "typing":
                    df_full_act = df_full_act.append(row)
                if row.activity == "notyping":
                    df_full_noact = df_full_noact.append(row)
                if row.activity == "writing":
                    df_full_act = df_full_act.append(row)
                if row.activity == "nowriting":
                    df_full_noact = df_full_noact.append(row)
        act_dur = df_full_act.f.sum()/(30*60*60)
        noact_dur = df_full_noact.f.sum()/(30*60*60)
        act_series  = (df_full_act.f/30).to_list()
        noact_series = (df_full_noact.f/30).to_list()
        print(self._actname)
        print("Activity duration(hrs):", act_dur)
        print("No-Activity duration(hrs):", noact_dur)
        print("No. of activity isntances", len(df_full_act))
        print("No. of no-activity isntances", len(df_full_noact))
        return act_series, noact_series

    def plot_histogram(self, act_series,act_name):
        df = pd.DataFrame({'Activity Duration':act_series})
        min_dur = str(round(np.min(act_series),2))
        max_dur = str(round(np.max(act_series),2))
        med_dur = str(round(np.median(act_series),2))
        #pdb.set_trace()
        hist = df.hist(bins=30)
        textstr = "min_dur: " + min_dur +"\n" + "max_dur : " + max_dur +"\n" + "median_dur : " + med_dur
        plt.text(180, 180, textstr, horizontalalignment='center',
            verticalalignment='center')
        plt.title(act_name)
        plt.show()
