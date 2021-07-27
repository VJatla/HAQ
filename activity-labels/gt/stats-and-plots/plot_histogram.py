import os
from aqua.data_tools import TrimStat

if __name__ == "__main__":

    tynty_rdir = "/home/vj/Dropbox/typing-notyping/"
    tynty_labels_fname = "gTruth-tynty_30fps.csv"
    wnw_rdir = "/home/vj/Dropbox/writing-nowriting-GT/"
    wnw_labels_fname = "gTruth-wnw_30fps.csv"

    # Initialize activity labels instance
    trim_stat = TrimStat(tynty_rdir, tynty_labels_fname, "Typing")
    act_series,noact_series = trim_stat.calculate_act_dur()
    trim_stat.plot_histogram(act_series,"Typing")
    trim_stat.plot_histogram(noact_series,"NoTyping")

    trim_stat = TrimStat(wnw_rdir, wnw_labels_fname,"Writing")
    act_series,noact_series = trim_stat.calculate_act_dur()
    trim_stat.plot_histogram(act_series,"Writing")
    trim_stat.plot_histogram(noact_series,"NoWriting")
