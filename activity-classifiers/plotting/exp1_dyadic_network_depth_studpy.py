"""
NOTE: WILL NOT WORK DUE OT CHANGES TO LogAnalyzer class

Plots
1. Training accuracy/Validation accuracy for dyadic networks
2. Plots training and validation accuracy gap for dyadic networks

Locations to log files are passed via json file
"""
import pdb
import sys
import json
import argparse
import numpy as np
import chart_studio
import chart_studio.plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from aqua.nn.log_analyzer import LogAnalyzer

save_loc = "/home/vj/Dropbox/Marios_Shared/HAQ-AOLME/documentation/reports/draft/org/experiments/wnw_dyadic_nn_depth_experiments.html"
logdict = {
    "title":"Dyadic network depth study on writing/nowriting",
    "rdir": "/mnt/twotb/dyadic_nn/workdir/wnw/one_trim_per_instance_3sec_224/trn_videos_150per_act",
    "logs":
    [
        {
	    "legend":"One dyad",
	    "loc":"dyad_1/run0",
	    "trnlog":"trn_log.json",
	    "vallog":"val_videos_all_log.json",
            "color":"red"
        },
        {
	    "legend":"Two dyad",
	    "loc":"dyad_2/run0",
	    "trnlog":"trn_log.json",
	    "vallog":"val_videos_all_log.json",
            "color":"blue"
        },
        {
	    "legend":"Three dyad",
	    "loc":"dyad_3/run0",
	    "trnlog":"trn_log.json",
	    "vallog":"val_videos_all_log.json",
            "color":"green"
        },
        {
	    "legend":"Four dyad",
	    "loc":"dyad_4/run0",
	    "trnlog":"trn_log.json",
	    "vallog":"val_videos_all_log.json",
            "color":"black"
        }
        ]
    
}

# Parse the log dictionary
plot_title = logdict['title']
rdir = logdict['rdir']
log_info_lst = logdict['logs']

# Create a plotly figure instance
fig = make_subplots(rows=1, cols=1)

# Loop over each log and plot
for loginfo in log_info_lst:
    crdir = f"{rdir}/{loginfo['loc']}"
    log_analyzer = LogAnalyzer(crdir)

    # Training accuracy
    trn_accuracy = log_analyzer.get_logged_metric_values(
        loginfo['trnlog'], "acc", "train")
    val_accuracy = log_analyzer.get_logged_metric_values(
        loginfo['vallog'], "acc", "val")

    # Adding training accuracy trace
    fig.add_trace(go.Scatter(x=np.arange(len(trn_accuracy)),
                             y=trn_accuracy,
                             mode='lines',
                             name=f'{loginfo["legend"]}(trn)',
                             line=dict(color=f'{loginfo["color"]}')),
                  row=1,
                  col=1)

    # Adding validation accuracy trace
    fig.add_trace(go.Scatter(x=np.arange(len(val_accuracy)),
                             y=val_accuracy,
                             mode='lines',
                             name=f'{loginfo["legend"]}(val)',
                             line=dict(color=f'{loginfo["color"]}', dash='dot')),
                  row=1,
                  col=1)

# Set title and axis titles and write
fig.update_xaxes(title_text="Epochs", row=1, col=1)
fig.update_yaxes(title_text="Accuracy", row=1, col=1)
fig.update_layout(height=1000,
                  title_text=plot_title,
                  font=dict(family="Courier New, monospace", size=23))
fig.write_html(save_loc)


