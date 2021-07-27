"""
Plots
1. Training accuracy/Validation accuracy for dyadic networks
2. Plots training and validation accuracy gap for dyadic networks

Locations to log files are passed via json file

groups = ['C1L1P-A', 'C1L1P-B', 'C1L1P-C', 'C1L1W-A', 'C2L1W-B', 'C3L1W-D']
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
# groups = ['C1L1P-B', 'C1L1P-C', 'C2L1P-B', 'C2L1P-C']
left_group = "C3L1W-D"
save_loc = f"/home/vj/Dropbox/Marios_Shared/HAQ-AOLME/documentation/reports/draft/org/experiments/dyadic/exp2/tynty_{left_group}.html"
logdict = {
    "title":f"typing/notyping with left out group {left_group}",
    "rdir": "/mnt/twotb/dyadic_nn/workdir/tynty/one_trim_per_instance_3sec_224/exp2_group_leave_one_out",
    "logs":
    [
        {
	    "legend":"Two dyad",
	    "loc":f"{left_group}/dyad_2",
	    "trnlog":"trn_log.json",
	    "vallog":"val_log.json",
            "color":"blue"
        },
        {
	    "legend":"Three dyad",
	    "loc":f"{left_group}/dyad_3",
	    "trnlog":"trn_log.json",
	    "vallog":"val_log.json",
            "color":"green"
        },
        {
	    "legend":"Four dyad",
	    "loc":f"{left_group}/dyad_4",
	    "trnlog":"trn_log.json",
	    "vallog":"val_log.json",
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
    trn_log = f"{rdir}/{loginfo['loc']}/{loginfo['trnlog']}"
    trn_log_analyzer = LogAnalyzer(trn_log)
    val_log = f"{rdir}/{loginfo['loc']}/{loginfo['vallog']}"
    val_log_analyzer = LogAnalyzer(val_log)

    # Training accuracy
    trn_accuracy = trn_log_analyzer.get_metric_values("acc")
    val_accuracy = val_log_analyzer.get_metric_values("acc")

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
fig.update_xaxes(title_text="Epochs", row=1, col=1, rangemode="tozero", fixedrange=True)
fig.update_yaxes(title_text="Accuracy", row=1, col=1, rangemode="tozero", fixedrange=True, autorange=False, range=[0,1])
fig.update_layout(height=1000,
                  title_text=plot_title,
                  font=dict(family="Courier New, monospace", size=23))

fig.write_html(save_loc)
