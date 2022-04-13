"""
Plots learning curves using plotly. It produces two plots, `accuracy.html` and `loss.html`
in the same location as input json files.

Usage
-----
python plot_learning_curves.py <trn json> <val json>

Example
-------
python plot_learning_curves.py \
/mnt/twotb/dyadic_nn/workdir/wnw/one_trim_per_instance_3sec_224/trn_videos_150per_act/dyad_4/run0/trn_log.json \
/mnt/twotb/dyadic_nn/workdir/wnw/one_trim_per_instance_3sec_224/trn_videos_150per_act/dyad_4/run0/val_log.json
"""

import os
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

def _arguments():
    """Parse input arguments."""
    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(
        description=("""
        Plots learning curves using plotly. It produces two plots, `accuracy.html` and `loss.html`
        in the same location as input json files.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args_inst.add_argument("trn_log", type=str, help="JSON file having training log.")
    args_inst.add_argument("val_log", type=str, help="JSON file having validation log.")
    args = args_inst.parse_args()

    args_dict = {"trn_log": args.trn_log, "val_log": args.val_log}
    return args_dict


# Execution starts from here
if __name__ == "__main__":

    # Parsing arguments
    args = _arguments()
    trn_log = args['trn_log']
    val_log = args['val_log']

    # Save locations
    dir_loc = os.path.dirname(trn_log)
    acc_save_loc = f"{dir_loc}/accuracy.html"
    loss_save_loc = f"{dir_loc}/loss.html"

    # Creating LogAnalyzer class instances
    trn_log_analyzer = LogAnalyzer(trn_log)
    val_log_analyzer = LogAnalyzer(val_log)

    # Getting Training and Validation accuracies
    trn_accuracy = trn_log_analyzer.get_metric_values("acc")
    val_accuracy = val_log_analyzer.get_metric_values("acc")

    # Getting Training and Validation loss
    trn_loss = trn_log_analyzer.get_metric_values("loss")
    val_loss = val_log_analyzer.get_metric_values("loss")

    # Plotting Training and validation accuracies
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=np.arange(len(trn_accuracy)),
                             y=trn_accuracy,
                             mode='lines',
                             name=f'Training',
                             line=dict(color='black')),
                  row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(val_accuracy)),
                             y=val_accuracy,
                             mode='lines',
                             name=f'Validation',
                             line=dict(color=f'black', dash='dot')),
                  row=1,
                  col=1)
    fig.update_xaxes(title_text="Epochs", row=1, col=1, rangemode="tozero", fixedrange=True)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1, rangemode="tozero", fixedrange=True, autorange=False, range=[0,1])
    fig.update_layout(height=1000,
                  title_text='Training and Validation accuracy per Epoch',
                  font=dict(family="Courier New, monospace", size=23))
    fig.write_html(f'{acc_save_loc}')

    # Plotting Training and validation loss
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=np.arange(len(trn_loss)),
                             y=trn_loss,
                             mode='lines',
                             name=f'Training',
                             line=dict(color='blue')),
                  row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(val_loss)),
                             y=val_loss,
                             mode='lines',
                             name=f'Validation',
                             line=dict(color=f'blue', dash='dot')),
                  row=1,
                  col=1)
    fig.update_xaxes(title_text="Epochs", row=1, col=1, rangemode="tozero", fixedrange=True)
    fig.update_yaxes(title_text="Loss", row=1, col=1, rangemode="tozero", fixedrange=True, autorange=False, range=[0,1])
    fig.update_layout(height=1000,
                  title_text='Training and Validation loss per Epoch',
                  font=dict(family="Courier New, monospace", size=23))
    fig.write_html(f'{loss_save_loc}')
