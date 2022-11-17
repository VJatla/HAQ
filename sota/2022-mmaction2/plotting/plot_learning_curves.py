"""
Plots learning curves using plotly. It produces two plots, `accuracy.html` and `loss.html`
in the same location as input json files.

Usage
-----
python plot_learning_curves.py <trn json> <val json>

Example
-------
python plot_learning_curves.py /mnt/twelvetb/vj/mmaction2_2022/workdir/tynty_table_roi/resized_224/i3d/run0_Aug24_2022/20220825_155026.log.json
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
from aqua.nn.log_analyzer.log_analyzer_mmaction2 import LogAnalyzerMMA2

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
    args = args_inst.parse_args()

    args_dict = {"trn_log": args.trn_log}
    return args_dict


# Execution starts from here
if __name__ == "__main__":

    # Parsing arguments
    args = _arguments()
    trn_log = args['trn_log']

    # Save locations
    dir_loc = os.path.dirname(trn_log)
    acc_save_loc = f"{dir_loc}/accuracy.html"

    # Creating LogAnalyzer class instances
    trn_log_analyzer = LogAnalyzerMMA2(trn_log)

    # Getting Training and Validation accuracies
    trn_epochs, trn_accuracy = trn_log_analyzer.get_metric_values("trn", "top1_acc")
    val_epochs, val_accuracy = trn_log_analyzer.get_metric_values("val", "top1_acc")

    # Plotting Training and validation accuracies
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=trn_epochs,
                             y=trn_accuracy,
                             mode='lines',
                             name=f'Training',
                             line=dict(color='black')),
                  row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=val_epochs,
                             y=val_accuracy,
                             mode='lines',
                             name=f'Validation',
                             line=dict(color=f'black', dash='dot')),
                  row=1,
                  col=1)
    fig.update_xaxes(title_text="Epochs", row=1, col=1, rangemode="tozero", fixedrange=True)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1, rangemode="tozero", fixedrange=True, autorange=False, range=[0,1])
    fig.update_layout(height=1000,
                  title_text=f'{os.path.split(os.path.split(dir_loc)[0])[1]}',
                  font=dict(family="Courier New, monospace", size=23))
    fig.write_html(f'{acc_save_loc}')
