"""
The following script plots,
1. Training loss per epoch
0:00:00 2. Validation accuracy per epoch

It is written to be used in RTX3 system (The one with 3 GPUs).
"""
import os
import pdb
import aqua
import argparse
import numpy as np
import chart_studio
import chart_studio.plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from aqua.learning_analysis.tsn import TSNAnalyzer


def _arguments():
    """ Parses input arguments """

    # Initialize arguments instance
    args_inst = argparse.ArgumentParser(description=(
        "The following script plots training loss and validation "
        "accuracy from json file dumped during TSN training process."))

    # Adding arguments
    args_inst.add_argument("rdir",
                           type=str,
                           help=("Root directory having training log as "
                                 " `.json` file"))
    args_inst.add_argument("plot_title",
                           type=str,
                           help=("Name of plotly graph"))

    # Parse arguments
    args = args_inst.parse_args()

    # Crate a dictionary having arguments and their values
    args_dict = {'rdir': args.rdir, 'plot_title': args.plot_title}

    # Return arguments as dictionary
    return args_dict


# Execution starts here
if __name__ == "__main__":
    args = _arguments()
    rdir = args['rdir']
    plot_title = args['plot_title']

    # Make sure one json file is present in root directory
    if os.path.isdir(rdir):
        json_file_lst = aqua.fd_ops.get_file_paths_with_kws(rdir, [".json"])
        if not len(json_file_lst) == 1:
            raise Exception(f"{rdir} has {len(json_file_lst)} json files.")
        else:
            json_file = json_file_lst[0]

    # json_file = f"{rdir}/20201023_063237.log.json"  # run2
    smooth_weight = 0.33  # 10% is default according to tensorboard

    # Creating an analyzer instance
    analyzer = TSNAnalyzer(rdir)

    # Training loss
    trnloss, valacc = analyzer.extract_trnloss_and_valacc(json_file)

    trnloss = analyzer.smooth(trnloss, smooth_weight)
    valacc = analyzer.smooth(valacc, smooth_weight)

    # Check if training loss and validation accuracy have same number of
    # epochs
    if len(trnloss) != len(valacc):
        raise Exception(
            "Disagreement between training loss and validation accuracy "
            "number of epochs")

    # Plotting figure using plotly
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=np.arange(len(trnloss)),
                             y=trnloss,
                             mode='lines',
                             name='Training Loss'),
                  row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(valacc)),
                             y=valacc,
                             mode='lines',
                             name='Validation accuracy'),
                  row=2,
                  col=1)

    # Update xaxis
    fig.update_xaxes(title_text="Epochs", row=1, col=1)
    
    fig.update_xaxes(title_text="Epochs", row=2, col=1)

    # Update yaxes
    fig.update_yaxes(title_text="Training loss", row=1, col=1)
    fig.update_yaxes(title_text="Vlidation accuracy", row=2, col=1)

    fig.update_layout(height=1000,
                      title_text=plot_title,
                      font=dict(family="Courier New, monospace", size=23))

    # Writing to chart studio
    chart_studio.tools.set_config_file(world_readable=True)
    py.plot(fig, filename=plot_title)
