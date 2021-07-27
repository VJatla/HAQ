import os
import pdb
import json
import numpy as np
import pandas as pd

class MMAction2Analyzer:
    def __init__(self, rdir):
        """ Methods that aid in analyzing TSN training session.
        Typically a training session has a `json` file (traning
        loss and validation accuracy) and `.pth` checkpoint
        files.

        Parameters
        ----------
        rdir: str
            Root directory having TSN training session
        """
        if not os.path.isdir(rdir):
            raise Exception(f"{rdir} does not exist")
        self._rdir = rdir

    def smooth(self, scalars, weight):  # Weight between 0 and 1
        """ Tensorboard smoothing copied from
            https://shijies.github.io/2018/06/08/How-TensorBoard-Smooth
        """
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (
                1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

        return smoothed

    def extract_trnloss_and_valacc(self, json_path):
        """ Extracts training loss and validation accuracy per epoch
        from json file created during training.

        Parameters
        ----------
        json_path: str
            Path to json file having training loss information

        Returns
        -------
        tuple of list of float
            List having training loss per epoch

        Note
        ----
        The C3D does not dump training loss by default
        """
        if not os.path.isfile(json_path):
            raise Exception(f"{json_path} does not exist")

        with open(json_path, "r") as f:
            lines = f.readlines()

        # Training data frame having all the iterations
        trn_df, val_df = self._create_trnloss_and_valacc_df(lines)

        # Training loss per epoch
        if not trn_df.empty:
            trn_df = trn_df[trn_df['iter'] == trn_df['iter'].max()]
            loss_per_epoch = trn_df['loss'].values.tolist()
        else:
            loss_per_epoch = []

        # Validation accuracy
        if not val_df.empty:
            vacc_per_epoch = val_df['top1 acc'].values.tolist()
        else:
            vacc_per_epoch = []

        if not loss_per_epoch:
            loss_per_epoch = np.zeros(len(vacc_per_epoch)).tolist()
        if not vacc_per_epoch:
            vacc_per_epoch = np.zeros(len(loss_per_epoch)).tolist()

        return (loss_per_epoch, vacc_per_epoch)

    def _create_trnloss_and_valacc_df(self, lines):
        """
        """
        # Creating dataframes for training and validation
        trn_dict_lst = []
        val_dict_lst = []
        for cl in lines:
            cl_dict = json.loads(cl.rstrip())

            if cl_dict['mode'] == "train":
                trn_dict_lst += [cl_dict]
            elif cl_dict['mode'] == "val":
                val_dict_lst += [cl_dict]
            else:
                raise Exception(f"Does not support {cl_dict['mode']}")

        # Creating dataframes from list of dictionaries
        trn_df = pd.DataFrame(trn_dict_lst)
        val_df = pd.DataFrame(val_dict_lst)

        return (trn_df, val_df)
