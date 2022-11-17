import json
import os
import numpy as np
import pandas as pd
import pdb
import numpy as np
class LogAnalyzerMMA2:
    def __init__(self, log_path):
        """ Aids in analyzing training, validation and testing logs.

        Parameters
        ----------
        log_path: str
            
        """
        self.trn_df, self.val_df = self._load_json_to_df(log_path)
        
    def get_metric_values(self, mode, metric):
        """ Returns a list having metic values

        Parameters
        ----------
        logfile: str
            Name of log file from which we are extracting the desired metric
        metric: str
            metric that needs to be extracted
        """
        if mode == "trn":
            metric = self.trn_df[metric].tolist()
            epochs = self.trn_df['epoch'].tolist()
        elif mode == "val":
            metric = self.val_df[metric].tolist()
            epochs = self.val_df['epoch'].tolist()
        else:
            sys.exit(f"Does not support mode {mode}")
            
        return epochs, metric
    
    def _load_json_to_df(self, log_path):
        """ Load log file as df. This loader excludeds "info"
        entry.

        Parameters
        ----------
        log_path: str
            Path of log file from which we are extracting the desired metric
        """
        # Loading training json file
        if not os.path.isfile(log_path):
            raise Exception(f"{log_path} does not exist")
        with open(log_path, "r") as f:
            lines = f.readlines()

        # Creating dataframes for training and validation
        trn_dict_lst = []
        val_dict_lst = []
        for cur_line in lines:
            try:
                cur_line_dict = json.loads(cur_line.rstrip())
            except:
                print(cur_line)
                pdb.set_trace()

            # If the current mode is train or validation load it
            if 'mode' in cur_line_dict.keys():
                if (cur_line_dict['mode'] == "train"):
                    trn_dict_lst += [cur_line_dict]
                if (cur_line_dict['mode'] == "val"):
                    val_dict_lst += [cur_line_dict]
                
            
            # if (cur_line_dict['mode'] != "info"):
            #     dict_lst += [cur_line_dict]from aqua.nn.log_analyzer_mmaction2 import LogAnalyzerMMA2


        # Creating dataframes from list of dictionaries
        trn_df = pd.DataFrame(trn_dict_lst)
        val_df = pd.DataFrame(val_dict_lst)

        # Cleaning up training data frame
        trn_df = trn_df.drop('iter', axis=1)
        trn_df = trn_df.groupby('epoch', as_index=False).mean()
        return trn_df, val_df
