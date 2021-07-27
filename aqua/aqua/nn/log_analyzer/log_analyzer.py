import json
import os
import numpy as np
import pandas as pd
import pdb
import numpy as np
class LogAnalyzer:
    def __init__(self, log_path):
        """ Aids in analyzing training, validation and testing logs.

        Parameters
        ----------
        log_path: str
            
        """
        self.df = self._load_json_to_df(log_path)
        
    def get_metric_values(self, metric):
        """ Returns a list having metic values

        Parameters
        ----------
        logfile: str
            Name of log file from which we are extracting the desired metric
        metric: str
            metric that needs to be extracted
        """
        metric = self.df[metric].tolist()
        return metric
    
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
        dict_lst = []
        for cur_line in lines:
            try:
                cur_line_dict = json.loads(cur_line.rstrip())
            except:
                print(cur_line)
                pdb.set_trace()

            # If the current mode is not info load it
            if (cur_line_dict['mode'] != "info"):
                dict_lst += [cur_line_dict]

        # Creating dataframes from list of dictionaries
        df = pd.DataFrame(dict_lst)
        return df
