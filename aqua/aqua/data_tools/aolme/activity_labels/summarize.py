import os
import sys
import pdb
import wget
import json
import pprint
import numpy as np
import pandas as pd
import aqua
import skvideo.io as skvio

class Summarize:
    summary = {}
    """
    Dictionary having summary of activity labels
    """
    
    def __init__(self, df):
        """
        Methods to summarize AOLME Activity labels data.

        Parameters
        ----------
        df: DataFrame
            A dataframe having all the instances of activity labels.

        """
        # get unique activities
        activities = self._get_activites(df)
             
        # Get unique persons involved in the activities
        npersons = self._get_num_persons(df)

        # Get width info (max, med, min)
        winfo = self._get_numerical_property_info(df, "w")

        # Get height info (max, med, min)
        hinfo = self._get_numerical_property_info(df, "h")
        
        # Get time info (max, med, min)
        tinfo = self._get_numerical_property_info(df, "t")

        # Creating properties dictionary
        self.summary = {
            'Activities': activities,
            'Number of persons': npersons,
            'Width (min, med, max)': winfo,
            'Height (min, med, max)':hinfo,
            'Time in sec (min, med, max)':tinfo
        }

        


    def _get_numerical_property_info(self, df, col_name):
        """ Get properties of columns having numerical data.
        For example width, height and time.
        
        Properties
        ----------
        df: DataFrame
            DataFrame having all the activity instances
        col: str
            Column name of numerical data we are processing
        """
        activities = list(df['activity'].unique())

        lst_ = []
        for activity in activities:
            df_ = df[df['activity'] == activity]
            carray = df_[col_name].to_numpy()
            lst_ += [(f"{activity:} "
                      f"({np.round(carray.min(),2)}, "
                      f"{np.round(np.median(carray),2)}, "
                      f"{np.round(carray.max(),2)})")]
        return lst_


        
    def _get_num_persons(self, df):
        """ Returns number of unique persons removing kidx
        """
        persons = list(df['person'].unique())
        if "kidx" in persons: persons.remove("kidx")
        if "Kidx" in persons: persons.remove("Kidx")
        if "KidX" in persons: persons.remove("KidX")
        return len(persons)
        

    def _get_activites(self, df):
        """ Returns activities and their corresponding number instances
        as tuple
        """
        activities = list(df['activity'].unique())

        act_tuple = []
        for activity in activities:
            df_ = df[df['activity'] == activity]
            act_tuple += [[activity , len(df_)]]

        return act_tuple
        
    def to_json(self, opth):
        """ Summarizes to text file located at `opth` if
        overwrite flag is true.
        
        Parameters
        ----------
        opth: str
            Full path to summary file
        """
        # Pretty print
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.summary)

        # Save as json
        with open(opth, 'w') as fp:
            json.dump(self.summary, fp)
        
