import os
import sys
import pdb
import wget
import aqua
import shutil
import pandas as pd
import skvideo.io as skvio


class Process:
    _lab_fpaths = []
    """ Array of paths containing activity labels."""
    def __init__(self, idir, lab_fname):
        """
        Methods to process activity labels. It currently supports
        1. one_trim_per_instance


        Parameters
        ----------
        idir: str
            Path to directory having activity labels and videos
        lab_fname: str
            Name of file having activity labels
        """
        self._lab_fpaths = aqua.fd_ops.get_file_paths_with_kws(
            idir, [lab_fname])

        if len(self._lab_fpaths) == 0:
            raise Exception(f"Cannot find {lab_fname} at {idir}")

    def one_trim_per_instance(self, odir, dur, overwrite):
        """
        Creates one spatiotemporal trim of duration `dur` per an 
        activity instance. The trims are stored in a directory 
        named after the activity it contains. The trims are taken
        from the middle.

        For exaple typing and notyping activities are stored in
        `odir/typing` and `odir/notyping` respectively.

        Parameters
        ----------
        odir: str
            Path of direcotry to store trims.
        dur: int
            Duration of each trim in seconds.
        overwrite: bool
            Overwrites existing trims.
        """
        # Array to hold trimmed videos information
        arr = []

        # Make output directory if it does not exist
        if not os.path.isdir(odir):
            os.mkdir(odir)
        else:
            if overwrite:
                print(f"INFO: Recreating {odir}")
                shutil.rmtree(odir, ignore_errors=True)
                os.mkdir(odir)

        # Initializing video indexes for each activity to 0
        activities = self._get_activities()
        tvidx = dict()
        for act in activities:
            tvidx[act] = 0

        for lab_fpath in self._lab_fpaths:
            dir_loc = os.path.dirname(lab_fpath)
            df = pd.read_csv(lab_fpath)

            for idx, inst in df.iterrows():

                # Load video
                inst_vpth = f"{dir_loc}/{inst['name']}"
                inst_vid = aqua.video_tools.Vid(inst_vpth)
                if not inst_vid.props['islocal']:
                    raise Exception(f"ERROR: {inst_vpth} not found")

                # Make output activity directory if it doesn not exist
                inst_dir = f"{odir}/{inst['activity']}"
                if not os.path.isdir(inst_dir):
                    os.mkdir(inst_dir)

                # greater than dur(argument) sec instances are processed
                inst_dur = int(inst['f']) / int(inst['FPS'])
                if inst_dur >= dur:
                    f0 = int(inst['f0'])
                    f = int(inst['f'])
                    fps = int(inst['FPS'])
                    mfrm = f0 + round(f / 2)
                    sfrm = int(mfrm - ((dur / 2) * fps))
                    efrm = int(mfrm + ((dur / 2) * fps))

                    # IF starting frame is less than f0 and ending frame is
                    # greater than f0 + f throw error.
                    if sfrm < f0 or efrm > (f0 + f):
                        raise Exception("Trim is going beyong f0 and f0+f")

                    person = inst['person']
                    bbox = [inst['w0'], inst['h0'], inst['w'], inst['h']]
                    bbox = [int(x) for x in bbox]
                    cact = inst['activity']

                    vi = aqua.data_tools.aolme.parse_video_name(inst['name'])
                    sess_info = (f"C{vi['cohort']}-"
                                 f"L{vi['level']}-"
                                 f"{vi['school']}-"
                                 f"{vi['group']}-"
                                 f"{vi['date']}")

                    # Trimmed video name
                    out_vpath = (f"{inst_dir}/"
                                 f"tv_{tvidx[cact]}_{inst_vid.props['name']}_"
                                 f"{sfrm}_{efrm}_{person}.mp4")
                    tvidx[cact] += 1

                    trim_loc = inst_vid.spatiotemporal_trim(
                        sfrm, efrm, bbox, out_vpath)

                    arr = arr + [[
                        f"{inst_vid.props['name']}.mp4", sess_info, sfrm,
                        efrm - sfrm, inst['w0'], inst['h0'], inst['w'],
                        inst['h'], person, trim_loc
                    ]]

        # Saving trimmed video information as csv file
        df = pd.DataFrame(arr,
                          columns=[
                              'name', 'session', 'f0', 'f', 'w0', 'h0', 'w',
                              'h', 'person', 'trim_loc'
                          ])

        # Saving trimmed videos information for future processing and analyzing
        save_loc = f"{odir}/all_trims.csv"
        df.to_csv(save_loc, index=False)

    def trim_instances(self, odir, overwrite):
        """
        Creates one spatiotemporal trim of duration `dur` per an 
        activity instance. The trims are stored in a directory 
        named after the activity it contains.

        For exaple typing and notyping activities are stored in
        `odir/typing` and `odir/notyping` respectively.

        Parameters
        ----------
        odir: str
            Path of direcotry to store trims.
        overwrite: bool
            Overwrites existing trims.
        """
        # Array to hold trimmed videos information
        arr = []

        # Make output directory if it does not exist
        if not os.path.isdir(odir):
            os.mkdir(odir)
        else:
            if overwrite:
                print(f"INFO: Recreating {odir}")
                shutil.rmtree(odir, ignore_errors=True)
                os.mkdir(odir)

        # Initializing video indexes for each activity to 0
        activities = self._get_activities()
        tvidx = dict()
        for act in activities:
            tvidx[act] = 0

        for lab_fpath in self._lab_fpaths:
            dir_loc = os.path.dirname(lab_fpath)
            df = pd.read_csv(lab_fpath)

            for idx, inst in df.iterrows():

                # Load video
                inst_vpth = f"{dir_loc}/{inst['name']}"
                inst_vid = aqua.video_tools.Vid(inst_vpth)
                if not inst_vid.props['islocal']:
                    raise Exception(f"ERROR: {inst_vpth} not found")

                # Make output activity directory if it doesn not exist
                inst_dir = f"{odir}/{inst['activity']}"
                if not os.path.isdir(inst_dir):
                    os.mkdir(inst_dir)

                # Trim entire duration
                sfrm = int(inst['f0'])
                efrm = int(inst['f0']) + int(inst['f'])
                person = inst['person']
                bbox = [inst['w0'], inst['h0'], inst['w'], inst['h']]
                cact = inst['activity']

                # Creating group name
                vi = aqua.data_tools.aolme.parse_video_name(inst['name'])
                sess_info = (f"C{vi['cohort']}-"
                             f"L{vi['level']}-"
                             f"{vi['school']}-"
                             f"{vi['group']}-"
                             f"{vi['date']}")

                # Trimmed video name
                out_vpath = (
                    f"{inst_dir}/"
                    f"tv_{tvidx[cact]}_{inst_vid.props['name']}_{sfrm}_{efrm}"
                    f"_{person}.mp4")
                tvidx[cact] += 1

                trim_loc = inst_vid.spatiotemporal_trim(
                    sfrm, efrm, bbox, out_vpath)

                arr = arr + [[
                    f"{inst_vid.props['name']}.mp4", sess_info, sfrm,
                    efrm - sfrm, inst['w0'], inst['h0'], inst['w'], inst['h'],
                    person, trim_loc
                ]]

        # Saving trimmed video information as csv file
        df = pd.DataFrame(arr,
                          columns=[
                              'name', 'session', 'f0', 'f', 'w0', 'h0', 'w',
                              'h', 'person', 'trim_loc'
                          ])

        # Saving trimmed videos information for future processing and analyzing
        save_loc = f"{odir}/all_trims.csv"
        df.to_csv(save_loc, index=False)

    def _get_activities(self):
        """ Loads all activity labels into one dataframe.
        """
        dflst = []
        for f in self._lab_fpaths:
            dflst += [pd.read_csv(f)]

        df = pd.concat(dflst, ignore_index=True)
        activities = list(df['activity'].unique())

        return activities
