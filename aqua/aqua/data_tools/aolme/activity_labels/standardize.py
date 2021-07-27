import os
import sys
import pdb
import wget
import pandas as pd
import aqua
import skvideo.io as skvio


class Standardize:
    def __init__(self, rdir, fname, fr, overwrite):
        """
        Methods to standardize AOLME Activity labels data. The
        primary focus is to standardize frame rate.

        Parameters
        ----------
        rdir: str
            Path to directory having activity labels.
        fname: str
            Name of file having activity labels
        fr: int
            Required frame rate
        overwrite: bool
            Overwrite previously standardized videos and activity
            labels?
        """
        self._rdir = rdir
        self._fname = fname
        self._fr = fr
        self._overwrite = overwrite

    def create_activity_labels_at_fr(self):
        """ Creates activity labes at required frame rate.
        """
        ifiles = aqua.get_file_paths_with_kws(self._rdir, [self._fname])
        for ifile in ifiles:

            # Input file properties
            ipth = ifile
            idir = os.path.dirname(ifile)
            iname = os.path.basename(ifile)
            idf = pd.read_csv(ipth)
            itag = ", ".join(idf.iloc[0]['name'].split("-")[1:4])

            # Output file properties
            odir = idir
            oname = (f"{os.path.splitext(iname)[0]}" f"_{self._fr}fps.csv")
            opth = f"{odir}/{oname}"
            oexists = os.path.isfile(opth)

            # If output exists and overwrite is `False` skip
            if oexists and (self._overwrite == False):
                print(f"INFO: Skipping activity labels for --- {itag}")

            # Create activity lables with new frame rate
            else:
                print(f"INFO: Deleting {opth}")
                os.remove(opth)

                ifr = idf['FPS'].unique()
                if len(ifr) > 1:
                    raise Exception(
                        f"ERROR: {ipth} has two unique frame rates {ifr}")
                ifr = round(ifr[0])

                print(f"INFO: Creating activity labels for --- {itag}\n")

                # loop through each activity instance
                odf = idf.copy()
                for i, row in idf.iterrows():

                    # Number of frames in activity instance video
                    ivname = row['name']
                    ovname = (f"{os.path.splitext(ivname)[0]}"
                              f"_{self._fr}fps.mp4")
                    ovpth = f"{odir}/{ovname}"
                    if not os.path.isfile(ovpth):
                        raise Exception(f"ERROR: {ovpth} does not exist")
                    ov = aqua.video_tools.Vid(ovpth)
                    ov_nfrms = int(ov.props['num_frames'])

                    # Calculating multiplication factor for converting frame rate and
                    # numbers
                    mfac = self._fr / round(row['FPS'])

                    # New frame numbers and FPS
                    new_fps = round(row['FPS'] * mfac)
                    new_f0 = round(row['f0'] * mfac)
                    new_f = round(row['f'] * mfac)

                    # A check to make sure we did not have activity labels
                    # outside video
                    final_frame = new_f0 + new_f
                    if final_frame > ov_nfrms:
                        raise Exception(
                            f"ERROR: activity label final frame "
                            f"crossed total number of frames, {ovpth} "
                            f"{final_frame} > {ov_nfrms}")

                    # Change the required fields
                    odf.loc[i, 'name'] = ovname
                    odf.loc[i, 'FPS'] = new_fps
                    odf.loc[i, 'f0'] = new_f0
                    odf.loc[i, 'f'] = new_f
                    odf.to_csv(opth)

    def transcode_videos_at_fr(self, vdb_path):
        """
        Downloads (if required) and transcodes
        videos having activity labels at `fr` frame rate.

        Parameters
        ----------
        vdb_path: str
            Path to video data base csv file.
        """
        # Loading database having download links to videos
        vinfo_df = self._load_vdb(vdb_path)

        # Creating a list of activity label files
        files = aqua.get_file_paths_with_kws(self._rdir, [self._fname])

        # Activity label file loop
        for cfile in files:

            # Read all videos present in the directory containing
            # activity labels
            cfile_path = os.path.dirname(cfile)
            vfiles_present = aqua.get_file_paths_with_kws(cfile_path, [".mp4"])

            # Videos having activity labels
            cact_df = pd.read_csv(cfile)
            vnames = cact_df.name.unique().tolist()

            # Loop over each video
            for vname in vnames:

                # Video name without extension
                vname_noext = os.path.splitext(vname)[0]

                # Input video instance
                iv_path = f"{cfile_path}/{vname_noext}.mp4"
                iv = aqua.video_tools.Vid(iv_path)

                # Output video instance
                ov_path = (f"{cfile_path}/" f"{vname_noext}_{self._fr}fps.mp4")
                ov = aqua.video_tools.Vid(ov_path)

                # Transcode video to required frame rate
                if self._overwrite == True:

                    # Download if input video `iv` is not valid
                    if iv.props['islocal'] == False:
                        self._download_video(iv, vinfo_df)

                    # Transcode video
                    ffmpeg_cmd = (
                        f"ffmpeg -threads 0 -y -i '{iv_path}' -filter:v "
                        f"fps=fps={self._fr} '{ov_path}'")
                    os.system(ffmpeg_cmd)

                    # Remove original video
                    os.remove(iv_path)
                else:

                    # Duration of input and output
                    iv_dur = self._get_video_dur(iv, vinfo_df)
                    ov_dur = ov.props['duration']

                    # Skip if already transcoded properly
                    if ((ov.props['islocal'] == True)
                            and (iv_dur - 1 <= ov_dur <= iv_dur)):
                        print_str = (
                            f"INFO: Skipping {ov.props['name']} "
                            f"(in,out) = ({iv_dur}, {ov_dur}) seconds")
                        if iv_dur != ov_dur:
                            print_str = f"{print_str} WARNING!!!"
                        print(f"{print_str}")

                    else:
                        # Download if `iv` if not present locally
                        if iv.props['islocal'] == False:
                            self._download_video(iv, vinfo_df)

                        # Transcode video
                        ffmpeg_cmd = (
                            f"ffmpeg -threads 0 -y -i '{iv_path}' -filter:v "
                            f"fps=fps={self._fr} '{ov_path}'")
                        os.system(ffmpeg_cmd)

                        # Remove original video
                        os.remove(iv_path)

    def _load_vdb(self, vdb):
        """ Loads data base file (exported from AOLME
        website).

        Also to make further processing easy make sure
        "video_name" has ".mp4" extension.

        Parameters
        ----------
        vdb: str
            Video data base having download link to videos.
        """
        df = pd.read_csv(vdb)
        df_nomp4 = df[~df['video_name'].str.contains(".mp4")]
        df_nomp4['video_name'] += ".mp4"
        df_mp4 = df[df['video_name'].str.contains(".mp4")]
        df = pd.concat([df_nomp4, df_mp4])
        return df

    def _download_video(self, iv, vinfo_df):
        """ Downloads video from AOLME Data server

        Parameters
        ----------
        iv: Vid instance
            video instance from `aqua.video_tools`
        df: DataFrame
            Video information as data frame.
        """
        vname = iv.props['name'] + iv.props['extension']
        vpath = iv.props['dir_loc'] + "/" + vname
        vinfo = vinfo_df[vinfo_df["video_name"] == vname]

        # Raise exception for more than 1 hit in video
        # data base
        if len(vinfo) != 1:
            raise Exception(f"ERROR: {vname} has " f"{len(vinfo)} hits.")

        # Download
        vlink = vinfo['link'].item()
        print(f"INFO: Downloading {vname}")
        wget.download(vlink, vpath)

    def _get_video_dur(self, iv, vinfo_df):
        """ Returns video duration

        Parameters
        ----------
        iv: Vid instance
            video instance from `aqua.video_tools`
        df: DataFrame
            Video information as data frame.

        Returns
        -------
        Duration of video as `int`
        """
        # Return duration for locally present video
        if iv.props['islocal']:
            return iv.props['duration']

        # Get url from video data base
        vname = iv.props['name'] + iv.props['extension']
        vpath = iv.props['dir_loc'] + "/" + vname
        vinfo = vinfo_df[vinfo_df["video_name"] == vname]
        if len(vinfo) != 1:
            raise Exception(f"ERROR: {vname} has " f"{len(vinfo)} hits.")
        vlink = vinfo['link'].item()

        # Get duration from video meta data
        vmeta = skvio.ffprobe(vlink)
        vdur = round(float(vmeta['video']['@duration']))

        return vdur
