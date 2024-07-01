# Table ROI labeling

This document outlines the table ROI labeling procedure. The labeling
procedure is designed to be fast, sparse and an outline. For this
tutorial we will be using the group videos form [C2L2-A on
20180614](https://aolme.unm.edu/Videos/cur_group_videos.php?cohort=2&school=Polk&level=2&group=A&date=2018-06-14).

Please change the paths as necessary.

## 1. Dowload videos

Download/Copy all the group interaction vidoes to a directory.

![C2L2-A, 20180614 video files](./pictures/table_roi_labeling/C2L2-A_20180614_video_files.png)

## 2. Standardize frame rate to 30 fps

If the downloaded videos are already at 30fps add *"_30fps"* at the
end of its name. You can check the frame rate using VLC media player, `Tools -> Media information -> Codec`.

![Frame rate using VLC](./pictures/table_roi_labeling/C2L2-A_20180614_framerate.png)

Otherwise, please use `ffmpeg` to change the frame rate.
The command is,

```shell
ffmpeg -i <input.mp4> -r 30 -c:v copy -c:a copy <output.mp4>
```

## 2. Create a sessioon video

We first combined all the videos in a session taking one frame every
second. We call this file, `session_video.mp4`. A video that has $`t`$
seconds will contribute $`t+1`$ frames.

The script that does this is, `create_session_videos.py`. Located at
`HAQ/table-roi`.

```shell
# Change directory to the python scrip
cd /home/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ/table-roi

# Create a session video that is 1 fps
python create_session_videos.py /home/vj/Dropbox/table_roi_annotation/C2L2P-A/20180614 all
```

## 3. 