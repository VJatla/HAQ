# Pre-processing
Scripts that help in processing activity ground truth created using MATLAB Video
Labeler.
## 1. `mat to csv.m`
The following script reads bounding boxes tracked using *point tracker* or 
*fixed_bbox* automation algorithm to csv file.  
This script addresses the issue of varying size and position of bounding box
(when using point tracker) by fixing the position and size to initialization.

## 2. Standardize frame rate
Before preparing data for further processing we need to standardize frame rate.
This is done by first standardizing videos then the activity labels csv files.
```bash
# Before standardizing labels please standardize the videos
python tynty-standardize_video_framerate.py
python wnw-standardize_video_framerate.py
python tynty-standardize_label_framerate.py
python wnw-standardize_label_framerate.py
```
## 3. Spatiotemporal video trimming
To trim videos please use `wnw-trim-activity-labels.py` and `tynty-trim-activity-labels.py`.
