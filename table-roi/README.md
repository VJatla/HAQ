# Table ROIs

## 1. Creating a session video
0. **Why?** To create table region of interests *fast*.
1. **Files:** `create_session_videos.py` and `rois.py` 
2. **Description:** We first create session video taking one frame every 
	1 second and combining all the videos in a session.
3. **Usage:**
	```bash
	# Change paths in create_session_videos.py
	python create_session_videos.py
	```

## 2. Labeling session video
We use MATLAB 2021b vidoe labeler to draw bounding boxes. The alorithm we use
is *"temporal interpolation"*. We save
	1. video labeling session as `session_roi.mat`.
	2. labels exported to mat file as `session_roi_exported.mat`.

## 3. Labels to csv files
Since labels in .mat files are hard to access we export them to csv file using
the script `mat_to_csv.m`. 

## 4. Video level roi
Once we have session level roi we need to map it back to the videos. To do this
run the script `session_roi_to_video.py`.