This directory contains scripts to process ground truth produced using
MATLAB 2018b Video Labeler application.

## Files:
	1. `+vision/+labeler/fixed_bbox.m`,
		Fixed bounding box class for automation engine.
		It worked with MATLAB 2018b

## Desription:
### Fixed bounding box (Automation Algorithm)
The following automation algorithm for *Video Labeler* allows 
fixed bounding boxes. Basically location and size of the
bouding box does not change.

To use this algorithm make sure to place it in `+vision/+labeler` directory.
Also add this path to MATLAB search paths.

**Example**
My file path,
```bash
C:\Users\vj\Dropbox\Marios_Shared\AOLME-HAR-root\software\AOLME-HAR\ground-truth\MATLAB-video-labeler\+vision\+labeler\fixed_bbox.m
```
To make this algorithm work I have to add
```bash
C:\Users\vj\Dropbox\Marios_Shared\AOLME-HAR-root\software\AOLME-HAR\ground-truth\MATLAB-video-labeler
```
to my path, using `Set Path`(near Preferences) button provided by MATLAB.

