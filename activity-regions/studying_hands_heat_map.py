"""
SCRATCH script: Things are hard coded here. I used it only to verify my ideas fast.
Please do not use this script.
"""
import os
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
from aqua.heatmaps import HeatMapsUsingDets

# hand_bboxes_csv = "/home/vj/Dropbox/objectdetection-aolme/keyboard/C1L1P-E/20170302/G-C1L1P-Mar02-E-Irma_q2_02-08_30fps_60_det_per_min.csv"
hand_bboxes_csv = "/home/vj/Dropbox/objectdetection-aolme/hand/C1L1P-E/20170302-5frames/test.csv"

fig, ax = plt.subplots(2,2)

# Create hands heatmap
hm = HeatMapsUsingDets(hand_bboxes_csv, 25)
hm_allbboxes = HeatMapsUsingDets(hand_bboxes_csv, 0)


# Remove regions in heatmap which are < Q1 intensity (quartile 1)
# Without small bounding boxes
hm_np = hm.np_img
hm_flat = hm_np.flatten()
hm_flat_nonzero = hm_flat[np.nonzero(hm_flat)]
q1 = np.percentile(hm_flat_nonzero, 75)
hm_th = 1*(hm_np > q1)

ax[0,0].imshow(hm.np_img, cmap="gray")
ax[0,0].set_title("Without small bboxes")
ax[1,0].imshow(hm_th, cmap="gray")

# Remove regions in heatmap which are < Q1 intensity (quartile 1)
# All bounding boxes
hm_np = hm_allbboxes.np_img
hm_flat = hm_np.flatten()
hm_flat_nonzero = hm_flat[np.nonzero(hm_flat)]
q1 = np.percentile(hm_flat_nonzero, 75)
hm_th = 1*(hm_np > q1)

ax[0,1].imshow(hm_allbboxes.np_img, cmap="gray")
ax[0,1].set_title("All bboxes")
ax[1,1].imshow(hm_th, cmap="gray")

fig.suptitle("G-C1L1P-Apr13-C-Windy_q2_03-07_30fps")


plt.show()
sys.exit()
