# Using mmaction2 in docker
- `$MMR`: mmaction2 root directory (`/home/vj/mmaction2`).
## Tutorial

For tutorial please refer [tutorials.org]('./tutorials.org) and
`../tutorial/` directory.
## Usage
The below use case is written to work with *"kinetics_tiny"* data.
### 1. Dowloading mmaction2 docker image
```bash
sudo docker pull venkatesh369/mmaction2:working
```
### 2. Run/Attach to docker container
- **Run** if `mmaction2` container is not running
```bash
sudo docker run --name mmaction2 --gpus 3 --shm-size 64G -it  \
-v /home/vj/DockerHome/mmaction2:/home -v /mnt/twotb:/mnt/twotb \
-v /home/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ:/home/vj/HAQ venkatesh369/mmaction2:working
su vj
cd ~/mmaction2/
```
- **Attach** if `mmaction2` container is running
```bash
sudo docker attach mmaction2
su vj
cd ~/mmaction2/
```
### 3. Preparing kinetics_tiny dataset
Download and unzip kinetics400_tiny dataset in `$MMR/data`.
### 4. Training (Not using Transfer learning)
Algorithms provided by `mmaction2` are trained either using `RawframeDataset`.
or `VideoDataset`. Typically algorithms that use optical flow vectors are using 
`RawframeDataset`. As of Nov 2, 2020 it supports
`c3d`, `csn`, `i3d`, `omnisource`, `r2plus1d`, `slowfast`, `slowonly`, `tin`,
`tpn`, `tsm`, `tsn`.  
***NOTE:**
- Before training please create `$MMR/checkpoints` to store all the checkpoints.
- Not using tranfer leaning implies we are not resuming/loading from checkpoints.
- All the scripts are tested with *1 RTX5000 GPU*
#### 4.1 `c3d`
- Dataset type: `RawframeDataset`
#### 4.2 `csn`
- Dataset type: `RawframeDataset`
--- 
#### 4.3 `i3d` 
- Dataset type: `RawframeDataset` and `VideoDataset`(<font color="green">Success</font>)
```bash
# From $MMR
cp configs/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb.py configs/kinetics_tiny/
# From $MMR
python tools/train.py configs/kinetics_tiny/i3d_r50_video_32x2x1_100e_kinetics400_rgb.py --validate --seed 0 --deterministic
```
#### 4.4 `omnisource`
- <font color="red">Not using this</font>
#### 4.5 `r2plus1d`
- Dataset type: `RawframeDataset` and `VideoDataset`(<font color="red">Failed</font>)
```bash
# From $MMR
cp configs/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb.py configs/kinetics_tiny/
# From $MMR
python tools/train.py configs/kinetics_tiny/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb.py --validate --seed 0 --deterministic
```
#### 4.6 `slowfast`
- Dataset type: `RawframeDataset` and `VideoDataset`(<font color="green">Success</font>)
```bash
# From $MMR
cp configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py configs/kinetics_tiny/
# From $MMR
python tools/train.py configs/kinetics_tiny/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py 
--validate --seed 0 --deterministic
```
#### 4.7 `slowonly`
- Dataset type: `RawframeDataset` and `VideoDataset`(<font color="green">Success</font>)
```bash
# FROM #MMR
cp configs/recognition/slowonly/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py configs/kinetics_tiny/
# From $MMR
python tools/train.py configs/kinetics_tiny/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py --validate --seed 0 --deterministic
```
#### 4.8 `tin`
- Dataset type: `RawframeDataset`
#### 4.9 `tpn`
- Dataset type: `RawframeDataset`
#### 4.10 `tsm`
- Dataset type: `RawframeDataset` and `VideoDataset`(<font color="green">Success</font>)
```bash
# FROM  $MMR
cp configs/recognition/tsm/tsm_r50_video_1x1x8_50e_kinetics400_rgb.py configs/kinetics_tiny
# FROM  $MMR
python tools/train.py configs/kinetics_tiny/tsm_r50_video_1x1x8_50e_kinetics400_rgb.py --validate --seed 0 --deterministic
```
#### 4.11 `tsn`
- Dataset type: `RawframeDataset` and `VideoDataset`(<font color="green">Success</font>)
```bash
cp configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb.py configs/kinetics_tiny/tsn_r50_video_1x1x8_6e_kinetics400_rgb.py
# From $MMR
python tools/train.py configs/kinetics_tiny/tsn_r50_video_1x1x8_6e_kinetics400_rgb.py --validate --seed 0 --deterministic
```
