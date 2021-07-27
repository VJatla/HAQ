# `mmaction2` installation in docker
The following instructions of installing `mmaction2` applies
to RTX3 machine. They can be used on other machines but might
require some tweaks.

## 1. Pull and start docker container
```bash
sudo docker pull venkatesh369/ubuntu1804:cuda102
mkdir ~/DockerHome/mmaction2
sudo docker run --name mmaction2 --gpus 3 --shm-size 64G -it  -v /home/vj/DockerHome/mmaction2:/home venkatesh369/ubuntu1804:cuda102
```
## 2. Create user and switch
```bash
useradd -s /bin/bash -d /home/vj/ -m -G sudo vj
passwd # as root to set root password
passwd vj # To set user password
su vj
```
## 3. Save container
```bash
sudo docker commit mmaction2 venkatesh369/mmaction2:created_user
```
## 4. Libraries needed
Reload saved image
```bash
sudo docker run --name mmaction2 --gpus 3 --shm-size 64G -it  -v /home/vj/DockerHome/mmaction2:/home venkatesh369/mmaction2:created_user
```
### 4.1 Python
```bash
# As root
apt-get update
apt-get install python3 python3-dev python3-pip python python-pip
ln -sf /usr/bin/pip3 /usr/bin/pip
ln -sf /usr/bin/python3 /usr/bin/python
```
### 4.2 Torch
```bash
su vj
cd ~
pip install torch==1.6.0 torchvision==0.7.0 --user
```
### 4.3 FFMPEG
It is recommended to install version 4.2.
```bash
# As root
apt-get install software-properties-common
add-apt-repository ppa:jonathonf/ffmpeg-4
apt-get update
apt-get install -y build-essential python3-dev python3-setuptools make cmake libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
apt-get install ffmpeg
apt-get install git
```
### 4.4 Decord
Before compiling from source we need to make sure that libnvcuvid is present. For
cuda versions > 9.2 it is not provided. To handle this, download 
[NVIDIA VIDEO CODEC SDK ](https://developer.nvidia.com/nvidia-video-codec-sdk)
and copy the header files to your cuda path (/usr/local/cuda-10.0/include/ for example)
```bash
unzip Video_Codec_SDK_11.0.10.zip
cp Video_Codec_SDK_11.0.10/Interface/nvcuvid.h /usr/local/cuda-10.2/include/
cp Video_Codec_SDK_11.0.10/Interface/cuviddec.h //usr/local/cuda-10.2/include/
cp Video_Codec_SDK_11.0.10/Lib/linux/stubs/x86_64/libnvcuvid.so /usr/local/cuda-10.2/lib64/libnvcuvid.so.1
cp Video_Codec_SDK_11.0.10/Lib/linux/stubs/x86_64/libnvcuvid.so /usr/local/cuda-10.2/lib64/libnvcuvid.so
```
Now compile decord from source
```bash
# As vj
cd /home/vj
git clone --recursive https://github.com/dmlc/decord
cd decord
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.2/bin/nvcc -DCMAKE_BUILD_TYPE=Release
make
```
Install python bindings for decord
```bash
cd /home/vj/decord/python
pwd=$PWD
echo "PYTHONPATH=$PYTHONPATH:$pwd" >> ~/.bashrc
source ~/.bashrc
python setup.py install --user
```
### 4.5 pyav 
	```bash
# As root
apt-get install pkg-config
apt-get install libavformat-dev libavdevice-dev
# As user
pip install av --user
```
### 4.6 PyTurboJPEG
```bash
# As user
pip install PyTurboJPEG --user
```
### 4.7 Dense flow
Installing dense flow using scripts loacted at [github](https://github.com/innerlee/setup).
We need,
1. boost
2. opencv
3. denseflow
ALl the scripts are located at `/home/vj/denseflow_scripts` for installation.

```bash
# As user
cd /home/vj/denseflow_scripts

# Boost (As user)
bash boost.sh
# Add the following to .bashrc
export BOOST_ROOT=/home/vj//app
source ~/.bashrc

# OpenCV (As user)
bash opencv.sh
# Add the following to .bashrc
export OpenCV_DIR=/home/vj/app
source ~/.bashrc

# cmake
apt-get remove cmake # As root
apt-get install libssl-dev
bash cmake.sh # As user
# Edit bashrc
export PATH=$PATH:/home/vj/app/bin
source ~/.bashrc

# Denseflow
bash denseflow.sh
```
On installing the libraries are located at `/home/vj/app`.
### 4.8 moviepy 
```bash
# As root
apt-get install imagemagick
# As user
pip install moviepy --user
```
### 4.9 Pillow-SIMD 
```bash
# As root
apt-get install zlib1g-dev libjpeg-dev
# As user
CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
conda install -y jpeg libtiff
```
### 4.10 onnx 
```bash
# As user
pip install onnx --user
pip install onnxruntime --user
```
### 4.11 mmcv
	MMCV should be above 1.1.1
```bash
pip install scikit-build --user
pip install mmcv-full==1.1.2+torch1.6.0+cu102 -f https://download.openmmlab.com/mmcv/dist/index.html --user
```
## 5. Installing mmaction2
```bash
# Using personal fork to support commits in later phase
git clone https://github.com/vjatla/mmaction2.git
cd mmaction2
pip install -v -e .
```
## 6. Clean and save docker container
### 6.1 Clean
```bash
# As user
cd ~
rm -r Video_Codec_SDK_11.0.10*
# As root
rm -rf /var/lib/apt/lists/*
rm -rf /root/.cache/pip/* # As root (necessary)
```
### 6.2Commit
After the container is exited use the following commands to commit.
```bash
# Exit docker
sudo docker commit mmaction2 venkatesh369/mmaction2:working
sudo docker push venkatesh369/mmaction2:working
```
