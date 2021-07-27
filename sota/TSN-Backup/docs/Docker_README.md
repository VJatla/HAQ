# TSN

- **NOTE:** the base image is venkatesh369/ubuntu1804

## Installation
0. Have `docker` and `nvidia-docker` installed in the system.
1. Pull docker container. Depending on your internet speed it might take quite a time. <font color="red">Please **do not** download vj_settings tag</font>.
```bash
sudo docker pull venkatesh369/mmaction:init
```
2. Run the container with `64 GB` fo RAM and load a local directory to `/home`. We will compile
`mmcv` and `mmaction` here. 
```bash
sudo docker run --name mmaction --gpus 3 --shm-size 64G -it  -v /home/vj/DockerHome/mmaction:/home venkatesh369/mmaction:init
```
**The commands from here are executed in `docker` container.**

3. Create user and clone `mmcv` and `mmaction`
    - Create user with same username as host machine. This way can freely access files from the host and use your fav editor.
      shared folders
      ```bash
      useradd -s /bin/bash -d /home/vj/ -m -G sudo vj
      passwd # as root to set root password
      passwd vj # To set user password
      su vj
      ```
    - Clone repos from my fork to avoid problems due to updates in the original repo
    ```bash
      cd /home/vj
      git clone --recursive https://github.com/vjatla/mmaction.git # My fork of original repository
      git clone https://github.com/vjatla/mmcv.git # My fork of the original repository
    ```
4. Install `mmcv`

 - Before installing `mmcv` install `scikit-build` and change verion of `OpenCV`
   in setup.py
 ```bash
 cd /home/vj/mmcv
 pip install scikit-build --user
 ```
 - Now open `/home/vj/mmcv/setup.py` in a text editor (you can use host system editor here)
   and edit lines having opencv version to
 ```python
 CHOOSE_INSTALL_REQUIRES = [('opencv-python-headless==4.1.0.25', 'opencv-python==4.1.0.25')]
 ```
 - Install mmcv executing the following from `/home/vj/mmcv/`
 ```bash
 pip install -e . --user
 ```
5. Compile `OpenCV` in `mmaction/third_party`
```bash
cd /home/vj/mmaction/third_party/
apt-get install unzip wget # su to root for this
wget -O OpenCV-4.1.0.zip wget https://github.com/opencv/opencv/archive/4.1.0.zip
unzip OpenCV-4.1.0.zip
wget -O OpenCV_contrib-4.1.0.zip https://github.com/opencv/opencv_contrib/archive/4.1.0.zip
unzip OpenCV_contrib-4.1.0.zip
cd opencv-4.1.0
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.0/modules/ -DWITH_TBB=ON -DBUILD_opencv_cnn_3dobj=OFF -DBUILD_opencv_dnn=OFF -DBUILD_opencv_dnn_modern=OFF -DBUILD_opencv_dnns_easily_fooled=OFF -DOPENCV_ENABLE_NONFREE=ON ..
make -j
```
6. Build `denseflow`
```bash
cd /home/vj/mmaction/third_party/dense_flow/
apt-get -qq install libzip-dev libboost-all-dev # as root
mkdir build && cd build
OpenCV_DIR=../../opencv-4.1.0/build cmake ..
make -j
```
7. Compile and install `mmaction`

    ```bash
    cd /home/vj/mmaction
    apt-get install libsm6 # as root
    pip install torch torchvision --user  # On 30th septemebr 2020 it installs pytorch with cuda 10.2
    chmod +x compile.sh
    ./compile.sh
    python setup.py develop --user
    ```

8. Clear out cache before pushing to docker hub
```bash
rm -rf /var/lib/apt/lists/*
rm -rf /home/vj/.cache/pip/*  # <-- Might not be necessary
rm -rf /root/.cache/pip/* # As root (necessary)
```
## Example run
1. Extract frames for videos in `/home/vj/mmaction/data/custom` directory
```bash
cd /home/vj/mmaction/data_tools/custom
bash extract_frames.sh
```
2. Run training with multiple GPUS
```bash
# from /home/vj/mmaction/
./tools/dist_train_recognizer.sh configs/TSN/custom/tsn_flow_bninception_vj.py 3 --validate
```
