# <font color="red">Abandoned</font>

# `SlowFast`
As suggested please start from venkatesh369/ubuntu1804:cuda102 image. I have updated image in this repository with my personal settings and you might need to delete user `vj` if
you are going to use this. 
```bash
deluser --remove-home vj # as root from the container
```

## Installation
1. Pull ubuntu18.04 with cuda 10.2 docker container
```bash
sudo docker pull venkatesh369/ubuntu1804:cuda102
```
2. Run docker mounting home directory to shared folder
```bash
sudo docker run --name slowfast --gpus 3 --shm-size 64G -it  -v /home/vj/DockerHome/SlowFast:/home venkatesh369/ubuntu1804:cuda102
```
3. Crate an user account (I strongly recommend to create with same account
name as host to avoid file permission problems in future)
```bash
useradd -s /bin/bash -d /home/vj/ -m -G sudo vj
passwd vj # Setting user password
passwd # Setting root password
su vj
cd ~
```

4. Install prerequisites (things that are to be executed as root are explicitly
mentioned with '# root').
    **Root section**
    ```bash
    apt-get update # root
    apt-get install -y build-essential python3-dev python3-setuptools make cmake libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev python3-pip python-pip git pkg-config# root
    ln -sf /usr/bin/python3 /usr/bin/python # root
    ln -sf /usr/bin/pip3 /usr/bin/pip # root
    # Video/Audio tools and libraries
    apt-get install -y liblapack-dev libatlas-base-dev # root
    apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libavdevice-dev# root
    apt-get install -y software-properties-common # root
    add-apt-repository ppa:jonathonf/ffmpeg-4 # root
    apt update # root
    apt install -y ffmpeg # root
    ```
    **User section**
    ```bash
    # User section
    cd ~
    pip install torch torchvision --user 
    pip install 'git+https://github.com/facebookresearch/fvcore' --user
    pip install simplejson --user
    pip install av --user
    pip install opencv-python==4.1.0.25 --user
    pip install psutil --user
    pip install tensorboard --user
    pip install moviepy --user
    pip install cython --user
    pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' --user
    # Installing detectron2
    git clone https://github.com/vjatla/detectron2 detectron2_repo # Cloning from my repo (forked on Oct 1, 2020)
    pip install -e detectron2_repo --user
    ```
3. Installing `SlowFast`
    - Clone
     ```bash
      # from /home/vj
      git clone https://github.com/vjatla/slowfast # Cloning from my repo to avoid update issues
    ```
    - Add to path
      ```bash
      # Add following line to ~/.bashrc (if not exist create it)
      export PYTHONPATH=/home/vj/slowfast/slowfast:$PYTHONPATH
      source ~/.bashrc
      ```
    - Install
      ```bash
      # from /home/vj/slowfast
      python setup.py build develop --user
      ```
## Usage
Please follow the instructions in the official repo(https://github.com/facebookresearch/SlowFast) or its fork(https://github.com/VJatla/SlowFast) 
