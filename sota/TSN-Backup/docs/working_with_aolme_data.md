# ??? Need proper documentation

# TSN using AOLME data
## Starting container
This loads `DockerHome`(having mmaction library) and `/mnt/twotb`(having data) into the container.
```bash
udo docker run --name mmaction --gpus 3 --shm-size 64G -it  -v /home/vj/DockerHome/mmaction:/home -v /mnt/twotb:/mnt/twotb -v /home/vj/Dropbox/Marios_Shared/HAQ-AOLME/software/HAQ:/mnt/HAQ venkatesh369/mmaction:vj_settings

# Switch to user vj
su vj
```
## Extracting frames
```bash
cd /home/vj/mmaction/data_tools/tynty/
bash extract_frames.sh
```
