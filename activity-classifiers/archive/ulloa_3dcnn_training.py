import sys
import pdb
import torch
import torch.nn as nn
import torch.optim as optim 
from aqua.nn.models import UlloaCNN3D
from torch.utils.data import DataLoader
from aqua.nn.dloaders import AOLMETrmsDLoader
from aqua.nn.trainer import SingleGPUTrainer

    
# DataLoaders
#     Using sizes similar to Ulloa oshape = (width, height), 
# Training
vdir = "/home/vj/Videos/one_trim_per_instance_3sec_224"
trn_list = "/home/vj/Videos/one_trim_per_instance_3sec_224/trn_videos_100per_act.txt"
train_data = AOLMETrmsDLoader(vdir,
                              trn_list, oshape=(150, 109)) 
trainloader = DataLoader(train_data, batch_size=24,
                         shuffle=True, num_workers=24)
# Validation
val_list = "/home/vj/Videos/one_trim_per_instance_3sec_224/val_videos_25per_act.txt"
val_data = AOLMETrmsDLoader(vdir,
                              val_list, oshape=(150, 109)) 
valloader = DataLoader(val_data, batch_size=24,
                         shuffle=True, num_workers=24)


# Build model
cnn3d = UlloaCNN3D(3, 3, nchannels=3, debug_prints=False)
pytorch_total_params = sum(p.numel() for p in cnn3d.parameters())
print(pytorch_total_params)
sys.exit()

# Loss and optimizer
criterion = nn.BCELoss() # Using Binary Cross Entropy loss
optimizer = optim.SGD(cnn3d.parameters(), lr=0.001, momentum=0.9)

# Initialize single gpu training instance
training_params = {
    "max_epochs": 100,
    "work_dir":"/home/vj/Workdir/ulloa_3x3x3/tynty/run1",
    "ckpt_save_interval": 1,
    "log_pth": "/home/vj/Workdir/ulloa_3x3x3/tynty/run1/log.json"
}
trainer = SingleGPUTrainer(training_params,
                           cnn3d,
                           optimizer,
                           criterion,
                           trainloader,
                           valloader,
                           cuda_device_id=0)

# Train
training_time = trainer.train()
