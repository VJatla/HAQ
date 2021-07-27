""" Test bench for creating activity classifier
"""
import pdb
import torch
from aqua.nn.dloaders import AOLMETrmsDLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=20, 
                               kernel_size=3, stride=1)
        self.conv2 = nn.Conv3d(in_channels=20, out_channels=50, 
                               kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features= 450, out_features=35)
        self.fc2 = nn.Linear(in_features=35, out_features=10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, 5, 5)
        x = F.relu(self.conv2(x))
        x = F.max_pool3d(x, 11 , 11)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(net, trainloader):
    for epoch in range(10): # no. of epochs
        print(f"INFO: Training for epoch {epoch}")
        running_loss = 0
        for idx, data in enumerate(trainloader):
            print(f"INFO: iteration {idx}")
            # data pixels and labels to GPU if available
            labels, inputs = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            # set the parameter gradients to zero
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # propagate the loss backward
            loss.backward()
            # update the gradients
            optimizer.step()
 
            running_loss += loss.item()
        print('[Epoch %d] loss: %.3f' %
                      (epoch + 1, running_loss/len(trainloader)))
 
    print('Done Training')


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


# directory location w.r.t rtx3
vdir = "/mnt/twotb/aolme_datasets/tynty/trimmed_videos/one_trim_per_instance_3sec_224/"
trn_list = "/mnt/twotb/aolme_datasets/tynty/trimmed_videos/one_trim_per_instance_3sec_224/trn_videos.txt"



# Device available
device = get_device()


# Training data
train_data = AOLMETrmsDLoader(vdir, trn_list)

# Training loader
trainloader = DataLoader(train_data, batch_size=8, shuffle=False, num_workers=8)



# Network
net = Net().to(device)

# loss
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(net)

# Training
train(net, trainloader)
pdb.set_trace()
