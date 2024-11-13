# # Object Detection with Faster RCNN
# 
# Code is for the following video: https://www.youtube.com/watch?v=Uc90rr5jbA4&t=71s

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
import copy
import math
from PIL import Image
import cv2
import albumentations as A  # our data augmentation library
import matplotlib.pyplot as plt

# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, deque
import datetime
import time
from tqdm import tqdm # progress bar
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
from albumentations.pytorch import ToTensorV2


def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.Resize(480, 858), # our input size can be 600px
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(480, 858), # our input size can be 600px
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform


# This is our collating function for the train dataloader, it allows
# us to create batches of data that can be easily pass into the model
def collate_fn(batch):
    return tuple(zip(*batch))

# ## Training
# 
# The following is a function that will train the model for one
# epoch. Torchvision Object Detections models have a loss function
# built in, and it will calculate the loss automatically if you pass
# in the `inputs` and `targets`
def train_one_epoch(model, optimizer, loader, device, epoch, training_log):
    model.to(device)
    model.train()
    
#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000 # do lr warmup
#         warmup_iters = min(1000, len(loader) - 1)
        
#         lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor = warmup_factor, total_iters=warmup_iters)
    
    all_losses = []
    all_losses_dict = []

    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping trainig") # train if loss becomes infinity
            print(loss_dict)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

#         if lr_scheduler is not None:
#             lr_scheduler.step() #
        
    all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
    log_message = ("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
        epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))
    print(log_message)

    # Write log message to log file
    with open(training_log, 'a') as f:
        f.write(f"{log_message}\n")


# ## Dataset
class KeyboardDetection(datasets.VisionDataset):

    
    def __init__(self, root, split='train', training_labels="", transform=None, target_transform=None, transforms=None):
        # the 3 transform parameters are reuqired for datasets.VisionDataset

        super().__init__(root, transforms, transform, target_transform)
        self.split = split #train, valid, test
        self.coco = COCO(os.path.join(root, split, training_labels)) # annotatiosn stored here
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]
    
    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, self.split, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        target = copy.deepcopy(self._load_target(id))
        
        boxes = [t['bbox'] + [t['category_id']] for t in target] # required annotation format for albumentations
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)
        
        image = transformed['image']
        boxes = transformed['bboxes']
        
        new_boxes = [] # convert from xywh to xyxy
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.tensor(new_boxes, dtype=torch.float32)
        
        targ = {} # here is our transformed target
        targ['boxes'] = boxes
        targ['labels'] = torch.tensor([t['category_id'] for t in target], dtype=torch.int64)
        targ['image_id'] = torch.tensor([t['image_id'] for t in target])
        targ['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # we have a different area
        targ['iscrowd'] = torch.tensor([t['iscrowd'] for t in target], dtype=torch.int64)
        return image.div(255), targ # scale images
    
    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":

    dataset_path = "/home/hannah/projects/keyboard_detection/datasets/aolme_from_ty_nty_gt/"
    training_labels = "keyboard_ground_truth.json"
    
    model_save_dir = "/home/hannah/projects/keyboard_detection/models/using_aolme_tynty__no_validation/"
    training_log = f"{model_save_dir}/training_log.txt"

    num_epochs=50
    batch_size = 4
    saving_interval = 2
    num_workers=4
    
    os.path.join(dataset_path, "train", training_labels)
    train_dataset = KeyboardDetection(root=dataset_path, split='train', training_labels=training_labels, transforms=get_transforms(True))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    #load classes
    coco = COCO(os.path.join(dataset_path, "train", training_labels))
    categories = coco.cats
    n_classes = len(categories.keys()) + 1  # The +1 is for background


    # ## Model
    # Our model is FasterRCNN with a backbone of `MobileNetV3-Large`.
    # model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)


    # The following blocks ensures that the model can take in the data and
    # that it will not crash during training
    images,targets = next(iter(train_loader))
    images = list(image for image in images)
    targets = [{k:v for k, v in t.items()} for t in targets]
    output = model(images, targets)

    # Load model to device
    device = torch.device("cuda") # use GPU to train
    model = model.to(device)

    # ## Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True, weight_decay=1e-4)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1) # lr scheduler

    # Resetting training log file
    with open(training_log, 'w') as f:
        f.write("========== Training Log =============\n")


    # Train
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, training_log)

        # Save model
        if epoch%saving_interval == 0:
            torch.save(model.state_dict(), f'{model_save_dir}/epoch_{epoch}.pth')


    #     lr_scheduler.step()




    # Validation
    # model.eval()
    # torch.cuda.empty_cache()
    # test_dataset = KeyboardDetection(root=dataset_path, split="validation", transforms=get_transforms(False))

    # # Visual inspection
    # img, _ = test_dataset[3]
    # img_int = torch.tensor(img*255, dtype=torch.uint8)
    # with torch.no_grad():
    #     prediction = model([img.to(device)])
    #     pred = prediction[0]

    # # Getting categories for bounding boxes
    # pred_threshold = 0.01
    # pred_bboxes = pred['boxes'][pred['scores'] > pred_threshold]

    # pred_cat_idx = pred['labels'][pred['scores'] > pred_threshold].to('cpu').tolist()
    # pred_cat_names = [categories[x]['name'] for x in pred_cat_idx]

    # fig = plt.figure(figsize=(14, 10))
    # plt.imshow(draw_bounding_boxes(img_int,
    #     pred_bboxes,
    #     pred_cat_names, width=4
    # ).permute(1, 2, 0))

    # fig.savefig("visualization.png")
