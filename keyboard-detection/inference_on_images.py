"""Inference on an image."""


import torch
from torchvision import models
import cv2
import albumentations as A  # our data augmentation library
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from torchvision.utils import draw_bounding_boxes
import random
import time
import os
import sys
import shutil

# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")

def overwrite_question(question):
    """Prompt the user with a question and expect a 'yes' or 'no' answer."""
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    while True:
        print(question + " (y/n)")
        choice = input().lower()
        if choice in valid:

            if choice in ["yes", "y", "ye"]:
                return True
            else:
                return False
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').")

if __name__ == "__main__":

    # Random seed
    random_seed_ = 43

    # Input images
    dataset_path = "/home/hannah/projects/keyboard_detection/datasets/aolme_Jun28C2L2_Keyboard/"
    img_extension = "png"
    num_images = 10

    # Model
    model_save_dir = "/home/hannah/projects/keyboard_detection/models/using_coco__no_validation/"
    model_pth = f"{model_save_dir}/epoch_48.pth"
    pred_threshold = 0.5

    # output visualization directory
    output_dir = f"/home/hannah/projects/keyboard_detection/inference/visualization/aolme_Jun28C2L2_Keyboard/random_seed_{random_seed_}"
    
    if os.path.isdir(output_dir):

        # Ask user if he wants to overwrite
        overwrite_ = overwrite_question(f"{output_dir} exists. Overwrite?")

        # If the user decided to not overwrite then exit
        if not overwrite_:
            sys.exit()
        else:
            shutil.rmtree(output_dir)

    os.mkdir(output_dir)    

    # randomly select `n` images
    all_images = [file for file in os.listdir(dataset_path) if file.endswith(f'.{img_extension}')]
    random.seed(random_seed_)
    random_images =  random.sample(all_images, num_images)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ## Model
    # Our model is FasterRCNN with a backbone of `MobileNetV3-Large`.
    # model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)  # 2 = keyabord + background
    model.load_state_dict(torch.load(model_pth))

    # Set the model in evaluation mode
    model.eval()
    torch.cuda.empty_cache()
    model.to(device)

    # Looping over images
    for i, random_image in enumerate(random_images):
        
        img_path = f"{dataset_path}/{random_image}"

        # Copied carefully from training script while going though the
        # exact same transormations it has in validaiton
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = A.Compose([
            A.Resize(480, 858), # our input size can be 600px
            ToTensorV2()
        ])
        transformed = transform(image=image)
        transformed_image = transformed['image']
        image_norm = transformed_image.div(255)
        img_int = torch.tensor(image_norm*255, dtype=torch.uint8)
        with torch.no_grad():
            prediction = model([image_norm.to(device)])
            pred = prediction[0]


        # Getting categories for bounding boxes
        pred_bboxes = pred['boxes'][pred['scores'] > pred_threshold]
        pred_cat_idx = pred['labels'][pred['scores'] > pred_threshold].to('cpu').tolist()
        pred_cat_names = ['keyboard' for x in pred_cat_idx]

        # Showing figure
        fig = plt.figure(figsize=(14, 10))
        plt.imshow(draw_bounding_boxes(
            img_int,
            pred_bboxes,
            pred_cat_names, width=4
        ).permute(1, 2, 0))

        fig_fpath = f"{output_dir}/kbdet_{os.path.splitext(random_image)[0]}.png"
        print(f"Saving {fig_fpath}")
        fig.savefig(fig_fpath)
