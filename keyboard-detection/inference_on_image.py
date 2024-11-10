"""Inference on an image."""


import torch
from torchvision import models
import cv2
import albumentations as A  # our data augmentation library
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from torchvision.utils import draw_bounding_boxes

# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")



if __name__ == "__main__":

    dataset_path = "/home/vj/projects/reach/keyboard_detector/dataset/coco/"
    training_labels = "train_coco_format.json"
    validation_labels = "validation_coco_format.json"
    
    model_save_dir = "/home/vj/projects/reach/keyboard_detector/models/using_coco__no_validation/"
    training_log = f"{model_save_dir}/training_log.txt"
    model_pth = f"{model_save_dir}/epoch_9.pth"

    img_path = f"{model_save_dir}/test.png"

    pred_threshold = 0.2

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

    # Copied carefully from training script while going though the
    # exact same transormations
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


    # Display figure
    fig = plt.figure(figsize=(14, 10))

    # Getting categories for bounding boxes
    pred_bboxes = pred['boxes'][pred['scores'] > pred_threshold]

    pred_cat_idx = pred['labels'][pred['scores'] > pred_threshold].to('cpu').tolist()
    pred_cat_names = ['keyboard' for x in pred_cat_idx]

    fig = plt.figure(figsize=(14, 10))
    plt.imshow(draw_bounding_boxes(
        img_int,
        pred_bboxes,
        pred_cat_names, width=4
    ).permute(1, 2, 0))

    fig.savefig("visualization.png")
