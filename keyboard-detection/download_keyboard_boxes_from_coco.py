import os
import fiftyone as fo
import fiftyone.zoo as foz

def get_data_and_export(download_dir, data_split_label):

    # Load the COCO 2017 validation dataset (downloads it if not already downloaded)
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=data_split_label,
        label_types=["detections"],
        classes=["keyboard"],        # Specify that we only want images with keyboards
        #max_samples=50               # Limit the number of samples to load (optional)
    )

    # Export the dataset in COCO format
    dataset.export(
        export_dir=f"{download_dir}/{data_split_label}/",
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
        classes=["keyboard"],
        labels_path=f"{download_dir}/{data_split_label}/{data_split_label}_coco_format.json",
        data_path = f"{download_dir}/{data_split_label}/"
    )


if __name__ == "__main__":

    # Root directory to download
    download_dir = "/home/vj/projects/reach/keyboard_detector/dataset/coco/"
    
    # Get training coco data for keyboard
    get_data_and_export(download_dir, "train")
    get_data_and_export(download_dir, "validation")
