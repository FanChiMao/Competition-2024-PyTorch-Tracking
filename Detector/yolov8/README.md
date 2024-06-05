# Train A YOLOv8 Detector with External Datasets

## Fetch Data
1. BDD100K
[DOWNLOAD LINK](https://dl.cv.ethz.ch/bdd100k/data/)
Download 3 of the following files 100k_images_train.zip, 100k_images_val.zip and bdd100k_det_20_labels_trainval.zip.
Unzip 3 files and organize the directory:
```
bdd100k
  - images
    - 100k
      - train
        - XXXX.jpg
        - ...
      - val
        - XXXX.jpg
        - ...
  - labels
    - det_20
      - det_train.json
      - det_val.json
```
2. UA-DETRACE
[DOWNLOAD LINK](https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset)
Download the dataset from Kaggle
Unzip the file and keeps only DETRAC_Upload folder
```
DETRAC_Upload
  - images
    - train
      - XXXX.jpg
      - ...
    - val
      - XXXX.jpg
      - ...
  - labels
    - train
      - XXX.txt
      - ...
    - val
      - XXX.txt
      - ...
```
3. AICUP
This dataset is conly available on T-Brain Machine Learning Competition site (not v2 version)

## Data Preparation
Run the following command to merge 3 datasets into one
```
python prepare_dataset.py --aicup AICUP_DATASET_DIR --uadetrac UADETRAC_DATASET_DIR --bdd100k BDD100K_DATASET_DIR --output OUTPUT_DIR
```
- Note that in L44 of prepare_dataset.py, we only use images and labels in 10/16 for validation

## Train YOLOv8 Detector
Run the following command to train yolov8 detector
```
python train_detecctor.py --data_dir DATA_DIR
```