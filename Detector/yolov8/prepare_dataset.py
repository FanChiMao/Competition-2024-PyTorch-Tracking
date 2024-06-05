import os
import glob
import json
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import yaml

def aicup2yolo(label_path, target_path, vit_format=False):

    with open(label_path, "rb") as buffer:
        label_npy = np.fromstring(buffer.read(), dtype=float, sep=" ").reshape(-1, 6)
    
    content = ""
    for i in range(len(label_npy)):
        if vit_format:
            content += "{} {:.16f} {:.16f} {:.16f} {:.16f}\n".format(int(label_npy[i][5]), label_npy[i][1], label_npy[i][2], label_npy[i][3], label_npy[i][4])
        else:
            content += "{} {:.16f} {:.16f} {:.16f} {:.16f}\n".format(int(label_npy[i][0]), label_npy[i][1], label_npy[i][2], label_npy[i][3], label_npy[i][4])
    
    with open(target_path, "w") as write_file:
        write_file.write(content)

def uadetrac2singlelabel(label_path, target_path):

    with open(label_path, "rb") as buffer:
        label_npy = np.fromstring(buffer.read(), dtype=float, sep=" ").reshape(-1, 5)
    
    content = ""
    for i in range(len(label_npy)):
        content += "{} {:.16f} {:.16f} {:.16f} {:.16f}\n".format(0, label_npy[i][1], label_npy[i][2], label_npy[i][3], label_npy[i][4])
    
    with open(target_path, "w") as write_file:
        write_file.write(content)
        
def prepare_AICUP_data(data_dir, target_dir, vit_format=False):
    image_list = glob.glob(os.path.join(data_dir, "images", "*", "*.jpg"))
    label_list = glob.glob(os.path.join(data_dir, "labels", "*", "*.txt"))
    for image_file, label_file in tqdm(zip(image_list, label_list), desc="prepare AICUP data", total=len(image_list)):
        dirname = os.path.basename(os.path.dirname(image_file))
        basename = os.path.basename(image_file).replace(".jpg", "")
        filename = dirname + "_" + basename
        if dirname.startswith("1016"):
            # val
            shutil.copy(image_file, os.path.join(target_dir, "val", "images", filename + ".jpg"))
            aicup2yolo(label_file, os.path.join(target_dir, "val", "labels", filename + ".txt"), vit_format)
        else:
            # train
            shutil.copy(image_file, os.path.join(target_dir, "train", "images", filename + ".jpg"))
            aicup2yolo(label_file, os.path.join(target_dir, "train", "labels", filename + ".txt"), vit_format)

def prepare_UADETRAC_date(data_dir, target_dir):
    image_list = glob.glob(os.path.join(data_dir, "images", "*", "*.jpg"))
    label_list = glob.glob(os.path.join(data_dir, "labels", "*", "*.txt"))
    
    image_dict = {}
    label_dict = {}
    for image_file in tqdm(image_list):
        filename = os.path.basename(image_file).replace(".jpg", "")
        image_dict[filename] = image_file
    
    for label_file in tqdm(label_list):
        filename = os.path.basename(label_file).replace(".txt", "")
        label_dict[filename] = label_file
    
    for filename in tqdm(label_dict, desc="prepare UADETRAC data", total=len(image_list)):
        image_file = image_dict[filename]
        label_file = label_dict[filename]
        shutil.copy(image_file, os.path.join(target_dir, "train", "images", filename + ".jpg"))
        uadetrac2singlelabel(label_file, os.path.join(target_dir, "train", "labels", filename + ".txt"))
        
def prepare_BDD100K_date(data_dir, target_dir):
    #{'bus', 'traffic sign', 'train', 'rider', 'motorcycle', 'trailer', 'car', 'pedestrian', 'truck', 'bicycle', 'other person', 'other vehicle', 'traffic light'}
    for fold in tqdm(["train", "val"], desc="prepare BDD100 data"):
        
        with open(os.path.join(data_dir, "labels", "det_20", "det_{}.json".format(fold)), "r") as read_file:
            label_infos = json.load(read_file)
            
        for info in label_infos:
            image_file = os.path.join(data_dir, "images", "100k", fold, info["name"])
            image = cv2.imread(image_file)
            H,W,C = image.shape
            
            content = ""                
            if info.get("labels"):
                for label in info.get("labels"):
                    if label["category"] in ["bus", "trailer", "car", "truck", "other vehicle"]:
                        x1, y1, x2, y2 = label["box2d"]["x1"], label["box2d"]["y1"], label["box2d"]["x2"], label["box2d"]["y2"]
                        xc, yc, w, h = (x1 + x2)*.5/W, (y1 + y2)*.5/H, (x2 - x1)/W, (y2 - y1)/H
                        content += "{} {:.16f} {:.16f} {:.16f} {:.16f}\n".format(0, xc, yc, w, h)
            
            shutil.copy(image_file, os.path.join(target_dir, "train", "images", info["name"]))
            with open(os.path.join(target_dir, "train", "labels", info["name"].replace(".jpg", ".txt")), "w") as write_file:
                write_file.write(content)

def write_yaml(target_dir, yaml_file):

    yml={
    "names": ["car"],
    "nc": 1,
    "path": target_dir,
    "train": "./train/images",
    "val": "./val/images"
    }

    with open(yaml_file, "w") as write_file:
        yaml.dump(yml, write_file, default_flow_style=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Dataset Preparation for Training YOLOv8 Detector") 
    parser.add_argument('--aicup', type=str, required=True, help="Directory of AICUP dataset")
    parser.add_argument('--uadetrac', type=str, required=True, help="Directory of UADETRAC dataset")
    parser.add_argument('--bdd100k', type=str, required=True, help="Directory of BDD100K dataset")
    parser.add_argument('--output', type=str, required=True, help="Directory for output dataset")
    args = parser.parse_args()

    AICUP_data_dir = args.aicup
    UADETRAC_data_dir = args.uadetrac
    BDD100K_data_dir = args.bdd100k    
    target_data_dir = args.output
    
    if os.path.exists(target_data_dir):
        shutil.rmtree(target_data_dir)    
    os.makedirs(os.path.join(target_data_dir, "train", "images"))
    os.makedirs(os.path.join(target_data_dir, "train", "labels"))
    os.makedirs(os.path.join(target_data_dir, "val", "images"))
    os.makedirs(os.path.join(target_data_dir, "val", "labels"))

    prepare_AICUP_data(AICUP_data_dir, target_data_dir)
    prepare_UADETRAC_date(UADETRAC_data_dir, target_data_dir)
    prepare_BDD100K_date(BDD100K_data_dir, target_data_dir)
    
    write_yaml(target_data_dir, os.path.join(target_data_dir, "data.yml"))