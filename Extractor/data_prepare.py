import os
import cv2
from glob import glob
from tqdm import tqdm
import argparse


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--AICUP_dir', type=str, default='', help='your AICUP dataset path')
    parser.add_argument('--reid_dir', type=str, default='', help='converted dataset directory')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = arg_parse()

    # local run
    # root_path = r"D:\Jonathan\AI_project\ObjectTracking\code\Competition-2024-PyTorch-Tracking\datasets\32_33_train_v2\train"
    # save_path = r"D:\Jonathan\AI_project\ObjectTracking\code\Competition-2024-PyTorch-Tracking\datasets\extractor_datasets"

    root_path = args.AICUP_dir
    save_path = args.reid_dir
    os.makedirs(save_path, exist_ok=True)
    image_path = os.path.join(root_path, "images")
    label_path = os.path.join(root_path, "labels")
    image_files = glob(os.path.join(image_path, "*", "*.jpg")) + glob(os.path.join(image_path, "*", "*.png"))
    label_files = glob(os.path.join(label_path, "*", "*.txt"))

    exist_track_id = dict()
    for image_path, label_path in tqdm(zip(image_files, label_files), desc="process crops labels"):
        image_name = os.path.basename(image_path)
        timestamp = image_path.split("\\")[-2]
        with open(label_path, 'r') as f:
            lines = f.readlines()

        img = cv2.imread(image_path)
        height, width, c = img.shape

        for line in lines:
            text = line.split(' ')
            x_norm, y_norm, w_norm, h_norm = map(float, text[1:5])
            track_id = int(text[-1])

            xc = int(x_norm * width)
            yc = int(y_norm * height)
            w = int(w_norm * width)
            h = int(h_norm * height)

            x = xc - w // 2
            y = yc - h // 2

            cropped_image = img[y:y + h, x:x + w, :]

            save_crop_img_path = os.path.join(save_path, str(track_id))
            if track_id not in exist_track_id:
                os.makedirs(save_crop_img_path, exist_ok=True)
                exist_track_id[track_id] = 1
            else:
                exist_track_id[track_id] += 1

            cv2.imwrite(os.path.join(save_crop_img_path, f"{timestamp}_{image_name}"), cropped_image)
