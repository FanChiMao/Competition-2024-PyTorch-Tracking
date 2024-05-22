import os
import glob
import shutil
import argparse

from tqdm import tqdm
from glob import glob

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--AICUP_dir', type=str, default='', help='your AICUP dataset path')
    parser.add_argument('--YOLOv9_dir', type=str, default='', help='converted dataset directory')
    parser.add_argument('--train_ratio', type=float, default=1, help='The ratio of the train set when splitting the train set and the validation set')

    opt = parser.parse_args()
    return opt


def aicup_to_yolo(args):
    train_dir = os.path.join(args.YOLOv9_dir, 'train')
    valid_dir = os.path.join(args.YOLOv9_dir, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)

    os.makedirs(os.path.join(valid_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, 'labels'), exist_ok=True)

    total_files = sorted(
        os.listdir(
            os.path.join(args.AICUP_dir, 'images')
        )
    )

    total_count = len(total_files)
    train_count = int(total_count * args.train_ratio)

    train_files = total_files[:train_count]
    valid_files = total_files[train_count:]

    for src_path in tqdm(glob.glob(os.path.join(args.AICUP_dir, '*', '*', '*')), desc=f'copying data'):
        text = src_path.split(os.sep)
        timestamp = text[-2]
        camID_frameID = text[-1]

        train_or_valid = 'train' if timestamp in train_files else 'valid'

        if 'images' in text:
            dst_path = os.path.join(args.YOLOv9_dir, train_or_valid, 'images', timestamp + '_' + camID_frameID)
        elif 'labels' in text:
            dst_path = os.path.join(args.YOLOv9_dir, train_or_valid, 'labels', timestamp + '_' + camID_frameID)

        shutil.copy2(src_path, dst_path)

    return 0


def delete_track_id(labels_dir):
    for file_path in tqdm(glob.glob(os.path.join(labels_dir, '*.txt')), desc='delete_track_id'):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            text = line.split(' ')

            if len(text) > 5:
                new_lines.append(line.replace(' ' + text[-1], '\n'))

        with open(file_path, 'w') as f:
            f.writelines(new_lines)

    return 0


def generate_cross_validation_txt(root):
    root = os.path.abspath(root)
    image_path = os.path.join(root, "images")
    total_images = glob(os.path.join(image_path, "*.jpg"))

    cross_validation = ['0902', '0903', '0924', '0925', '1015', '1016']
    for i, validation_set in enumerate(cross_validation):
        for _, image in enumerate(tqdm(total_images, desc=f"=> Fold {i} Train/Valid splitting")):
            name = os.path.basename(image)
            date = name.split("_")[0]
            if date == validation_set:
                with open(os.path.join(root, f"{validation_set}_valid.txt"), "a") as f:
                    f.write(f"./images/{name}\n")
            else:
                with open(os.path.join(root, f"{validation_set}_train.txt"), "a") as f:
                    f.write(f"./images/{name}\n")

        generate_yaml_file(f"./detector_{validation_set}.yaml", root, f"{validation_set}_train.txt", f"{validation_set}_valid.txt")

def generate_yaml_file(file_path, root, train, valid):
    content = f"""
# Path Setting
path: {root}  # dataset root dir
train: {train}
val: {valid}
test:
                    
# Classes
names:
 0: Car
"""

    with open(file_path, "w") as file:
        file.write(content)


if __name__ == '__main__':
    args = arg_parse()

    # local run
    # current_file = os.path.dirname(os.path.abspath(__file__))
    # args.AICUP_dir = os.path.join(current_file, "../datasets/32_33_train_v2/train")
    # args.YOLOv9_dir = os.path.join(current_file, "../datasets/detector_datasets")
    # args.train_ratio = 1  # split train/val by date

    aicup_to_yolo(args)
    train_dir = os.path.join(args.YOLOv9_dir, 'train', 'labels')
    delete_track_id(train_dir)

    dataset_root = os.path.join(args.YOLOv9_dir, 'train')
    generate_cross_validation_txt(dataset_root)

