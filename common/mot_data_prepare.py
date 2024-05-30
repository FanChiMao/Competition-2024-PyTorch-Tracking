import os
import argparse

from tqdm import tqdm


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--AICUP_dir', type=str, default="")
    parser.add_argument('--MOT15_dir', type=str, default="", help='converted dataset directory')
    parser.add_argument('--imgsz', type=tuple, default=(720, 1280), help='img size, (height, width)')

    opt = parser.parse_args()
    return opt


def show_files(path, all_files):
    file_list = os.listdir(path)
    file_list.sort()
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files)
        else:
            if not cur_path.endswith('txt'):
                continue
            else:
                all_files.append(cur_path)
    return all_files


def AI_CUP_to_MOT15(args):
    os.makedirs(args.MOT15_dir, exist_ok=True)
    total_files = show_files(args.AICUP_dir, [])

    frame_ID = 0
    timestamp = ''
    img_height, img_width = args.imgsz

    for src_path in tqdm(total_files, desc='convert to MOT15 format'):
        text = src_path.split(os.sep)
        if 'labels' in text:
            if timestamp != text[-2]:
                timestamp = text[-2]
                frame_ID = 0

            frame_ID = frame_ID + 1
            f_mot15 = open(os.path.join(args.MOT15_dir, timestamp + '.txt'), 'a+')
            f_aicup = open(src_path, 'r')

            for line in f_aicup.readlines():
                data = line.split(' ')
                bb_width = float(data[3]) * img_width
                bb_height = float(data[4]) * img_height
                bb_left = float(data[1]) * img_width - bb_width / 2
                bb_top = float(data[2]) * img_height - bb_height / 2
                track_id = data[5].split('\n')
                f_mot15.write(
                    f'{str(frame_ID)},{track_id[0]},{str(bb_left)},{str(bb_top)},{str(bb_width)},{str(bb_height)},1,-1,-1,-1\n'
                )

            f_aicup.close()

        f_mot15.close()


if __name__ == '__main__':
    args = arg_parse()

    # local run
    # args.AICUP_dir = r"D:\Jonathan\AI_project\ObjectTracking\code\Competition-2024-PyTorch-Tracking\datasets\32_33_train_v2\train"
    # args.MOT15_dir = r"D:\Jonathan\AI_project\ObjectTracking\code\Competition-2024-PyTorch-Tracking\datasets\mot_gt_datasets"

    AI_CUP_to_MOT15(args)
