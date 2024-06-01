import os
import wget
import argparse
from tqdm import tqdm


def main(target_path):
    target_path = os.path.abspath(target_path)
    weights_name_list = ["osnet_x1_0.pth.tar-50", "yolov8-x_finetune.pt", "yolov8-x_worldv2_pretrained.pt",
                         "yolov9-c_0902.pt", "yolov9-c_1016.pt"]
    weight_source_url = "https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking/releases/tag/v0.0"
    print('It will cost about 5 minutes to download (depends on your network) ...')
    print(f'Start downloading pretrained models from: {weight_source_url}')

    try:
        for i, weight in enumerate(tqdm(weights_name_list)):
            if not os.path.exists(os.path.join(target_path, weight)):
                wget.download(weight_source_url + f"/{weight}", out=os.path.join(target_path, weight))

    except Exception as error:
        print(f"Error Exception: {error}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_path', type=str, default=".")
    args = parser.parse_args()
    main(args.target_path)
