import wget
from tqdm import tqdm


def main():
    print('It will cost about 5 minutes to download...')
    try:
        with tqdm(total=5) as bar:
            wget.download('https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking/releases/download/v0.0/osnet_x1_0.pth.tar-50')
            bar.update(1)
            wget.download('https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking/releases/download/v0.0/yolov8-x_finetune.pt')
            bar.update(1)
            wget.download('https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking/releases/download/v0.0/yolov8-x_worldv2_pretrained.pt')
            bar.update(1)
            wget.download('https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking/releases/download/v0.0/yolov9-c_0902.pt')
            bar.update(1)
            wget.download('https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking/releases/download/v0.0/yolov9-c_1016.pt')
            bar.update(1)

    except Exception as error:
        print(f"Error Exception: {error}")



if __name__ == '__main__':
    print('Start downloading pretrained models from https://github.com/FanChiMao/SRMNet-thesis/releases/tag/v0.0')
    main()
    print('Done !!')