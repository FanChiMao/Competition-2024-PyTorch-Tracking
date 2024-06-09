import os
import wget
import zipfile
import argparse

def main(target_path):
    target_path = os.path.abspath(target_path)
    wget.download('https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking/releases/download/v0.0/demo_images.zip',
                  out=target_path)

    try:
        print("Start download the demo images...")
        if os.path.exists(os.path.join(target_path, "./demo_images.zip")):
            with zipfile.ZipFile(os.path.join(target_path, "./demo_images.zip")) as zip_ref:
                zip_ref.extractall(target_path)
    except Exception as error:
        print(f"Error Exception: {error}")
    finally:
        if os.path.exists(os.path.join(target_path, "./demo_images.zip")):
            os.remove(os.path.join(target_path, "./demo_images.zip"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_path', type=str, default=".")
    args = parser.parse_args()
    main(args.target_path)
