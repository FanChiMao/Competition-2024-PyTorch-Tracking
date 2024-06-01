import os
import wget
import zipfile

def main():
    wget.download(
        'https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking/releases/download/v0.0/demo_images.zip')

    try:
        print("Start download the demo images...")
        if os.path.exists("./demo_images.zip"):
            with zipfile.ZipFile("demo_images.zip") as zip_ref:
                zip_ref.extractall(".")
    except Exception as error:
        print(f"Error Exception: {error}")
    finally:
        if os.path.exists("./demo_images.zip"):
            os.remove("./demo_images.zip")


if __name__ == '__main__':
    main()
