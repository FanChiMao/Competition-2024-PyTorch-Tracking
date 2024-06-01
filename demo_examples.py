import os
import sys
sys.path.append("Detector")

print(">> Download pretrained weights")
try:
    os.system("python weights/download_model_weights.py --target_path weights")
except Exception as error:
    raise f"Error Exception when download the pretrained weights: {error}"
print('Download pretrained weight successfully !!')


print(">> Download example frames")
if not os.path.exists("assets"):
    os.makedirs("assets", exist_ok=True)
if not os.path.exists("assets/demo_images"):
    try:
        os.system("python assets/download_demo_images.py")
        if not os.path.exists("assets/demo_images"):
            raise "demo_images doesn't exist"
    except Exception as error:
        raise f"Error Exception when download the example images: {error}"
print('Download demo images successfully !!')


print(">> Start quick demo for test dataset: 0902_130006_131041 camera_0")
os.system("python inference_testset.py")
