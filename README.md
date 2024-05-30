# [AICUP 2024] Competition-2024-PyTorch-Tracking


## TEAM_5045: Kelvin, Jonathan, Sam, Henry, Harry  
- [**AI 驅動出行未來：跨相機多目標車輛追蹤競賽 － 模型組**](https://tbrain.trendmicro.com.tw/Competitions/Details/33)  

<a href="https://tbrain.trendmicro.com.tw/Competitions/Details/33"><img src="https://i.imgur.com/3nfLbdW.png" title="source: imgur.com" /></a>  



[![report](https://img.shields.io/badge/Supplementary-Report-yellow)]()
[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FFanChiMao%2FCompetition-2024-PyTorch-Tracking&label=visitors&countColor=%232ccce4&style=plastic)]()


## 🚗 Demo Results
### Here are some tracking results on testing dataset  

<br>


## 📌 Quick Inference
### To reproduce our submit inference results, please following instructions.

<details>
  <summary><span style="font-size: 1.1em; vertical-align: middle;"><b>Step 0: Environment Setting</b></span></summary>

  - **Download the Repo**
    ```commandline
    git clone https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking.git
    git submodule update --init
    ```
  
  - **Prepare the environment**  
    ❗ **Noted:** Please check your GPU and OS environment, and go to the [**PyTorch Website**](https://pytorch.org/get-started/previous-versions/) to install first. 

    ```commandline
    conda create --name AICUP_envs python=3.8
    pip install -r requirements.txt
    ```
  
  - **Prepare datasets**
    - Go to the [**official website**](https://tbrain.trendmicro.com.tw/Competitions/Details/33) to download the datasets.
    - Place testing set (`32_33_AI_CUP_testdataset` folder) in [./datasets](datasets).  
    <br>
  - **Prepare trained model weights**  
    - Go to the download the pretrained weights in our [**github release**]().
    - Place all of the model weights in [./weights](weights)

</details>




<br>

## 📉 Train from scratch
### If you don't want to use our trained model weights, you can consider trained with scratch.

<details>
  <summary><span style="font-size: 1.1em; vertical-align: middle;"><b>Step 0: Environment Setting</b></span></summary>

  - **Download the Repo**
    ```commandline
    git clone https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking.git
    git submodule update --init
    ```
  
  - **Prepare the environment**  
    ❗ **Noted:** Please check your GPU and OS environment, and go to the [**PyTorch Website**](https://pytorch.org/get-started/previous-versions/) to install first. 

    ```commandline
    conda create --name AICUP_envs python=3.8
    pip install -r requirements.txt
    ```
  
  - **Prepare datasets**
    - Go to the [**official website**](https://tbrain.trendmicro.com.tw/Competitions/Details/33) to download the datasets, and place them in the `./datasets` folder.

</details>


<details>
  <summary><span style="font-size: 1.1em; vertical-align: middle;"><b>Step 1: Train Detector (YOLOv9-c model)</b></span></summary>

  - Preprocess the datasets
    ```commandline
    python .\Detector\data_prepare.py --AICUP_dir ./datasets/32_33_train_v2/train --YOLOv9_dir ./datasets/detector_datasets --train_ratio 1
    ```
  
  - Set the correct data path  
    Correct the `path` argument in [**Detector\detector.yaml**](./Detector/detector.yaml) as the path after previous preprocessing  
    <br>
  
  - Start training by using following command
    ```commandline
    python .\Detector\yolov9\train_dual.py --weights .\Detector\yolov9-c.pt --cfg .\Detector\yolov9\models\detect\yolov9-c.yaml --data .\Detector\detector.yaml --device 0 --batch-size 4 --epochs 50 --hyp .\Detector\yolov9\data\hyps\hyp.scratch-high.yaml --name yolov9-c --close-mosaic 15 --cos-lr
    ```
  
  - 📑 For more details about the `Detector` of our method, you can check [**here**](Detector/README.md).

</details>


<details>
  <summary><span style="font-size: 1.1em; vertical-align: middle;"><b>Step 2: Train Extractor (ReID model)</b></span></summary>


  - Preprocess the datasets
    ```commandline
    python .\Extractor\data_prepare.py --AICUP_dir ./datasets/32_33_train_v2/train --reid_dir ./datasets/extractor_datasets
    ```
  
  - Set the correct data path  
    Correct the `path` argument in [**Extractor\extractor.yaml**](./Extractor/extractor.yaml) as the path after previous preprocessing  
    <br>
  
  - Start training by using following command
    ```commandline
    python .\Extractor\train_reid_model.py
    ```
  
  - 📑 For more details about the `Extractor` of our method, you can check [**here**](Extractor/README.md).

</details>
<br>



## 🧾 Reference
- **YOLOv9**: https://github.com/WongKinYiu/yolov9
- **YOLOv8**: https://github.com/ultralytics/ultralytics
- **YOLO-World**: https://github.com/AILab-CVC/YOLO-World
- **torchreid**: https://github.com/KaiyangZhou/deep-person-reid
<br>

## 📫 Contact Us
- **Kelvin**: 
- **Jonathan**: [qaz5517359@gmail.com]()
- **Sam**: 
- **Henry**: 
- **Harry**:
