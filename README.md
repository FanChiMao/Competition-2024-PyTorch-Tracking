# [AICUP 2024] Competition-2024-PyTorch-Tracking

📹 **Extremely low frame-rate (1 fps) video object tracking challenge**  

## TEAM_5045: Kelvin, Jonathan, Sam, Henry, Harry  
> This project is developed by the Product R&D Department of the Digital Image Technology Division at ASUS Computer Inc. AI Solution Business Unit.

---

- [**AI 驅動出行未來：跨相機多目標車輛追蹤競賽 － 模型組**](https://tbrain.trendmicro.com.tw/Competitions/Details/33)  
  
<a href="https://tbrain.trendmicro.com.tw/Competitions/Details/33"><img src="https://i.imgur.com/3nfLbdW.png" title="source: imgur.com" /></a>  
> In recent years, surveillance camera systems have been widely used on roads due to the demands for
home security and crime prevention. Since most surveillance systems are currently based on singlecamera recording, each camera operates independently, making it impossible to continue identifying
moving objects once they leave the field of view. Additionally, in the event of accidents or criminal
incidents, because each camera records independently and there is no mechanism for cooperative
operation between cameras, law enforcement agencies must expend significant manpower resources to
manually search through surveillance recordings to track the paths and trajectories of suspicious vehicles
or pedestrians. 


<a href="https://drive.google.com/file/d/1QTJYdmQ3_8ppwGKZ8fVVssbLCv1NZZKb/view?usp=sharing" target="_blank">
  <img src="https://img.shields.io/badge/Supplementary-Report_EN-yellow" alt="Report-en">
</a>

<a href="https://drive.google.com/file/d/1Lzh9F76LVzpfDlPsITBbl5rvRX9_pFtB/view?usp=sharing" target="_blank">
  <img src="https://img.shields.io/badge/Supplementary-Report_CH-yellow" alt="Report-ch">
</a>

<a href="https://colab.research.google.com/drive/1c9WrlZB4_OPnpzf22jhei6grAo9_VLoe?usp=sharing" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>

<a href="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FFanChiMao%2FCompetition-2024-PyTorch-Tracking&label=visitors&countColor=%232ccce4&style=plastic" target="_blank">
  <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FFanChiMao%2FCompetition-2024-PyTorch-Tracking&label=visitors&countColor=%232ccce4&style=plastic" alt="Visitors">
</a>

<a href="https://img.shields.io/github/downloads/FanChiMao/Competition-2024-PyTorch-Tracking/total" target="_blank">
  <img src="https://img.shields.io/github/downloads/FanChiMao/Competition-2024-PyTorch-Tracking/total" alt="Download">
</a>



## 🎉 This work earn the 2nd place among 286 participated teams


<details>
  <summary><b>Award 🏆</b></summary>
<a href="https://imgur.com/9RPgk64"><img src="https://i.imgur.com/9RPgk64.jpg" title="source: imgur.com" /></a>
</details>

<details>
  <summary><b>LeaderBoard 🎖️</b></summary>
<a href="https://imgur.com/j5BbnzQ"><img src="https://i.imgur.com/j5BbnzQ.png" title="source: imgur.com" /></a>
</details>


<br>


## 🚗 Demo Results
### Here are some tracking results on testing dataset.



<details>
  <summary><b>Example demo results 📷</b></summary>

<br>


1. High movement speed   

  https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking/assets/85726287/22b109dc-f615-400f-ab2d-825cbbfb8047


2. Unstable connection (Heavy)

  https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking/assets/85726287/b1e3f8d7-e37e-49c2-a4b2-b60e7249349d

3. Unstable connection (Slight)

  https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking/assets/85726287/77f081ba-6088-4b60-b0c4-1329919eaadf

4. Flare issue

  https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking/assets/85726287/a0e5cffd-710b-4094-9e86-2eb4475a8196

  https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking/assets/85726287/1c0cc0b1-7b82-479c-b7b4-85351d68ad8f

6. Disconnect issue

  https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking/assets/85726287/36a4ea7b-f0f5-4387-97d2-62c2d10a8485

</details>


<br>

## 🗿 Model Architecture
<a href="https://imgur.com/QBq40de"><img src="https://i.imgur.com/QBq40de.png" title="source: imgur.com" /></a>


[//]: # (## 👀 Design the Tracker for low frame-rate tracking)

[//]: # (TODO)


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
    - Go to the download the pretrained weights in our [**release**](https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking/releases).
    - Place all the model weights in [./weights](weights).
    - Or you can run the python script in [./weights/download_model_weights.py](./weights/download_model_weights.py)

</details>

<details>
  <summary><span style="font-size: 1.1em; vertical-align: middle;"><b>Step 1: Set the yaml file correctly</b></span></summary>
  
  - Modify the inference setting ([**inference_testset.yaml**](inference_testset.py)) to prepare inference (following setting is our best submitted setting)

      ```
        # [Default Setting]
        Default :
          RESULT_FOLDER: ./aicup_results
          FRAME_FOLDER: ./datasets/32_33_AI_CUP_testdataset/AI_CUP_testdata/images
        
          # write mot (AICUP submit) txt file
          WRITE_MOT_TXT: true
        
          # write final inference video
          SAVE_OUT_VIDEO: true
          SAVE_OUT_VIDEO_FPS: 3
        
        
        # [Detector]
        Detector:
          ENSEMBLE: true  # if set true, fill the detector_weight_list and corresponding score
          ENSEMBLE_MODEL_LIST: [
                weights/yolov9-c_0902.pt,
                weights/yolov9-c_1016.pt,
                weights/yolov8-x_finetune.pt,
                weights/yolov8-x_worldv2_pretrained.pt
            ]
          ENSEMBLE_WEIGHT_LIST: [0.8, 0.76, 0.75, 0.7]
        
          DETECTOR_WEIGHT: weights/yolov9-c_0902.pt
          DETECTOR_CONFIDENCE: 0.05
        
        
        # [Extractor]
        Extractor:
          EXTRACTOR_WEIGHT: weights/osnet_x1_0.pth.tar-50
          EXTRACTOR_TYPE: osnet_x1_0
          EXTRACTOR_THRESHOLD: 0.6
        
        
        # [Tracker]
        Tracker:
          TRACKER_MOTION_PREDICT: lr  # lr / kf  (Linear / Kalman Filter)
          TRACKER_MAX_UNMATCH_FRAME : 3

      ```
</details>

<details>
  <summary><span style="font-size: 1.1em; vertical-align: middle;"><b>Step 2: Run inference code</b></span></summary>

  - After setting the correct configuration of yaml file, simply run:
    ```commandline inference_testset.py
    python inference_testset.py
    ```

- ⏱ **We use single RTX 2070 (8 GB) to run the inference, here is the estimation inference time for single/ensemble model:**  
  - 1 model (YOLOv9c model): about 45 minutes
  - 2 model (YOLOv9c + YOLOv8x): about 1 hours
  - 3 model (YOLOv9c + YOLOv8x + YOLOv8world): about 2 hours
  - 4 model (YOLOv9c + YOLOv8x + YOLOv8world + YOLOv9c): about 2.5 hours

</details>

<br>

## ⚙️ Train on scratch
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

### 1. Preprocess the datasets  
- After downloading the dataset from official website, simply run  

    ```
    python .\Detector\yolov9\data_prepare.py --AICUP_dir ./datasets/32_33_train_v2/train --YOLOv9_dir ./datasets/detector_datasets --train_ratio 1
    ```

### 2. Train YOLOv9 Detector
  - Set the correct data path  
    Correct the `path` argument in [**Detector\detector.yaml**](Detector/detector.yaml) as the path after previous preprocessing  
    <br>
  
  - Start training by using following command
    ```
    python .\Detector\yolov9\train_dual.py --weights .\Detector\yolov9-c.pt --cfg .\Detector\yolov9\models\detect\yolov9-c.yaml --data .\Detector\detector.yaml --device 0 --batch-size 4 --epochs 50 --hyp .\Detector\yolov9\data\hyps\hyp.scratch-high.yaml --name yolov9-c --close-mosaic 15 --cos-lr
    ```
  
💬 For more details about the `Detector` of our method, you can check [**here**](Detector/README.md).
<br>

</details>

<details>
  <summary><span style="font-size: 1.1em; vertical-align: middle;"><b>Step 2: Train Detector with External Datasets (YOLOv8-x model)</b></span></summary>
    
### 1. Fetch Data  
- BDD100K  
[DOWNLOAD LINK](https://dl.cv.ethz.ch/bdd100k/data/)
Download 3 of the following files 100k_images_train.zip, 100k_images_val.zip and bdd100k_det_20_labels_trainval.zip.
Unzip 3 files and organize the directory:  

    ```
    bdd100k
      - images
        - 100k
          - train
            - XXXX.jpg
            - ...
          - val
            - XXXX.jpg
            - ...
      - labels
        - det_20
          - det_train.json
          - det_val.json
    ```
- UA-DETRACE  
[DOWNLOAD LINK](https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset)
Download the dataset from Kaggle
Unzip the file and keeps only DETRAC_Upload folder  

    ```
    DETRAC_Upload
      - images
        - train
          - XXXX.jpg
          - ...
        - val
          - XXXX.jpg
          - ...
      - labels
        - train
          - XXX.txt
          - ...
        - val
          - XXX.txt
          - ...
    ```  

### 2. Data Preparation  

This dataset is can only available on T-Brain Machine Learning Competition site (not v2 version)  
<br>
Run the following command to merge 3 datasets into one  

```
python prepare_dataset.py --aicup AICUP_DATASET_DIR --uadetrac UADETRAC_DATASET_DIR --bdd100k BDD100K_DATASET_DIR --output OUTPUT_DIR
```
- Note that in L44 of prepare_dataset.py, we only use images and labels in 10/16 for validation

### 3. Train YOLOv8 Detector
Run the following command to train yolov8 detector
```
python train_detecctor.py --data_dir DATA_DIR
```
💬 For more details about the `Detector` of our method, you can check [**here**](Detector/README.md).
<br>

</details>


<details>
  <summary><span style="font-size: 1.1em; vertical-align: middle;"><b>Step 3: Train Extractor (ReID model)</b></span></summary>


### 1. Preprocess the datasets
- After downloading the dataset from official website, simply run  

    ```
    python .\Extractor\data_prepare.py --AICUP_dir ./datasets/32_33_train_v2/train --reid_dir ./datasets/extractor_datasets
    ```

### 2. Train OSNet Extractor
  - Set the correct data path  
    Correct the `path` argument in [**Extractor\extractor.yaml**](./Extractor/extractor.yaml) as the path after previous preprocessing  
    <br>
  
  - Start training by using following command
    ```
    python .\Extractor\train_reid_model.py
    ```
  
💬 For more details about the `Extractor` of our method, you can check [**here**](Extractor/README.md).

</details>
<br>



## 🧾 Reference
- **YOLOv9**: https://github.com/WongKinYiu/yolov9
- **YOLOv8**: https://github.com/ultralytics/ultralytics
- **YOLO-World**: https://github.com/AILab-CVC/YOLO-World
- **torchreid**: https://github.com/KaiyangZhou/deep-person-reid
<br>

## 📫 Contact Us
- **Kelvin**: [fxp61005@gmail.com]()  
- **Jonathan**: [qaz5517359@gmail.com]()  
- **Sam**: [a839212013@gmail.com]()  
- **Henry**: [jupiter5060812@gmail.com]()
- **Harry**: [ms024929548@gmail.com]()  
