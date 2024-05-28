# [AICUP 2024] Competition-2024-PyTorch-Tracking


## TEAM_5045: Kelvin, Jonathan, Sam, Henry, Harry  
- [**AI 驅動出行未來：跨相機多目標車輛追蹤競賽 － 模型組**](https://tbrain.trendmicro.com.tw/Competitions/Details/33)  

<a href="https://tbrain.trendmicro.com.tw/Competitions/Details/33"><img src="https://i.imgur.com/3nfLbdW.png" title="source: imgur.com" /></a>  

## Step 0: Environment Setting

- Download the Repo
    ```commandline
    git clone https://github.com/FanChiMao/Competition-2024-PyTorch-Tracking.git
    git submodule update --init
    ```
  
- Prepare the environment
    ```commandline
    conda create --name AICUP_envs python=3.8
    pip install -r requirements.txt
    ```
  
- Prepare datasets  
  - Go to the [official website](https://tbrain.trendmicro.com.tw/Competitions/Details/33) to download the datasets, and place them in `./datasets` folder.


## Step 1: Train Detector (YOLOv9 model)
- Preprocess the datasets
  ```commandline
  python .\Detector\data_prepare.py --AICUP_dir ./datasets/32_33_train_v2/train --YOLOv9_dir ./datasets/detector_datasets --train_ratio 1
  ```

- Set the correct data path  
  Correct the `path` argument in [**`Detector\detector.yaml`**](./Detector/detector.yaml) as the path after previous preprocessing  
  <br>

- Start training by using following command
  ```commandline
  python .\Detector\yolov9\train_dual.py --weights .\Detector\yolov9-c.pt --cfg .\Detector\yolov9\models\detect\yolov9-c.yaml --data .\Detector\detector.yaml --device 0 --batch-size 4 --epochs 50 --hyp .\Detector\yolov9\data\hyps\hyp.scratch-high.yaml --name yolov9-c --close-mosaic 15 --cos-lr
  ```
  

## Step 2: Train Extractor (ReID model)
- Preprocess the datasets
  ```commandline
  python .\Extractor\data_prepare.py --AICUP_dir ./datasets/32_33_train_v2/train --reid_dir ./datasets/extractor_datasets
  ```

- Set the correct data path  
  Correct the `path` argument in [**`Extractor\extractor.yaml`**](./Extractor/extractor.yaml) as the path after previous preprocessing  
  <br>

- Start training by using following command
  ```commandline
  python .\Extractor\train_reid_model.py
  ```

[//]: # (TODO: visualize feature and evaluate)


## Step 3: Design a Tracker (Track algorithm)
- Preprocess the datasets for **evaluating** Tracker results
  ```commandline
  python .\Tracker\data_prepare.py --AICUP_dir ./datasets/32_33_train_v2/train --MOT15_dir ./datasets/mot_gt_datasets
  ```



## Step 4: Integrate Model and Algorithm to inference 
