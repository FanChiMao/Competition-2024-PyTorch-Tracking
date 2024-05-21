# [AICUP 2024] Competition-2024-PyTorch-Tracking


## TEAM_xxxx: Kelvin, Jonathan, Sam, Henry, Harry  
- [**AI 驅動出行未來：跨相機多目標車輛追蹤競賽 － 模型組**](https://tbrain.trendmicro.com.tw/Competitions/Details/33)  


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

  ```

- Set the correct data path  
  Correct the `path` argument in [**`Detector\detector.yaml`**](./Detector/detector.yaml) as the path after previous preprocessing  
  <br>

- Start training by using following command
  ```commandline
  python .\Detector\yolov9\train_dual.py --weights .\Detector\yolov9-c.pt --cfg .\Detector\yolov9\models\detect\yolov9-c.yaml --data .\Detector\detector.yaml --hyp .\Detector\yolov9\data\hyps\hyp.scratch-high.yaml --name yolov9-c --close-mosaic 15 --cos-lr
  ```
  

## Step 2: Train Extractor (ReID model)
- Preprocess the datasets
  ```commandline
  
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



## Step 4: Integrate Model and Algorithm to inference 
