# Datasets

Due to the rule of the competition, the public training/testing dataset could only download from the [**official website**](https://tbrain.trendmicro.com.tw/Competitions/Details/33). 

- Official Public Dataset  
  Totally contain **31947** images/labels in `32_33_train_v2` folder with the txt label format as  

  |         | Class ID | center x | center y | width  | height | Track ID |
  |:-------:|:--------:|:--------:|:--------:|:------:|:------:|:--------:| 
  | Example |    0     |  0.4717  |  0.1956  | 0.0593 | 0.0990 |    1     |


- Process Public Dataset for Training the `Detector`  
  After process the official public dataset with track id for training the detector, the txt label format will be  

  |         | Class ID | center x | center y | width  | height |
  |:-------:|:--------:|:--------:|:--------:|:------:|:------:|
  | Example |    0     |  0.4717  |  0.1956  | 0.0593 | 0.0990 |


- Process Public Dataset for Training the `Extractor`  
  After process the official public dataset with track id for training the extractor, the image label folder will be  

  ```commandline
  extractor_datasets
    ├── track_id_1
    |     ├── track_id_1_crop_image_1.png
    |     ├── track_id_1_crop_image_2.png
    |      ...
    |
    ├── track_id_2
    |     ├── track_id_1_crop_image_1.png
    |     ├── track_id_1_crop_image_2.png
    |      ...  
    .
    .
    .
    └── track_id_4964
          ├── track_id_4964_crop_image_1.png
          ├── track_id_4964_crop_image_2.png
           ...  
  ```

- Official Public/Private Testing Dataset  
  Totally contain **22403** images/labels in `32_33_AI_CUP_testdataset` folder.  
