# Detector record
> To record the detail results of our detector models such as
> YOLOv9c, YOLOv8x and YOLOv8World.


## YOLOv9-c_0902
- Training Data: Official Training dataset without the suffix start with `0902_...`
- Validation Data: Official Training dataset with the suffix start with `0902_...`  
- Hyperparameters: [hyperparameters/hyp_YOLOv9c_0902.yaml](hyperparameters/hyp_YOLOv9c_0902.yaml)

## YOLOv9-c_1016
- Training Data: Official Training dataset without the suffix start with `1016_...`
- Validation Data: Official Training dataset with the suffix start with `1016_...`   
- Hyperparameters: [hyperparameters/hyp_YOLOv9c_1016.yaml](hyperparameters/hyp_YOLOv9c_1016.yaml)

## YOLOv8-x_finetune
- Training Data: Official Training dataset without the suffix start with `1016_...` + BDD100K + UA-DETRACE
- Validation Data: Official Training dataset with the suffix start with `1016_...`   
- Hyperparameters: [hyperparameters/hyp_YOLOv8x_1016.yaml](hyperparameters/hyp_YOLOv8x_1016.yaml)



## YOLOv8-x_worldv2_pretrained
- Training Data: directly using the pretrained model  
- Validation Data: directly using the pretrained model    
