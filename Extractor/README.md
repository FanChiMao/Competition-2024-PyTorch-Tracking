# Extractor

We directly train/evaluate the feature extractor by the [torchreid library](https://pypi.org/project/torchreid/).  

We also use the vehicle ReID pretrained model for fine-tuning.
```commandline
model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True
)
```


To visualize the feature embeddings are correct distributed in high-dimension space, you can follow the [**tools/visualize_features.py**](tools/visualize_features.py).  
After running the visualization via the T-SNE, the result will be  

- Example feature space after extracting the features  
  [1016_150000_151900_camera_0](feature_space_examples/1016_150000_151900_camera_0.html)  
  [1016_150000_151900_camera_1](feature_space_examples/1016_150000_151900_camera_1.html)  
  [1016_150000_151900_camera_2](feature_space_examples/1016_150000_151900_camera_2.html)  
  [1016_150000_151900_camera_3](feature_space_examples/1016_150000_151900_camera_3.html)
   


