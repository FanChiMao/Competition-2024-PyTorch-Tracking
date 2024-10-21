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

