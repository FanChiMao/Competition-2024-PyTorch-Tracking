import os
import argparse
from ultralytics import YOLO

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a YOLOv8 Detector") 
    parser.add_argument('--epochs', type=int, default=100, help="The number of epochs to train the model")
    parser.add_argument('--imgsz', type=int, default=640, help="The size of the input image to the model")
    parser.add_argument('--batch', type=int, default=8, help="The batch size for training the model")
    parser.add_argument('--data_dir', type=str, required=True, help="The directory of the dataset for training the model")
    parser.add_argument('--model', type=str, default="yolov8x.pt", help="The path or model name for the pretrained model")
    args = parser.parse_args()

    epochs=args.epochs
    imgsz=args.imgsz
    batch=args.batch  
    data_dir = args.data_dir
    pretrained_model_path = args.model
    
    data_yml = os.path.join(data_dir, "data.yml")

    model = YOLO(model=pretrained_model_path)
    results = model.train(data=data_yml, epochs=epochs, imgsz=imgsz, batch=batch)    
    metrics = model.val()