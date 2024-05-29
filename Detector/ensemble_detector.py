import os
import sys
import cv2

current_file = os.path.abspath(__file__)
sys.path.append(os.path.join(current_file, "yolov9"))
from Detector.yolov9.models.common import DetectMultiBackend, AutoShape
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm
import numpy as np


class DetectMultiBackendEnsemble(object):
    def __init__(self, weights_list, ensemble_weight_list, device):
        self.weight_path_list = weights_list
        self.ensemble_weight_list = ensemble_weight_list
        self.device = device
        self.model_instances = []
        self._set_model_instances()

    def _set_model_instances(self):
        for _, weight in enumerate(tqdm(self.weight_path_list, desc="Load ensemble model weights")):
            if "yolov8" in weight:
                from ultralytics import YOLO
                model = YOLO(weight)
                if "world" in weight:
                    model.set_classes(["car", "bus"])
            else:  # yolov9
                model = DetectMultiBackend(weights=weight, device=self.device, fuse=True)
                model = AutoShape(model)

            self.model_instances.append(model)


    def __call__(self, cv_image: cv2.Mat):
        assert len(self.model_instances) == len(self.weight_path_list) == len(self.ensemble_weight_list),  \
                "Model weight number must equal to model instance"
        ensemble_boxes_list = []
        ensemble_scores_list = []
        ensemble_labels_list = []

        for model, weight_name in zip(self.model_instances, self.weight_path_list):
            if "yolov8" in weight_name:
                results = model.predict(cv_image)  # conf=0.8, iou=0.8 / conf=0.75, iou=0.5
                boxes = results[0].boxes.xyxyn.tolist()
                classes = results[0].boxes.cls.tolist()
                confidences = results[0].boxes.conf.tolist()
                ensemble_boxes_list.append(boxes)
                ensemble_scores_list.append(confidences)
                ensemble_labels_list.append([0 for _ in classes])  # only single class
            else:
                results = model(cv_image)
                boxes_list = []
                scores_list = []
                labels_list = []
                for i, det in enumerate(results.pred[0]):
                    label, confidence, bbox = det[5], det[4], det[:4]
                    if confidence < 0.05:
                        continue
                    x1, y1, x2, y2 = map(int, bbox)
                    boxes_list.append([x1/1280, y1/720, x2/1280, y2/720])
                    scores_list.append(confidence.cpu().item())
                    labels_list.append(0)  # only has i class
                ensemble_boxes_list.append(boxes_list)
                ensemble_scores_list.append(scores_list)
                ensemble_labels_list.append(labels_list)

        # ensemble model results
        boxes, scores, labels = weighted_boxes_fusion(
            ensemble_boxes_list, ensemble_scores_list, ensemble_labels_list, weights=self.ensemble_weight_list,
            iou_thr=0.5, skip_box_thr=0.05
        )
        boxes = np.round(boxes * np.array([1280, 720, 1280, 720]))
        return np.concatenate((boxes, scores.reshape(-1, 1), labels.reshape(-1, 1)), axis=1)
