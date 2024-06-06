import os
import sys

sys.path.append("Detector/yolov9")

import yaml
from boxmot import OCSORT, DeepOCSORT
import cv2
import torch
import numpy as np
from glob import glob
from datetime import datetime

from Detector.yolov9.models.common import DetectMultiBackend, AutoShape
from Extractor.model.feature_extractor import ReIDModel, match_features
from Tracker.tacker import Tracklet

from common.plot_boxes import get_random_color, plot_box_on_img
from common.txt_writer import MOT_TXT, write_txt_by_line


with open('inference_develop.yaml', 'r') as file:
    config = yaml.safe_load(file)

config_default = config['Default']
config_detector = config['Detector']
config_extractor = config['Extractor']
config_tracker = config['Tracker']

########################################################################################################################
# Initialize folder setting
FRAME_FOLDER = config_default['FRAME_FOLDER']
RESULT_FOLDER = config_default['RESULT_FOLDER']
os.makedirs(RESULT_FOLDER, exist_ok=True)
EXP_FOLDER = os.path.join(RESULT_FOLDER, datetime.now().strftime('%Y%m%d%H%M%S'))

TEMP_CROP_FOLDER = os.path.join(EXP_FOLDER, 'crop_results')
AICUP_CSV_FOLDER = os.path.join(EXP_FOLDER, 'submit_csv_results')
os.makedirs(TEMP_CROP_FOLDER, exist_ok=True)
os.makedirs(AICUP_CSV_FOLDER, exist_ok=True)

SELECT_DATE = config_default['SELECT_DATE']
SELECT_TIME = config_default['SELECT_TIME']
SELECT_CAMERA = config_default['SELECT_CAMERA']

if config_default['SAVE_OUT_VIDEO']:
    VIDEO_FOLDER = os.path.join(EXP_FOLDER, 'video_results')
    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    FPS = config_default['SAVE_OUT_VIDEO_FPS']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(f"{VIDEO_FOLDER}/{SELECT_DATE}_{SELECT_TIME}_{SELECT_CAMERA}.mp4", fourcc, fps=FPS, frameSize=(1280, 720))

if config_default['WRITE_MOT_TXT']:
    MOT_TXT_FOLDER = os.path.join(EXP_FOLDER, 'mot_txt_results')
    os.makedirs(MOT_TXT_FOLDER, exist_ok=True)
    txt_file = os.path.join(MOT_TXT_FOLDER, f"{SELECT_DATE}_{SELECT_TIME}_{SELECT_CAMERA}.txt")

########################################################################################################################
# Initialize Detector, Extractor
# [Detector] Initialize YOLO model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights=config_detector['DETECTOR_WEIGHT'], device=device, fuse=True)
model = AutoShape(model)

# [Feature Extractor]
extractor = ReIDModel(trained_weight=config_extractor['EXTRACTOR_WEIGHT'], model_type=config_extractor['EXTRACTOR_TYPE'])

########################################################################################################################
# Prepare inference frames
frames = glob(os.path.join(FRAME_FOLDER, "*.jpg")) + glob(os.path.join(FRAME_FOLDER, "*.png"))

for image_path in list(frames):  # list(frames): copy of frames
    image_name_ = image_path.split("\\")[-1]
    date, time_start, time_finish, camera_id, _ = image_name_.split("_")
    if not (int(date) == SELECT_DATE and int(time_start) == SELECT_TIME and (SELECT_CAMERA == 'all' or int(camera_id) == SELECT_CAMERA)):
        frames.remove(image_path)

########################################################################################################################
# Start Tracking
tracks = []
next_track_id = 0
GLOBAL_FRAME_ID = 1

# Initial frame (frame 0)
frame_current = cv2.imread(frames[0])
img_name = os.path.basename(frames[0])
results = model(frame_current)
detection_prev_label = []
detection_prev_image = []

for i, det in enumerate(results.pred[0]):
    label, confidence, bbox = det[5], det[4], det[:4]
    if confidence < config_detector['DETECTOR_CONFIDENCE']: continue
    x1, y1, x2, y2 = map(int, bbox)
    class_id = int(label)
    detection_prev_label.append([x1, y1, x2, y2, confidence.cpu().item(), class_id])

    cropped_image = frame_current[y1:y2, x1:x2, :]
    save_path = os.path.join(TEMP_CROP_FOLDER, f"{img_name[:-4]}_det_{i:04}.png")
    cv2.imwrite(save_path, cropped_image)
    detection_prev_image.append(save_path)

previous_feature = extractor.get_features(detection_prev_image) if len(detection_prev_label) != 0 else np.empty((0, 512))
previous_detect = np.array(detection_prev_label) if len(detection_prev_label) != 0 else np.empty((0, 6))

for detect, feature in zip(previous_detect, previous_feature):
    current_box = detect[:4]
    init_track = Tracklet(next_track_id, detect[-1], current_box, feature, get_random_color())
    tracks.append(init_track)
    frame_current = plot_box_on_img(frame_current, current_box, init_track.track_id, init_track.color)
    next_track_id += 1
    cv2.imshow("Test tracking", frame_current)

    if config_default['WRITE_MOT_TXT']:
        x1, y1, x2, y2 = current_box[0], current_box[1], current_box[2], current_box[3]
        mot_txt_line = MOT_TXT(GLOBAL_FRAME_ID, init_track.track_id, x1, y1, x2, y2, detect[4])
        write_txt_by_line(txt_file, mot_txt_line)

# Loop other frames (frame 1 to end)
for frame_path in frames[1:]:
    # Current frame (frame n)
    GLOBAL_FRAME_ID += 1
    frame_current = cv2.imread(frame_path)
    results = model(frame_current)
    detection_current_label = []
    detection_current_image = []
    for i, det in enumerate(results.pred[0]):
        label, confidence, bbox = det[5], det[4], det[:4]
        if confidence < config_detector['DETECTOR_CONFIDENCE']: continue
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)
        detection_current_label.append([x1, y1, x2, y2, confidence.cpu().item(), class_id])

        cropped_image = frame_current[y1:y2, x1:x2, :]
        save_path = os.path.join(TEMP_CROP_FOLDER, f"{os.path.basename(frame_path)[:-4]}_det_{i:04}.png")
        cv2.imwrite(save_path, cropped_image)
        detection_current_image.append(save_path)

    current_features = extractor.get_features(detection_current_image) if len(detection_current_label) != 0 else np.empty((0, 512))
    current_detects = np.array(detection_current_label) if len(detection_current_label) != 0 else np.empty((0, 6))

    # Previous frame (frame n-1)
    previous_features = np.array([track.feature for track in tracks])

    if len(previous_features) != 0 and len(current_features) != 0:
        matched_indices, similarity_matrix = match_features(previous_features, current_features, config_extractor['EXTRACTOR_THRESHOLD'])

        # Update tracks and add new tracks for unmatched detections
        used_indices = set()
        for prev_idx, curr_idx in matched_indices:
            bbox = current_detects[curr_idx][:4]
            tracks[prev_idx].update(bbox=bbox, feature=current_features[curr_idx])
            used_indices.add(curr_idx)
            frame_current = plot_box_on_img(frame_current, bbox, tracks[prev_idx].track_id, tracks[prev_idx].color)

            if config_default['WRITE_MOT_TXT']:
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                mot_txt_line = MOT_TXT(GLOBAL_FRAME_ID, tracks[prev_idx].track_id, x1, y1, x2, y2, current_detects[curr_idx][4])
                write_txt_by_line(txt_file, mot_txt_line)


        for i, feature in enumerate(current_features):
            if i not in used_indices:
                # Initialize new track for unmatched detections
                current_box = current_detects[i][:4]
                new_track = Tracklet(next_track_id, current_detects[i][-1], current_box, feature, get_random_color())
                tracks.append(new_track)
                frame_current = plot_box_on_img(frame_current, current_box, new_track.track_id, new_track.color)
                next_track_id += 1

                if config_default['WRITE_MOT_TXT']:
                    x1, y1, x2, y2 = current_box[0], current_box[1], current_box[2], current_box[3]
                    mot_txt_line = MOT_TXT(GLOBAL_FRAME_ID, new_track.track_id, x1, y1, x2, y2, current_detects[i][4])
                    write_txt_by_line(txt_file, mot_txt_line)

        # Optionally deactivate unmatched tracks
        matched_prev_indices = {prev_idx for prev_idx, curr_idx in matched_indices}

        to_remove = []
        for i, track in enumerate(tracks):
            if track.active:
                if i not in matched_prev_indices:
                    track.increment_unmatched()
                    if track.unmatched_count > config_tracker['TRACKER_MAX_UNMATCH_FRAME']:
                        track.active = False
                        to_remove.append(i)  # Schedule removal of the track
            else:
                to_remove.append(i)  # Already inactive tracks also scheduled for removal

        # Remove inactive tracks
        for index in sorted(to_remove, reverse=True):
            del tracks[index]

    elif len(current_features) != 0 and len(previous_feature) == 0:
        for detect, feature in zip(current_detects, current_features):
            current_box = detect[:4]
            init_track = Tracklet(next_track_id, detect[-1], current_box, feature, get_random_color())
            tracks.append(init_track)
            frame_current = plot_box_on_img(frame_current, current_box, init_track.track_id, init_track.color)
            next_track_id += 1
            cv2.imshow("Test tracking", frame_current)

            if config_default['WRITE_MOT_TXT']:
                x1, y1, x2, y2 = current_box[0], current_box[1], current_box[2], current_box[3]
                mot_txt_line = MOT_TXT(GLOBAL_FRAME_ID, init_track.track_id, x1, y1, x2, y2, detect[4])
                write_txt_by_line(txt_file, mot_txt_line)

    elif len(current_features) == 0 and len(previous_feature) != 0:
        tracks = []  # reset tracks
        next_track_id += 1

    # else:
    #     tracks = []  # reset tracks

    cv2.imshow("Test tracking", frame_current)

    if config_default['SAVE_OUT_VIDEO']:
        writer.write(frame_current)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if config_default['SAVE_OUT_VIDEO']:
    writer.release()

cv2.destroyAllWindows()

