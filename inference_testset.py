import os
import shutil
import sys
sys.path.append("Detector/yolov9")

import yaml
import cv2
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from datetime import datetime

from Detector.yolov9.models.common import DetectMultiBackend, AutoShape
from Extractor.model.feature_extractor import ReIDModel, match_features
from Tracker.tacker import Track

from common.plot_boxes import get_random_color, plot_box_on_img
from common.txt_writer import MOT_TXT, write_txt_by_line


with open('inference_testset.yaml', 'r') as file:
    config = yaml.safe_load(file)

config_default = config['Default']
config_detector = config['Detector']
config_extractor = config['Extractor']
config_tracker = config['Tracker']

########################################################################################################################
# Initialize folder settings
FRAME_FOLDER = config_default['FRAME_FOLDER']
RESULT_FOLDER = config_default['RESULT_FOLDER']
os.makedirs(RESULT_FOLDER, exist_ok=True)
EXP_FOLDER = os.path.join(RESULT_FOLDER, datetime.now().strftime('%Y%m%d%H%M%S'))
YAML_LOG_FOLDER = os.path.join(EXP_FOLDER, 'yaml_log_results')
os.makedirs(YAML_LOG_FOLDER, exist_ok=True)

# Record experiment settings
current_file_path = os.path.abspath(__file__)
with open(current_file_path, 'r', encoding='utf-8') as f:
    current_file_content = f.read()
with open(os.path.join(YAML_LOG_FOLDER, "codes_record.py"), 'w', encoding='utf-8') as f:
    f.write(current_file_content)
if os.path.exists("inference_testset.yaml"):
    shutil.copyfile(src="inference_testset.yaml", dst=os.path.join(YAML_LOG_FOLDER, "param_record.yaml"))

########################################################################################################################
# Build Detector, Extractor model architecture and load weights

# [Detector] Initialize YOLO model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if config_detector['ENSEMBLE']:  # ensemble model
    from Detector.ensemble_detector import DetectMultiBackendEnsemble
    weight_list = config_detector['ENSEMBLE_MODEL_LIST']
    weight_path_list = config_detector['ENSEMBLE_WEIGHT_LIST']
    model = DetectMultiBackendEnsemble(weights_list=weight_list, ensemble_weight_list=weight_path_list, device=device)
else:  # single yolov9 model
    model = DetectMultiBackend(weights=config_detector['DETECTOR_WEIGHT'], device=device, fuse=True)
    model = AutoShape(model)

# [Feature Extractor]
extractor = ReIDModel(trained_weight=config_extractor['EXTRACTOR_WEIGHT'], model_type=config_extractor['EXTRACTOR_TYPE'])

########################################################################################################################
# Start inference test set
track_id = 0
folder_list = os.listdir(FRAME_FOLDER)
for i, DATE_TIME in enumerate(tqdm(folder_list)):
    TEMP_CROP_FOLDER = os.path.join(EXP_FOLDER, 'crop_results')
    os.makedirs(TEMP_CROP_FOLDER, exist_ok=True)

    frames = glob(os.path.join(FRAME_FOLDER, DATE_TIME, "*.jpg")) + glob(os.path.join(FRAME_FOLDER, DATE_TIME, "*.png"))
    if config_default['WRITE_MOT_TXT']:
        MOT_TXT_FOLDER = os.path.join(EXP_FOLDER, 'mot_txt_results')
        os.makedirs(MOT_TXT_FOLDER, exist_ok=True)
        txt_file = os.path.join(MOT_TXT_FOLDER, f"{DATE_TIME}.txt")
    if config_default['SAVE_OUT_VIDEO']:
        VIDEO_FOLDER = os.path.join(EXP_FOLDER, 'video_results')
        os.makedirs(VIDEO_FOLDER, exist_ok=True)
        FPS = config_default['SAVE_OUT_VIDEO_FPS']
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(f"{VIDEO_FOLDER}/{DATE_TIME}.mp4", fourcc, fps=FPS, frameSize=(1280, 720))

    # Start Tracking
    # Initialize global variables and configurations
    frame_id = 1
    previous_camera_id = None
    for frame_index, frame_path in enumerate(frames):
        frame_current = cv2.imread(frame_path)
        results = model(frame_current)

        frame_name = os.path.basename(frame_path)
        camera_id, _ = frame_name.split("_")

        if not config_detector['ENSEMBLE']:
            results = results.pred[0]

        detection_label = []
        detection_image = []
        for i, det in enumerate(results):
            label, confidence, bbox = det[5], det[4], det[:4]
            if confidence < config_detector['DETECTOR_CONFIDENCE']:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)

            if not config_detector['ENSEMBLE']:
                detection_label.append([x1, y1, x2, y2, confidence.cpu().item(), class_id])
            else:
                detection_label.append([x1, y1, x2, y2, confidence, class_id])

            cropped_image = frame_current[y1:y2, x1:x2, :]
            save_path = os.path.join(TEMP_CROP_FOLDER, f"{os.path.basename(frame_path)[:-4]}_det_{i:04}.png")
            cv2.imwrite(save_path, cropped_image)
            detection_image.append(save_path)

        current_features = extractor.get_features(detection_image) if len(detection_label) != 0 else np.empty((0, 512))
        current_detects = np.array(detection_label) if len(detection_label) != 0 else np.empty((0, 6))

        if frame_index == 0 or (previous_camera_id is not None and camera_id != previous_camera_id):  # Reset track objects
            tracks = []
            track_id += 1
            for detect, feature in zip(current_detects, current_features):
                current_box = detect[:4]
                init_track = Track(track_id, detect[-1], current_box, feature, get_random_color())
                tracks.append(init_track)
                frame_current = plot_box_on_img(frame_current, current_box, init_track.track_id, init_track.color)
                track_id += 1
                cv2.imshow("Test tracking", frame_current)

                if config_default['WRITE_MOT_TXT']:
                    x1, y1, x2, y2 = current_box[0], current_box[1], current_box[2], current_box[3]
                    mot_txt_line = MOT_TXT(frame_id, init_track.track_id, x1, y1, x2, y2, detect[4])
                    write_txt_by_line(txt_file, mot_txt_line)
        else:  # Subsequent frames logic
            previous_features = np.array([track.feature for track in tracks])

            if len(previous_features) != 0 and len(current_features) != 0:
                matched_indices, similarity_matrix = match_features(previous_features, current_features, config_extractor['EXTRACTOR_THRESHOLD'])
                used_indices = set()
                for prev_idx, curr_idx in matched_indices:
                    bbox = current_detects[curr_idx][:4]
                    tracks[prev_idx].update(bbox=bbox, feature=current_features[curr_idx], mode=config_tracker["TRACKER_MOTION_PREDICT"])
                    used_indices.add(curr_idx)
                    frame_current = plot_box_on_img(frame_current, bbox, tracks[prev_idx].track_id, tracks[prev_idx].color)

                    if config_default['WRITE_MOT_TXT']:
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                        mot_txt_line = MOT_TXT(frame_id, tracks[prev_idx].track_id, x1, y1, x2, y2, current_detects[curr_idx][4])
                        write_txt_by_line(txt_file, mot_txt_line)

                for i, feature in enumerate(current_features):
                    if i not in used_indices:
                        current_box = current_detects[i][:4]
                        new_track = Track(track_id, current_detects[i][-1], current_box, feature, get_random_color())
                        tracks.append(new_track)
                        frame_current = plot_box_on_img(frame_current, current_box, new_track.track_id, new_track.color)
                        track_id += 1

                        if config_default['WRITE_MOT_TXT']:
                            x1, y1, x2, y2 = current_box[0], current_box[1], current_box[2], current_box[3]
                            mot_txt_line = MOT_TXT(frame_id, new_track.track_id, x1, y1, x2, y2, current_detects[i][4])
                            write_txt_by_line(txt_file, mot_txt_line)

                matched_prev_indices = {prev_idx for prev_idx, curr_idx in matched_indices}
                to_remove = []
                for i, track in enumerate(tracks):
                    if track.active:
                        if i not in matched_prev_indices:
                            track.increment_unmatched(mode=config_tracker['TRACKER_MOTION_PREDICT'])
                            if track.unmatched_count > config_tracker['TRACKER_MAX_UNMATCH_FRAME']:
                                track.active = False
                                to_remove.append(i)
                    else:
                        to_remove.append(i)

                for index in sorted(to_remove, reverse=True):
                    del tracks[index]

            elif len(current_features) != 0 and len(previous_features) == 0:
                for detect, feature in zip(current_detects, current_features):
                    current_box = detect[:4]
                    init_track = Track(track_id, detect[-1], current_box, feature, get_random_color())
                    tracks.append(init_track)
                    frame_current = plot_box_on_img(frame_current, current_box, init_track.track_id, init_track.color)
                    track_id += 1

                    if config_default['WRITE_MOT_TXT']:
                        x1, y1, x2, y2 = current_box[0], current_box[1], current_box[2], current_box[3]
                        mot_txt_line = MOT_TXT(frame_id, init_track.track_id, x1, y1, x2, y2, detect[4])
                        write_txt_by_line(txt_file, mot_txt_line)

            elif len(current_features) == 0 and len(previous_features) != 0:
                tracks = []
                track_id += 1

        previous_camera_id = camera_id
        cv2.imshow("Test tracking", frame_current)

        if config_default['SAVE_OUT_VIDEO']:
            writer.write(frame_current)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    if os.path.exists(TEMP_CROP_FOLDER):
        shutil.rmtree(TEMP_CROP_FOLDER)

    cv2.destroyAllWindows()
    if config_default['SAVE_OUT_VIDEO']:
        writer.release()

print(f"All inference process is done! Results are saved in: \n{os.path.abspath(EXP_FOLDER)}")
