import cv2
import random
import numpy as np
random.seed(1234)

def get_random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

def plot_box_on_img(image, box, track_id, color):
    thickness = 2
    fontscale = 0.5

    w, h = cv2.getTextSize(f"id: {track_id}", 0, fontScale=fontscale, thickness=thickness)[0]  # text width, height
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness, cv2.LINE_AA)
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled

    brightness = np.mean(color)
    threshold = 128  # You can adjust this threshold based on your specific needs
    text_color = (0, 0, 0) if brightness > threshold else (255, 255, 255)

    img = cv2.putText(
        image, f'id: {int(track_id)}', (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
        0, fontscale, text_color, 1, lineType=cv2.LINE_AA
    )
    return img
