import numpy as np


def is_within_frame(bbox, width, height):

    x1, y1, x2, y2 = bbox
    return x1 >= 0 and x2 <= width and y1 >= 0 and y2 <= height


class Track:
    def __init__(self, track_id, label, bbox, feature, color):
        self.track_id = track_id
        self.label = label
        self.bbox = bbox
        self.feature = feature
        self.velocity = np.array([0, 0])  # Initial velocity (x velocity, y velocity)
        self.color = color
        self.unmatched_count = 0
        self.active = True  # Track is currently being updated

    def predict_next_position(self):
        # Simple linear motion model
        x1, y1, x2, y2 = self.bbox
        dx, dy = self.velocity
        return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    def update(self, bbox, feature):
        # Update position and calculate new velocity
        x1_old, y1_old, x2_old, y2_old = self.bbox
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.velocity = np.array([(x1 + x2 - x1_old - x2_old) / 2, (y1 + y2 - y1_old - y2_old) / 2])
        self.feature = feature
        self.unmatched_count = 0

    def increment_unmatched(self):
        self.unmatched_count += 1
        predicted_bbox = self.predict_next_position()
        if not is_within_frame(predicted_bbox, 1280, 720):
            self.active = False

