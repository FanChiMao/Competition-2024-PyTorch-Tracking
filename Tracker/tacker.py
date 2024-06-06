import numpy as np
from filterpy.kalman import KalmanFilter


def is_within_frame(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return x1 >= 0 and x2 <= width and y1 >= 0 and y2 <= height


def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=7, dim_z=4)
    kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 0, 1],
                     [0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0]])
    kf.P *= 1000.  # Initial uncertainty
    kf.R[2:, 2:] *= 10.  # Measurement noise
    kf.Q[-1, -1] *= 0.01  # Process noise
    kf.Q[4:, 4:] *= 0.01
    return kf

def convert_bbox_to_z(bbox):
    """
    Convert bounding box to measurement space.
        :bbox formulated as [x1, y1, x2, y2]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """Convert state vector to bounding box."""
    w = np.sqrt(x[2, 0] * x[3, 0])
    h = x[2, 0] / w
    bbox = [x[0, 0] - w / 2., x[1, 0] - h / 2., x[0, 0] + w / 2., x[1, 0] + h / 2.]
    return bbox


def are_directions_opposite(vector1, vector2, tolerance=0.5):
    # Calculate the dot product and magnitudes
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate the cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)

    # Directions are opposite if cosine similarity is less than or equal to the negative of the tolerance
    return cosine_similarity <= -tolerance

class Track:
    def __init__(self, track_id, label, bbox, feature, color):
        self.track_id = track_id
        self.label = label
        self.bbox = bbox
        self.feature = feature
        self.velocity = np.array([0, 0])  # Initial velocity (x velocity, y velocity)
        self.direction_vector  = None  # Initial direction
        self.previous_bbox = bbox
        self.color = color
        self.unmatched_count = 0
        self.kf = initialize_kalman_filter()
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.active = True  # Track is currently being updated

    def update_direction_vector(self, bbox):
        old_x_center = (self.previous_bbox[0] + self.previous_bbox[2]) / 2
        old_y_center = (self.previous_bbox[1] + self.previous_bbox[3]) / 2
        new_x_center = (bbox[0] + bbox[2]) / 2
        new_y_center = (bbox[1] + bbox[3]) / 2

        dx = new_x_center - old_x_center
        dy = new_y_center - old_y_center

        self.direction_vector = np.array([dx, dy])
        self.previous_bbox = bbox

    def update(self, bbox, feature, mode='lr'):
        self.update_direction_vector(bbox)
        if mode == 'lr':
            # Update position and calculate new velocity
            x1_old, y1_old, x2_old, y2_old = self.bbox
            self.bbox = bbox
            x1, y1, x2, y2 = bbox
            self.velocity = np.array([(x1 + x2 - x1_old - x2_old) / 2, (y1 + y2 - y1_old - y2_old) / 2])
        elif mode == 'kf':
            # Update position using Kalman Filter
            self.kf.update(convert_bbox_to_z(bbox))
            self.bbox = convert_x_to_bbox(self.kf.x)
        else:
            raise ValueError(f"Motion model only support \"lr\" (linear) or \"kf\" (Kalman Filter), get {mode}")
        self.feature = feature
        self.unmatched_count = 0

    def predict_next_position(self, mode='lr'):
        if mode == 'lr':
            # Simple linear motion model
            x1, y1, x2, y2 = self.bbox
            dx, dy = self.velocity
            return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        elif mode == "kf":
            # Kalman Filter motion model
            self.kf.predict()
            return convert_x_to_bbox(self.kf.x)
        else:
            raise ValueError(f"Motion model only support \"lr\" (linear) or \"kf\" (Kalman Filter), get {mode}")

    def increment_unmatched(self, mode):
        self.unmatched_count += 1
        predicted_bbox = self.predict_next_position(mode)
        if not is_within_frame(predicted_bbox, 1280, 720):
            self.active = False
