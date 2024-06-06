from torchreid.reid.utils import FeatureExtractor
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class ReIDModel(object):
    def __init__(self, trained_weight=None, model_type=None):
        super(ReIDModel, self).__init__()
        self.trained_weight = trained_weight
        self.model_type = model_type
        self.extractor = None
        self._set_extractor()

    def _set_extractor(self):
        if self.extractor is None:
            # Feature extraction model
            self.extractor = FeatureExtractor(
                model_name=self.model_type,
                model_path=self.trained_weight,
                device='cuda'
            )

    def get_features(self, image_path_list):
        feature_results = self.extractor(image_path_list)
        flattened_features = feature_results.cpu().numpy()
        return flattened_features


def match_features(previous_features, current_features, direction_matrix, threshold=0.5, large_value=1e10):
    # Normalize the feature vectors to compute cosine similarity
    norm_previous = np.linalg.norm(previous_features, axis=1, keepdims=True)
    norm_current = np.linalg.norm(current_features, axis=1, keepdims=True)
    previous_features_normalized = previous_features / norm_previous
    current_features_normalized = current_features / norm_current

    similarity_matrix = 1 - cdist(previous_features_normalized, current_features_normalized, 'cosine')
    cost_matrix = 1 - similarity_matrix

    # Apply direction constraints
    cost_matrix[~direction_matrix] = large_value

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_indices = [(row, col) for row, col in zip(row_ind, col_ind) if similarity_matrix[row, col] >= threshold]

    return matched_indices, similarity_matrix
