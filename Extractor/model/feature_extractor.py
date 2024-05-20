from torchreid.reid.utils import FeatureExtractor


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
