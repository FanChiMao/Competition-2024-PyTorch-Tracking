import torchreid

import os
import os.path as osp
from glob import glob
from tqdm import tqdm

from torchreid.reid.data import ImageDataset
from sklearn.model_selection import train_test_split


class AICUPDataset(ImageDataset):
    dataset_dir = 'vehicle-reid'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        train_set, valid_set = self.process_dir(self.root)
        train = train_set + valid_set
        query = train_set + valid_set
        gallery = train_set + valid_set

        super(AICUPDataset, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, trainval_ratio=0.8):
        image_path = os.path.join(dir_path, "*", "*.jpg")

        updated_tracker_id_dict = dict()
        updated_tracker_id = 0
        data = []

        for path in tqdm(glob(image_path), desc="process training data"):
            parts = path.split('\\')
            tracker_id = int(parts[-2])

            if tracker_id not in updated_tracker_id_dict:
                updated_tracker_id_dict[tracker_id] = updated_tracker_id
                updated_tracker_id += 1
            stem = os.path.basename(path)[:-4]
            camera_id, frame_id = map(int, stem.split("_"))
            data.append((path, updated_tracker_id_dict[tracker_id], camera_id - 1))
        train_data, valid_data = train_test_split(data, test_size=(1 - trainval_ratio), random_state=42)

        return train_data, valid_data


if __name__ == '__main__':
    import yaml
    import os
    current_file = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_file, 'extractor.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    torchreid.data.register_image_dataset('vehicle-reid', AICUPDataset)

    datamanager = torchreid.data.ImageDataManager(
        root=config['path'],
        sources=['vehicle-reid'],
        height=config['image_size'],
        width=config['image_size'],
        batch_size_train=config['train_batch'],
        batch_size_test=config['valid_batch'],
        transforms=["random_flip"]
    )

    model = torchreid.models.build_model(
        name=config["model_name"],
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True
    )

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim=config['optim'],
        lr=config['lr']
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=20
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    engine.run(
        save_dir=os.path.join(current_file, config['save_path']),
        max_epoch=config['epochs'],
        eval_freq=10,
        print_freq=10,
        test_only=False
    )
