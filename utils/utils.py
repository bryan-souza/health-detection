import os
import shutil
from pycocotools.coco import COCO
from pathlib import Path


EXTRAS_PATH = Path('.', 'extra')
HEALTHY_CATEGORIES = [1, 8]

def create_dataset_structure():
    instances_path = Path(EXTRAS_PATH, 'instances_default.json')
    if not instances_path.exists():
        raise FileNotFoundError('%s could not be found' % 'extra/instances_default.json')

    # Load annotations
    dataset = COCO(instances_path)

    # Create file structure
    class_names = ['healthy', 'unhealthy']
    ROOT_DIR = Path('.', 'dataset')

    if not ROOT_DIR.exists():
        os.mkdir( ROOT_DIR.absolute() )

    for _class in class_names:
        CLASS_PATH = Path(ROOT_DIR, _class)
        if not CLASS_PATH.exists():
            os.mkdir( CLASS_PATH.absolute() )

    # Move files to folders
    for category_id in dataset.getCatIds():
        image_ids = dataset.catToImgs[category_id]
        images = dataset.loadImgs(image_ids)

        for image in images:
            image_path = Path(EXTRAS_PATH, image['file_name'])
            image_name = image_path.name

            if category_id in HEALTHY_CATEGORIES:
                shutil.copyfile(src=image_path, dst=Path(ROOT_DIR, 'healthy', image_name))
                continue

            shutil.copyfile(src=image_path, dst=Path(ROOT_DIR, 'unhealthy', image_name))

if __name__ == '__main__':
    create_dataset_structure()