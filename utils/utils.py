import os
import shutil
import numpy as np

from pycocotools.coco import COCO
from pathlib import Path
from PIL import Image


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

def standardize_background(source: Path, target: Path):
    for img in source.iterdir():
        im = Image.open(img)

        data = np.array(im)   # "data" is a height x width x 4 numpy array
        red, green, blue = data.T # Temporarily unpack the bands for readability

        # Replace white with black...
        white_areas = (red >= 225) & (blue >= 225) & (green >= 225)    
        data[white_areas.T] = (0, 0, 0) # Transpose back needed

        im_conv = Image.fromarray(data)
        im_conv.save(Path(target, img.name))
