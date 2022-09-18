import os
import shutil
import numpy as np

from pycocotools.coco import COCO
from pathlib import Path
from PIL import Image


class ImgToolbox():
    HEALTHY_CATEGORIES = [1, 8]

    def __init__(self, root_path: str | None = None):
        self.ROOT_PATH = Path('..')
        if (root_path) is not None:
            self.ROOT_PATH = Path(root_path)

        self.DATASET_PATH = Path(self.ROOT_PATH, 'dataset')
        self.EXTRAS_PATH = Path(self.ROOT_PATH, 'extra')

    def create_dataset_structure(self):
        instances_path = Path(self.EXTRAS_PATH, 'instances_default.json')
        if not instances_path.exists():
            raise FileNotFoundError('"extra/instances_default.json" could not be found')

        # Load annotations
        dataset = COCO(instances_path)

        # Create file structure
        class_names = ['healthy', 'unhealthy']
        
        # Create dataset root folder if not exists
        if not self.DATASET_PATH.exists():
            os.mkdir( self.DATASET_PATH.absolute() )

        for _class in class_names:
            CLASS_PATH = Path(self.DATASET_PATH, _class)

            # Create feature folders if not exists
            if not CLASS_PATH.exists():
                os.mkdir( CLASS_PATH.absolute() )

        # Move files to folders
        for category_id in dataset.getCatIds():
            image_ids = dataset.catToImgs[category_id]
            images = dataset.loadImgs(image_ids)

            for image in images:
                image_path = Path(self.EXTRAS_PATH, image['file_name'])
                image_name = image_path.name

                if category_id in self.HEALTHY_CATEGORIES:
                    shutil.copyfile(src=image_path, dst=Path(self.DATASET_PATH, 'healthy', image_name))
                    continue

                shutil.copyfile(src=image_path, dst=Path(self.DATASET_PATH, 'unhealthy', image_name))

    @staticmethod
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
