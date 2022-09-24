import os
import random
import shutil
from typing import Tuple
import numpy as np

from pycocotools.coco import COCO
from pathlib import Path
from PIL import Image
from PIL.Image import Resampling


class ImgToolbox():
    HEALTHY_CATEGORIES = [1, 8]

    def __init__(self, root_path: str | None = None):
        # TODO: Docs
        self.ROOT_PATH = Path('.')
        if (root_path) is not None:
            self.ROOT_PATH = Path(root_path)

        self.DATASET_PATH = Path(self.ROOT_PATH, 'dataset')
        self.EXTRAS_PATH = Path(self.ROOT_PATH, 'extra')
        
        annotations_path = Path(self.EXTRAS_PATH, 'instances_default.json')
        if not annotations_path.exists():
            self.ANNOTATIONS = None
            print('[WARN] Annotations file could not be found. Some methods will not work')
            return None
        
        self.ANNOTATIONS = COCO(annotations_path)

    def create_dataset_structure(self):
        # TODO: Docs
        # Check if annotations file were loaded
        if self.ANNOTATIONS is None:
            raise RuntimeError('Annotations file not loaded.')

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
        for category_id in self.ANNOTATIONS.getCatIds():
            image_ids = self.ANNOTATIONS.catToImgs[category_id]
            images = self.ANNOTATIONS.loadImgs(image_ids)

            for image in images:
                image_path = Path(self.EXTRAS_PATH, image['file_name'])
                image_name = image_path.name

                if category_id in self.HEALTHY_CATEGORIES:
                    shutil.copyfile(src=image_path, dst=Path(self.DATASET_PATH, 'healthy', image_name))
                    continue

                shutil.copyfile(src=image_path, dst=Path(self.DATASET_PATH, 'unhealthy', image_name))    


    @staticmethod
    def resize_image(source: str, target: str, target_size=Tuple[int, int]):
        # TODO: Docs
        # Type enforcing
        source_path = Path(source)
        if not source_path.exists():
            raise ValueError('%s does not exist on filesystem' % source)
        
        target_path = Path(target)

        if not isinstance(target_size, tuple):
            raise TypeError('parameter "target_size" must be a tuple')

        if not len(target_size) == 2:
            raise ValueError('parameter "target_size" has length of %i, should be 2' % len(target_size))

        if not all([ isinstance(size, int) for size in target_size ]):
            raise ValueError('parameter "target_size" contents must be integers')

        
        for img in source_path.iterdir():
            im = Image.open(img)
            im.thumbnail(target_size, Resampling.LANCZOS)
            im.save(Path(target_path, img.name))

    @staticmethod
    def standardize_background(source: str, target: str):
        # TODO: Docs
        # Type enforcing
        source_path = Path(source)
        if not source_path.exists():
            raise ValueError('%s does not exist on filesystem' % source)
        
        target_path = Path(target)

        
        for img in source_path.iterdir():
            im = Image.open(img)

            data = np.array(im)   # "data" is a height x width x 4 numpy array
            red, green, blue = data.T # Temporarily unpack the bands for readability

            # Replace white with black...
            white_areas = (red >= 225) & (blue >= 225) & (green >= 225)    
            data[white_areas.T] = (0, 0, 0) # Transpose back needed

            im_conv = Image.fromarray(data)
            im_conv.save( Path(target_path, img.name) )

    @staticmethod
    def purge_elements(source: str, target_amount: int):
        # TODO: Docs
        # Type enforcing
        source_path = Path(source)
        if not source_path.exists():
            raise ValueError('%s does not exist on filesystem' % source)

        if not isinstance(target_amount, int):
            raise TypeError('parameter "target_amount" must be an integer')

        if not target_amount > 0:
            raise ValueError('parameter "target_amount" is %i, should be at least 1' % target_amount)
        

        # Pick images to be purged (chosen randomly)
        images = [ img for img in source_path.iterdir() ]
        purged_images = []
        while len(purged_images) <= (len(images) - target_amount):
            choice = random.choice(images)
            if not choice in purged_images:
                purged_images.append(choice)

        # Delete images
        for img in purged_images:
            os.remove( img )
