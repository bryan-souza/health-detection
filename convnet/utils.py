import numpy as np

from pathlib import Path
from typing import Iterable, Tuple
from PIL import Image

SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def downscale_images(path: Path, out: Path, factor: int) -> None:
    """
    Downscales images on a given folder based on a factor,
    preserving its scale.

    Parameters:
        path: A folder path where the images reside
        out: A folder path where you wish to save the resized images
        factor: How many times you wish to downscale the image

    Example:
    >>>  from PIL import Image
    >>>  folder = './img'
    >>>  img_before = Image.open(f'{folder}/image.jpg')
    >>>  img_before.size # (1024, 1024)
    >>>  downscale_images(folder, folder, factor=4)
    >>>  img_after = Image.open(f'{folder}/image_256x256.jpg')
    >>>  img_after.size # (256, 256)
    """

    assert path.exists(), f'{path} does not exist on filesystem'
    assert path.is_dir(), f'{path} is not a directory'
    
    assert out.exists(), f'{out} does not exist on filesystem'
    assert out.is_dir(), f'{out} is not a directory'

    for file in path.iterdir():
        if file.suffix in SUPPORTED_EXTENSIONS:
            image = Image.open(file)
            new_image = image.reduce(factor)
            
            filename = f'{file.stem}_{new_image.width}x{new_image.height}.{file.suffix}'
            filepath = Path(out, filename)

            new_image.save(filepath)

def load_dataset(path: Path) -> Iterable[Tuple]:
    """
    Iterates through a given folder and yields tuples with
    numpy arrays and labels. Please note that only level 1
    subfolders are considered to label images; level 2+
    subfolders WILL NOT be used to yield dataset
    images/labels.

    Parameters:
        path: The root path where the subfolders reside

    Yields:
        A tuple containing an image array and its label

    Example:
    >>> # Consider the following structure for 'dataset':
    >>> # dataset
    >>> # - one
    >>> #   - img_001.jpg
    >>> #   - img_002.jpg
    >>> #   - ...
    >>> # - two
    >>> #   - img_100.jpg
    >>> #   - img_101.jpg
    >>> #   - ...
    >>> root_dir = './dataset'
    >>> for img_array, label in load_dataset(root_dir):
    >>>     print(img_array) # numpy.array([[[..., ..., ...]]], dtype=numpy.uint8)
    >>>     print(label) # 0
    """
    assert path.exists(), f'{path} does not exist on filesystem'
    assert path.is_dir(), f'{path} is not a directory'
    
    labels = [ item.stem for item in path.iterdir() if item.is_dir() ]

    for i, label in enumerate(labels):
        sub_path = Path(path, label)
        for sub_item in sub_path.iterdir():
            if sub_item.suffix in SUPPORTED_EXTENSIONS:
                image = Image.open(sub_item)
                yield np.asarray(image), i