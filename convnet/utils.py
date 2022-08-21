from pathlib import Path
from PIL import Image

SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def downscale_images(path: str | Path, out: str | Path, factor: int | None = 2):
    """
    Downscales images on a given folder based on a factor, preserving its scale.

    Parameters:
        path: A folder path where the images reside
        out: A folder path where you wish to save the resized images
        factor: How many times you wish to downscale the image

    Example:
    >>>  from PIL import Image
    >>>  folder = './img'
    >>>  img_before = Image.open(f'{folder}/image.jpg')
    >>>  img_before.size # Returns (1024, 1024)
    >>>  downscale_images(folder, folder, factor=4)
    >>>  img_after = Image.open(f'{folder}/image_256x256.jpg')
    >>>  img_after.size # Returns (256, 256)
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
