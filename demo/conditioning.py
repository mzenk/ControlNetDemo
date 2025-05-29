import imageio
import numpy as np

from annotator.canny import CannyDetector
from annotator.util import HWC3, resize_image


def get_canny_conditioning(
    image_path: str,
    low_threshold: int = 100,
    high_threshold: int = 200,
    image_resolution: int = 512,
) -> np.ndarray:
    """
    Get Canny edge detection conditioning from an input image.

    Args:
        image_path (str): Input image file path.
        low_threshold (int): Low threshold for Canny edge detection.
        high_threshold (int): High threshold for Canny edge detection.
        image_resolution (int): Desired resolution for the output image.

    Returns:
        np.ndarray: Processed Canny edge map.
    """
    img = imageio.imread(image_path)
    img = resize_image(HWC3(img), image_resolution)
    apply_canny = CannyDetector()
    canny_map = apply_canny(img, low_threshold, high_threshold)
    canny_map = HWC3(canny_map)
    return canny_map
