import math
from PIL import Image
import numpy as np
import os


# YOLOv3-608 has been trained with these 80 categories from COCO:
# Lin, Tsung-Yi, et al. "Microsoft COCO: Common Objects in Context."
# European Conference on Computer Vision. Springer, Cham, 2014.

class PreprocessYOLO(object):
    """A simple class for loading images with PIL and reshaping them to the specified
    input resolution for YOLOv3-608.
    """

    def __init__(self, yolo_input_resolution):
        """Initialize with the input resolution for YOLOv3, which will stay fixed in this sample.

        Keyword arguments:
        yolo_input_resolution -- two-dimensional tuple with the target network's (spatial)
        input resolution in HW order
        """
        self.yolo_input_resolution = yolo_input_resolution

    def process(self, input_image_path):
        """Load an image from the specified input path,
        and return it together with a pre-processed version required for feeding it into a
        YOLOv3 network.

        Keyword arguments:
        input_image_path -- string path of the image to be loaded
        """
        image_raw, image_resized = self._load_and_resize(input_image_path)
        image_preprocessed = self._shuffle_and_normalize(image_resized)
        return image_raw, image_preprocessed

    def _load_and_resize(self, input_image_path):
        """Load an image from the specified path and resize it to the input resolution.
        Return the input image before resizing as a PIL Image (required for visualization),
        and the resized image as a NumPy float array.

        Keyword arguments:
        input_image_path -- string path of the image to be loaded
        """

        image_raw = Image.open(input_image_path)
        # Expecting yolo_input_resolution in (height, width) format, adjusting to PIL
        # convention (width, height) in PIL:
        new_resolution = (
            self.yolo_input_resolution[1],
            self.yolo_input_resolution[0])
        image_resized = image_raw.resize(
            new_resolution, resample=Image.BICUBIC)
        image_resized = np.array(image_resized, dtype=np.float32, order='C')
        return image_raw, image_resized

    def _shuffle_and_normalize(self, image):
        """Normalize a NumPy array representing an image to the range [0, 1], and
        convert it from HWC format ("channels last") to NCHW format ("channels first"
        with leading batch dimension).

        Keyword arguments:
        image -- image as three-dimensional NumPy float array, in HWC format
        """
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.array(image, dtype=np.float32, order='C')
        return image