from __future__ import annotations
from typing import Literal
import cv2
import numpy as np

class ImageProcessor:
    """Pure OpenCV operations"""

    @staticmethod
    def _require(img: np.ndarray) -> None:
        """
        Validate that an image is present and correctly typed.

        Args:
            img: Expected OpenCV image.

        Raises:
            ValueError: If `img` is missing or not a numpy array.
        """
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("No image loaded.")