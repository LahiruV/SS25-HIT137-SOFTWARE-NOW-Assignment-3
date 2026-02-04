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

    @staticmethod
    def grayscale(img_bgr: np.ndarray) -> np.ndarray:
        """
        Convert an image to grayscale and return as 3-channel BGR.

        Keeping BGR output simplifies GUI rendering and avoids branching
        in code that assumes 3 channels.
        """
        ImageProcessor._require(img_bgr)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def blur(img_bgr: np.ndarray, intensity: int) -> np.ndarray:
        """
        Apply Gaussian blur using an intensity-based kernel size.

        Args:
            img_bgr: Source BGR image.
            intensity: Intended kernel size.

        Returns:
            Blurred BGR image.
        """
        ImageProcessor._require(img_bgr)
        if intensity < 1:
            intensity = 1
        if intensity > 31:
            intensity = 31
            
        # GaussianBlur requires an odd kernel size; bump to next odd if needed.
        k = intensity if intensity % 2 == 1 else intensity + 1
        return cv2.GaussianBlur(img_bgr, (k, k), 0)

    @staticmethod
    def edge_detect(img_bgr: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
        """
        Perform Canny edge detection and return as 3 channel BGR.

        Args:
            img_bgr: Source BGR image.
            low: Lower threshold for hysteresis.
            high: Upper threshold for hysteresis.

        Returns:
            Edge map converted back to BGR for consistent display.
        """
        ImageProcessor._require(img_bgr)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low, high)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)