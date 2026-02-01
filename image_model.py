from __future__ import annotations
from typing import Optional, Tuple
import os
import cv2
import numpy as np


class ImageModel:
    """
    Stores and manages image state for the editor.

    Load images from disk
    Preserve original image for reset
    Provide safe access to the current working image
    Images are stored internally in BGR format (OpenCV default)
    All getters return copies to preserve encapsulation
    """
