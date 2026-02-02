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
 SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self):
        """
        Initialise an empty image model with no loaded image.
        """
        self._original_bgr: Optional[np.ndarray] = None
        self._current_bgr: Optional[np.ndarray] = None
        self._file_path: Optional[str] = None

 def has_image(self) -> bool:
        """
        Check whether an image is currently loaded.
        """
        return self._current_bgr is not None

 def filename(self) -> str:
        """
        Return the image filename for display purposes.
        """
        return os.path.basename(self._file_path) if self._file_path else "Untitled"

 def current(self) -> Optional[np.ndarray]:
        """
        Get a copy of the current working image.

        A copy is returned to prevent accidental modification
        of internal state by other components.
        """
        return None if self._current_bgr is None else self._current_bgr.copy()

