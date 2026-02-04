from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np



@dataclass
class EditorState:
    """
    Immutable snapshot of editor state for undo and redo.

    Stores both image data and UI values so visual controls
    can be restored exactly when navigating history.
    """
    image_bgr: np.ndarray
    file_path: Optional[str]
    ui: Dict[str, Any] 

class HistoryManager:
    """
    Manages undo and redo stacks for the image editor.

    undo: past states
    redo: states undone and available to restore
    """
