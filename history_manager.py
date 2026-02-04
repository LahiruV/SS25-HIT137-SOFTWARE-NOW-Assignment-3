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
    def __init__(self, max_states: int = 30):
        """
        Initialise history manager.

        Args:
            max_states: Maximum number of undo states to retain.
        """
        self.max_states = max_states
        self._undo: List[EditorState] = []
        self._redo: List[EditorState] = []
     
    def clear(self) -> None:
        """Clear all undo and redo history."""
        self._undo.clear()
        self._redo.clear()

    def can_undo(self) -> bool:
        """Return True if an undo operation is possible."""
        return len(self._undo) > 0
