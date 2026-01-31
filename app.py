from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from image_model import ImageModel
from image_processor import ImageProcessor
from history_manager import HistoryManager, EditorState

class ImageEditorApp:
    """
    Uses ImageModel (data)
    Uses ImageProcessor (operations)
    Uses HistoryManager (undo/redo)
    """

    def __init__(self, root: tk.Tk):
        """Initialise the application UI, state managers and event bindings."""
        self.root = root
        self.root.title("Image Editor")
        self.root.geometry("1150x720")
        self.root.minsize(950, 620)

        # Core components
        self.model = ImageModel()
        self.history = HistoryManager(max_states=30)

        # Tk image holder
        self._tk_image: Optional[ImageTk.PhotoImage] = None

        # UI variables 
        self.blur_var = tk.IntVar(value=1)           
        self.brightness_var = tk.IntVar(value=0)    
        self.contrast_var = tk.IntVar(value=0)       
        self.scale_var = tk.IntVar(value=100)       

        # Slider preview control 
        self._preview_base_bgr: Optional[np.ndarray] = None
        self._preview_active: bool = False

        self.UI: Dict[str, str] = {}

        self._apply_theme()
        self._build_menu()
        self._build_layout()
        self._bind_shortcuts()
        self._update_status("Ready. Open an image to begin.")