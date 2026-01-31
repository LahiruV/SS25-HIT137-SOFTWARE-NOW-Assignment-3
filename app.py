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
        
    def _apply_theme(self):
        """Apply a theme using ttk styling."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.UI = {
            "bg": "#0b1220",
            "panel": "#0f1a2b",
            "panel2": "#0c1626",
            "border": "#1e2a3e",
            "text": "#e6eefc",
            "muted": "#a9b7d0",
            "accent": "#3b82f6",
            "danger": "#ef4444",
        }

        self.root.configure(bg=self.UI["bg"])

        style.configure("TFrame", background=self.UI["bg"])
        style.configure("Card.TFrame", background=self.UI["panel"], relief="flat")
        style.configure("TLabel", background=self.UI["bg"], foreground=self.UI["text"])
        style.configure("Muted.TLabel", background=self.UI["bg"], foreground=self.UI["muted"])
        style.configure("Card.TLabel", background=self.UI["panel"], foreground=self.UI["text"])
        style.configure("Title.TLabel", background=self.UI["bg"], foreground=self.UI["text"],
                        font=("Segoe UI", 14, "bold"))
        style.configure("H2.TLabel", background=self.UI["panel"], foreground=self.UI["text"],
                        font=("Segoe UI", 10, "bold"))

        # Buttons
        style.configure("TButton",
                        font=("Segoe UI", 10),
                        padding=(12, 8),
                        background=self.UI["panel"],
                        foreground=self.UI["text"])
        style.map("TButton",
                  background=[("active", self.UI["panel2"])],
                  foreground=[("active", self.UI["text"])])

        style.configure("Accent.TButton", background=self.UI["accent"], foreground="white")
        style.map("Accent.TButton",
                  background=[("active", "#777777")],
                  foreground=[("active", "white")])

        style.configure("Danger.TButton", background=self.UI["danger"], foreground="white")
        style.map("Danger.TButton",
                  background=[("active", "#dc2626")],
                  foreground=[("active", "white")])

        style.configure("Small.TButton", padding=(10, 6))

        # Scales
        style.configure("TScale", background=self.UI["panel"])

    # Menu

    def _build_menu(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open...", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self.save_image, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.save_image_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._exit_app, accelerator="Alt+F4")
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=False)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Reset to Original", command=self.reset_to_original)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        self.root.config(menu=menubar)

