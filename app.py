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

    # Layout
    def _build_layout(self):
        """Build the main window layout."""
        self.main = ttk.Frame(self.root, padding=14)
        self.main.pack(fill=tk.BOTH, expand=True)

        self.main.columnconfigure(0, weight=3)
        self.main.columnconfigure(1, weight=1)
        self.main.rowconfigure(0, weight=1)

        left = ttk.Frame(self.main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        header = ttk.Frame(left)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Image Editor", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Open | Edit | Undo/Redo | Save", style="Muted.TLabel").grid(row=1, column=0, sticky="w")

        canvas_card = ttk.Frame(left, style="Card.TFrame", padding=10)
        canvas_card.grid(row=1, column=0, sticky="nsew")
        canvas_card.rowconfigure(0, weight=1)
        canvas_card.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_card,
            bg="#050a14",
            highlightthickness=1,
            highlightbackground=self.UI["border"],
            bd=0
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        right = ttk.Frame(self.main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        # Quick actions card
        quick = ttk.Frame(right, style="Card.TFrame", padding=12)
        quick.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        quick.columnconfigure((0, 1), weight=1)

        ttk.Label(quick, text="Quick Filters", style="H2.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Button(quick, text="Grayscale", style="Accent.TButton", command=self.apply_grayscale).grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(quick, text="Edge Detect", command=self.apply_edge).grid(row=1, column=1, sticky="ew", padx=(6, 0))

        # Transform card
        transform = ttk.Frame(right, style="Card.TFrame", padding=12)
        transform.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        transform.columnconfigure((0, 1, 2), weight=1)

        ttk.Label(transform, text="Transforms", style="H2.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Button(transform, text="↻ 90°", style="Small.TButton", command=lambda: self.apply_rotate(90)).grid(row=1, column=0, sticky="ew", padx=3)
        ttk.Button(transform, text="⤾ 180°", style="Small.TButton", command=lambda: self.apply_rotate(180)).grid(row=1, column=1, sticky="ew", padx=3)
        ttk.Button(transform, text="↺ 270°", style="Small.TButton", command=lambda: self.apply_rotate(270)).grid(row=1, column=2, sticky="ew", padx=3)

        ttk.Button(transform, text="⇋ Flip H", style="Small.TButton", command=lambda: self.apply_flip("horizontal")).grid(row=2, column=0, sticky="ew", padx=3, pady=(8, 0))
        ttk.Button(transform, text="⇅ Flip V", style="Small.TButton", command=lambda: self.apply_flip("vertical")).grid(row=2, column=1, sticky="ew", padx=3, pady=(8, 0))

        # Adjustments card
        adjust = ttk.Frame(right, style="Card.TFrame", padding=12)
        adjust.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        adjust.columnconfigure(0, weight=1)

        ttk.Label(adjust, text="Adjustments (drag to preview)", style="H2.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))

        def slider_row(r: int, title: str, scale_widget: ttk.Scale):
            ttk.Label(adjust, text=title, style="Card.TLabel").grid(row=r, column=0, sticky="w")
            scale_widget.grid(row=r + 1, column=0, sticky="ew", pady=(4, 10))

        self.blur_scale = ttk.Scale(adjust, from_=1, to=31, orient="horizontal",
                                    command=lambda _v: self._preview_slider("blur"))
        slider_row(1, "Blur (1–31)", self.blur_scale)
        self.blur_scale.set(self.blur_var.get())

        self.brightness_scale = ttk.Scale(adjust, from_=-100, to=100, orient="horizontal",
                                          command=lambda _v: self._preview_slider("brightness"))
        slider_row(3, "Brightness (-100..100)", self.brightness_scale)
        self.brightness_scale.set(self.brightness_var.get())

        self.contrast_scale = ttk.Scale(adjust, from_=-100, to=100, orient="horizontal",
                                        command=lambda _v: self._preview_slider("contrast"))
        slider_row(5, "Contrast (-100..100)", self.contrast_scale)
        self.contrast_scale.set(self.contrast_var.get())

        self.scale_scale = ttk.Scale(adjust, from_=10, to=200, orient="horizontal",
                                     command=lambda _v: self._preview_slider("scale"))
        slider_row(7, "Resize % (10–200)", self.scale_scale)
        self.scale_scale.set(self.scale_var.get())

        for w in (self.blur_scale, self.brightness_scale, self.contrast_scale, self.scale_scale):
            w.bind("<ButtonPress-1>", self._start_preview)
            w.bind("<ButtonRelease-1>", self._commit_preview)

        # History card
        hist = ttk.Frame(right, style="Card.TFrame", padding=12)
        hist.grid(row=3, column=0, sticky="ew")
        hist.columnconfigure((0, 1, 2), weight=1)

        ttk.Label(hist, text="History", style="H2.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Button(hist, text="↶ Undo", command=self.undo).grid(row=1, column=0, sticky="ew", padx=3)
        ttk.Button(hist, text="↷ Redo", command=self.redo).grid(row=1, column=1, sticky="ew", padx=3)
        ttk.Button(hist, text="↻ Reset", style="Danger.TButton", command=self.reset_to_original).grid(row=1, column=2, sticky="ew", padx=3)

        # Status bar
        self.status = ttk.Label(self.root, text="", anchor="w", padding=(12, 8), style="Muted.TLabel")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # Helpers 

    def _require_image(self) -> bool:
        """Return True if an image is loaded; otherwise show a warning and return False."""
        if not self.model.has_image():
            messagebox.showwarning("No Image", "Please open an image first (File → Open).")
            return False
        return True
    
    def _cv_to_tk(self, img_bgr: np.ndarray, max_w: int, max_h: int) -> ImageTk.PhotoImage:
        """
        Convert an OpenCV BGR image to a Tk-compatible PhotoImage.

        The output is resized to fit within (max_w, max_h) while preserving aspect ratio.
        """
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        w, h = pil.size
        scale = min(max_w / w, max_h / h, 1.0)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        pil = pil.resize(new_size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil)

    def _render(self):
        """Render the current image onto the canvas."""
        self.canvas.delete("all")
        img = self.model.current()

        if img is None:
            self.canvas.create_text(
                30, 30,
                anchor="nw",
                fill="#93a4c7",
                font=("Segoe UI", 12),
                text="No image loaded\n\nUse File → Open to select a JPG / PNG / BMP."
            )
            return

        self.root.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            cw, ch = 800, 500

        self._tk_image = self._cv_to_tk(img, cw, ch)
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_image, anchor="center")

        w, h = self.model.get_dimensions()
        self._update_status(f"{self.model.filename()}  |  {w}x{h}px")

    def _update_status(self, text: str):
        """Update the status bar message."""
        self.status.config(text=text)

    def _ui_snapshot(self) -> Dict[str, Any]:
        """
        Capture current UI control values.

        Uses the scales as the source of truth because they reflect what the user
        is actually dragging.
        """
        return {
            "blur": int(float(self.blur_scale.get())),
            "brightness": int(float(self.brightness_scale.get())),
            "contrast": int(float(self.contrast_scale.get())),
            "scale": int(float(self.scale_scale.get())),
        }
    
                    
