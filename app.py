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

