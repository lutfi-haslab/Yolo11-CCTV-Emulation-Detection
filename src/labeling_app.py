#!/usr/bin/env python3
"""
AI Image Labeling Tool
======================
A CustomTkinter GUI application for labeling images with bounding boxes.
- Draw bounding boxes on images
- Save/load preset configs in JSON
- Export dataset in YOLO format
- Auto-copy images with label prefix naming (img_01_labelA.png)
"""

import os
import sys
import json
import shutil
import glob
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
from PIL import Image, ImageTk, ImageDraw
import customtkinter as ctk
from pathlib import Path

# === Constants ===
APP_TITLE = "AI Image Labeling Tool"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGES_DIR = os.path.join(BASE_DIR, "images")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "labeled_dataset")
CONFIG_FILE = os.path.join(BASE_DIR, "labeling_config.json")

# Color palette for labels
LABEL_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F1948A", "#82E0AA", "#F8C471", "#AED6F1", "#D2B4DE",
    "#A3E4D7", "#FAD7A0", "#A9CCE3", "#D5DBDB", "#EDBB99",
]


class LabelingConfig:
    """Manages preset configuration saved as JSON."""

    def __init__(self, config_path=CONFIG_FILE):
        self.config_path = config_path
        self.data = {
            "labels": [],
            "images_dir": DEFAULT_IMAGES_DIR,
            "output_dir": DEFAULT_OUTPUT_DIR,
            "last_image_index": 0,
            "train_split": 0.8,
            "image_size": 640,
        }
        self.load()

    def load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    saved = json.load(f)
                    self.data.update(saved)
            except (json.JSONDecodeError, IOError):
                pass

    def save(self):
        with open(self.config_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.save()


class BoundingBox:
    """Represents a single bounding box annotation."""

    def __init__(self, x1, y1, x2, y2, label, label_id):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.label = label
        self.label_id = label_id

    def to_yolo(self, img_width, img_height):
        """Convert to YOLO format: class_id x_center y_center width height (normalized)."""
        x_center = ((self.x1 + self.x2) / 2) / img_width
        y_center = ((self.y1 + self.y2) / 2) / img_height
        width = (self.x2 - self.x1) / img_width
        height = (self.y2 - self.y1) / img_height
        return f"{self.label_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    def to_dict(self):
        return {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "label": self.label, "label_id": self.label_id,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["x1"], d["y1"], d["x2"], d["y2"], d["label"], d["label_id"])


class ImageLabeler(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # === App Setup ===
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title(APP_TITLE)
        self.geometry("1400x900")
        self.minsize(1200, 750)

        # === State ===
        self.config = LabelingConfig()
        self.images_dir = self.config.get("images_dir", DEFAULT_IMAGES_DIR)
        self.output_dir = self.config.get("output_dir", DEFAULT_OUTPUT_DIR)
        self.image_files = []
        self.current_index = 0
        self.current_image = None
        self.display_image = None
        self.photo_image = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Labels
        self.labels = self.config.get("labels", [])
        self.current_label = self.labels[0] if self.labels else None
        self.label_colors = {}
        self._assign_colors()

        # Annotations: {filename: [BoundingBox, ...]}
        self.annotations = {}
        self._load_annotations()

        # Drawing state
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None

        # === Build UI ===
        self._build_ui()
        self._load_images()

        # Load last position
        last_idx = self.config.get("last_image_index", 0)
        if 0 <= last_idx < len(self.image_files):
            self.current_index = last_idx
        self._show_current_image()

    def _assign_colors(self):
        for i, label in enumerate(self.labels):
            self.label_colors[label] = LABEL_COLORS[i % len(LABEL_COLORS)]

    # ========================================
    # UI BUILDING
    # ========================================

    def _build_ui(self):
        # Main grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # === LEFT SIDEBAR ===
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0,
                                      fg_color=("#1a1a2e", "#1a1a2e"))
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Logo / Title
        title_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        title_frame.pack(fill="x", padx=15, pady=(20, 5))
        ctk.CTkLabel(title_frame, text="🏷️ AI Labeler",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color="#4ECDC4").pack(anchor="w")
        ctk.CTkLabel(title_frame, text="Image Annotation Tool",
                     font=ctk.CTkFont(size=11),
                     text_color="#888").pack(anchor="w")

        # Separator
        ctk.CTkFrame(self.sidebar, height=2, fg_color="#333").pack(fill="x", padx=15, pady=10)

        # --- Directory Section ---
        dir_section = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        dir_section.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(dir_section, text="📁 DIRECTORIES",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#aaa").pack(anchor="w", pady=(0, 5))

        self.images_dir_label = ctk.CTkLabel(dir_section,
                                              text=f"Images: {os.path.basename(self.images_dir)}",
                                              font=ctk.CTkFont(size=11),
                                              text_color="#ccc", anchor="w")
        self.images_dir_label.pack(fill="x")
        ctk.CTkButton(dir_section, text="Change Images Dir",
                      command=self._change_images_dir,
                      height=28, font=ctk.CTkFont(size=11),
                      fg_color="#2d3436", hover_color="#636e72").pack(fill="x", pady=(3, 5))

        self.output_dir_label = ctk.CTkLabel(dir_section,
                                              text=f"Output: {os.path.basename(self.output_dir)}",
                                              font=ctk.CTkFont(size=11),
                                              text_color="#ccc", anchor="w")
        self.output_dir_label.pack(fill="x")
        ctk.CTkButton(dir_section, text="Change Output Dir",
                      command=self._change_output_dir,
                      height=28, font=ctk.CTkFont(size=11),
                      fg_color="#2d3436", hover_color="#636e72").pack(fill="x", pady=(3, 5))

        # Separator
        ctk.CTkFrame(self.sidebar, height=2, fg_color="#333").pack(fill="x", padx=15, pady=10)

        # --- Labels Section ---
        label_section = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        label_section.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(label_section, text="🏷️ LABELS",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#aaa").pack(anchor="w", pady=(0, 5))

        # Add label row
        add_label_frame = ctk.CTkFrame(label_section, fg_color="transparent")
        add_label_frame.pack(fill="x", pady=(0, 5))
        self.new_label_entry = ctk.CTkEntry(add_label_frame, placeholder_text="New label...",
                                             height=30, font=ctk.CTkFont(size=12))
        self.new_label_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(add_label_frame, text="+ Add", width=60,
                      command=self._add_label,
                      height=30, font=ctk.CTkFont(size=11),
                      fg_color="#00b894", hover_color="#00a381").pack(side="right")

        # Labels list (scrollable)
        self.labels_scroll = ctk.CTkScrollableFrame(label_section, height=180,
                                                     fg_color="#16213e",
                                                     corner_radius=8)
        self.labels_scroll.pack(fill="x", pady=5)
        self._refresh_labels_list()

        # Separator
        ctk.CTkFrame(self.sidebar, height=2, fg_color="#333").pack(fill="x", padx=15, pady=10)

        # --- Actions Section ---
        action_section = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        action_section.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(action_section, text="⚡ ACTIONS",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#aaa").pack(anchor="w", pady=(0, 8))

        ctk.CTkButton(action_section, text="💾 Save Annotations",
                      command=self._save_annotations,
                      height=35, font=ctk.CTkFont(size=12),
                      fg_color="#0984e3", hover_color="#0773c5").pack(fill="x", pady=3)

        ctk.CTkButton(action_section, text="📦 Export YOLO Dataset",
                      command=self._export_yolo_dataset,
                      height=35, font=ctk.CTkFont(size=12),
                      fg_color="#6c5ce7", hover_color="#5a4bd1").pack(fill="x", pady=3)



        ctk.CTkButton(action_section, text="🗑️ Clear Current Annotations",
                      command=self._clear_current_annotations,
                      height=30, font=ctk.CTkFont(size=11),
                      fg_color="#d63031", hover_color="#b52a2b").pack(fill="x", pady=3)

        # --- Config Section ---
        ctk.CTkFrame(self.sidebar, height=2, fg_color="#333").pack(fill="x", padx=15, pady=10)
        config_section = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        config_section.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(config_section, text="⚙️ CONFIG",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#aaa").pack(anchor="w", pady=(0, 5))

        split_frame = ctk.CTkFrame(config_section, fg_color="transparent")
        split_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(split_frame, text="Train split:", font=ctk.CTkFont(size=11),
                     text_color="#ccc").pack(side="left")
        self.train_split_var = ctk.StringVar(value=str(self.config.get("train_split", 0.8)))
        ctk.CTkEntry(split_frame, textvariable=self.train_split_var,
                     width=60, height=25, font=ctk.CTkFont(size=11)).pack(side="right")

        size_frame = ctk.CTkFrame(config_section, fg_color="transparent")
        size_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(size_frame, text="Image size:", font=ctk.CTkFont(size=11),
                     text_color="#ccc").pack(side="left")
        self.img_size_var = ctk.StringVar(value=str(self.config.get("image_size", 640)))
        ctk.CTkEntry(size_frame, textvariable=self.img_size_var,
                     width=60, height=25, font=ctk.CTkFont(size=11)).pack(side="right")

        ctk.CTkButton(config_section, text="Save Config",
                      command=self._save_config,
                      height=28, font=ctk.CTkFont(size=11),
                      fg_color="#2d3436", hover_color="#636e72").pack(fill="x", pady=(8, 3))

        # === CENTER / MAIN AREA ===
        main_area = ctk.CTkFrame(self, fg_color=("#0f0f23", "#0f0f23"), corner_radius=0)
        main_area.grid(row=0, column=1, sticky="nsew")
        main_area.grid_rowconfigure(1, weight=1)
        main_area.grid_columnconfigure(0, weight=1)

        # --- Top Bar (navigation) ---
        top_bar = ctk.CTkFrame(main_area, height=50, fg_color="#16213e", corner_radius=0)
        top_bar.grid(row=0, column=0, sticky="ew")
        top_bar.grid_columnconfigure(2, weight=1)

        ctk.CTkButton(top_bar, text="◀ Prev", width=80,
                      command=self._prev_image,
                      height=32, font=ctk.CTkFont(size=12),
                      fg_color="#2d3436", hover_color="#636e72").grid(row=0, column=0, padx=10, pady=8)

        ctk.CTkButton(top_bar, text="Next ▶", width=80,
                      command=self._next_image,
                      height=32, font=ctk.CTkFont(size=12),
                      fg_color="#2d3436", hover_color="#636e72").grid(row=0, column=1, padx=5, pady=8)

        self.image_info_label = ctk.CTkLabel(top_bar, text="No images loaded",
                                              font=ctk.CTkFont(size=13),
                                              text_color="#ddd")
        self.image_info_label.grid(row=0, column=2, padx=15, pady=8, sticky="w")

        self.bbox_count_label = ctk.CTkLabel(top_bar, text="Boxes: 0",
                                              font=ctk.CTkFont(size=12),
                                              text_color="#4ECDC4")
        self.bbox_count_label.grid(row=0, column=3, padx=15, pady=8, sticky="e")

        # --- Canvas ---
        canvas_frame = ctk.CTkFrame(main_area, fg_color="#0a0a1a", corner_radius=8)
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="#0a0a1a", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Canvas bindings
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)  # Right-click to delete

        # --- Bottom Status Bar ---
        status_bar = ctk.CTkFrame(main_area, height=30, fg_color="#16213e", corner_radius=0)
        status_bar.grid(row=2, column=0, sticky="ew")
        self.status_label = ctk.CTkLabel(status_bar,
                                          text="Ready — Select a label and draw boxes on the image",
                                          font=ctk.CTkFont(size=11),
                                          text_color="#888")
        self.status_label.pack(side="left", padx=15, pady=5)

        self.coords_label = ctk.CTkLabel(status_bar, text="",
                                          font=ctk.CTkFont(size=11),
                                          text_color="#666")
        self.coords_label.pack(side="right", padx=15, pady=5)
        self.canvas.bind("<Motion>", self._on_mouse_move)

        # Keyboard shortcuts
        self.bind("<Left>", lambda e: self._prev_image())
        self.bind("<Right>", lambda e: self._next_image())
        self.bind("<Control-s>", lambda e: self._save_annotations())
        self.bind("<Command-s>", lambda e: self._save_annotations())
        self.bind("<Delete>", lambda e: self._delete_last_bbox())
        self.bind("<BackSpace>", lambda e: self._delete_last_bbox())

    # ========================================
    # LABELS MANAGEMENT
    # ========================================

    def _refresh_labels_list(self):
        """Rebuild the labels list in the sidebar."""
        # Safely remove old widgets — use pack_forget + after-idle destroy
        # to avoid CTkButton._font AttributeError on Python 3.13
        for widget in self.labels_scroll.winfo_children():
            widget.pack_forget()
            try:
                widget.destroy()
            except (AttributeError, tk.TclError):
                pass

        if not self.labels:
            ctk.CTkLabel(self.labels_scroll, text="No labels yet",
                         font=ctk.CTkFont(size=11), text_color="#666").pack(pady=10)
            return

        for i, label in enumerate(self.labels):
            color = self.label_colors.get(label, "#aaa")
            is_selected = (label == self.current_label)

            row = ctk.CTkFrame(self.labels_scroll, fg_color="transparent")
            row.pack(fill="x", pady=2)

            if is_selected:
                btn = ctk.CTkButton(
                    row,
                    text=f"  {i}: {label}",
                    anchor="w",
                    height=30,
                    font=ctk.CTkFont(size=12, weight="bold"),
                    fg_color=color,
                    hover_color=color,
                    text_color="#fff",
                    border_width=2,
                    border_color=color,
                    command=lambda l=label: self._select_label(l),
                )
            else:
                btn = ctk.CTkButton(
                    row,
                    text=f"  {i}: {label}",
                    anchor="w",
                    height=30,
                    font=ctk.CTkFont(size=12),
                    fg_color="#2d3436",
                    hover_color=color,
                    text_color="#ccc",
                    border_width=0,
                    command=lambda l=label: self._select_label(l),
                )
            btn.pack(side="left", fill="x", expand=True, padx=(0, 3))

            # Color indicator
            color_dot = ctk.CTkFrame(row, width=12, height=12,
                                      fg_color=color, corner_radius=6)
            color_dot.pack(side="left", padx=2)

            # Delete button
            del_btn = ctk.CTkButton(row, text="✕", width=25, height=25,
                                     font=ctk.CTkFont(size=10),
                                     fg_color="#d63031", hover_color="#b52a2b",
                                     command=lambda l=label: self._remove_label(l))
            del_btn.pack(side="right", padx=2)

    def _add_label(self):
        """Add a new label (no image copying — copies are made only on export)."""
        label = self.new_label_entry.get().strip()
        if not label:
            messagebox.showwarning("Warning", "Please enter a label name.")
            return
        if label in self.labels:
            messagebox.showwarning("Warning", f"Label '{label}' already exists.")
            return

        self.labels.append(label)
        self.label_colors[label] = LABEL_COLORS[len(self.labels) - 1 % len(LABEL_COLORS)]
        self.current_label = label
        self._assign_colors()
        self.config.set("labels", self.labels)
        self.new_label_entry.delete(0, "end")
        self._refresh_labels_list()

        self._set_status(f"Label '{label}' added — select it to start annotating")

    def _remove_label(self, label):
        if messagebox.askyesno("Confirm", f"Remove label '{label}'?"):
            self.labels.remove(label)
            if self.current_label == label:
                self.current_label = self.labels[0] if self.labels else None
            self.config.set("labels", self.labels)
            self._assign_colors()
            self._refresh_labels_list()
            self._redraw_canvas()

    def _select_label(self, label):
        self.current_label = label
        self._refresh_labels_list()
        self._set_status(f"Active label: {label}")

    # ========================================
    # IMAGE LOADING & NAVIGATION
    # ========================================

    def _load_images(self):
        """Scan images directory for image files."""
        self.image_files = []
        if not os.path.isdir(self.images_dir):
            self._set_status(f"Images directory not found: {self.images_dir}")
            return

        supported = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
        for f in sorted(os.listdir(self.images_dir)):
            if f.lower().endswith(supported) and os.path.isfile(os.path.join(self.images_dir, f)):
                self.image_files.append(f)

        if self.image_files:
            self._set_status(f"Loaded {len(self.image_files)} images from {self.images_dir}")
        else:
            self._set_status("No images found in directory")

    def _show_current_image(self):
        """Display the current image on the canvas."""
        if not self.image_files:
            self.image_info_label.configure(text="No images loaded")
            self.canvas.delete("all")
            return

        filename = self.image_files[self.current_index]
        filepath = os.path.join(self.images_dir, filename)

        try:
            self.current_image = Image.open(filepath).convert("RGB")
        except Exception as e:
            self._set_status(f"Error loading {filename}: {e}")
            return

        self.image_info_label.configure(
            text=f"[{self.current_index + 1}/{len(self.image_files)}] {filename}  "
                 f"({self.current_image.width}×{self.current_image.height})"
        )
        self.config.set("last_image_index", self.current_index)
        self._fit_image_to_canvas()
        self._redraw_canvas()

    def _fit_image_to_canvas(self):
        """Scale image to fit canvas while maintaining aspect ratio."""
        if not self.current_image:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return

        img_w, img_h = self.current_image.size
        scale = min(canvas_w / img_w, canvas_h / img_h)

        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        self.scale_x = img_w / new_w
        self.scale_y = img_h / new_h
        self.offset_x = (canvas_w - new_w) // 2
        self.offset_y = (canvas_h - new_h) // 2

        self.display_image = self.current_image.resize((new_w, new_h), Image.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(self.display_image)

    def _redraw_canvas(self):
        """Redraw the canvas with image and all bounding boxes."""
        self.canvas.delete("all")

        if self.photo_image:
            self.canvas.create_image(self.offset_x, self.offset_y,
                                      anchor="nw", image=self.photo_image)

        # Draw bounding boxes
        filename = self._current_filename()
        if filename and filename in self.annotations:
            boxes = self.annotations[filename]
            for box in boxes:
                self._draw_bbox(box)

        # Update count
        count = len(self.annotations.get(filename, [])) if filename else 0
        self.bbox_count_label.configure(text=f"Boxes: {count}")

    def _draw_bbox(self, box):
        """Draw a single bounding box on canvas."""
        color = self.label_colors.get(box.label, "#fff")

        # Convert from image coords to canvas coords
        x1 = box.x1 / self.scale_x + self.offset_x
        y1 = box.y1 / self.scale_y + self.offset_y
        x2 = box.x2 / self.scale_x + self.offset_x
        y2 = box.y2 / self.scale_y + self.offset_y

        self.canvas.create_rectangle(x1, y1, x2, y2,
                                      outline=color, width=2, tags="bbox")

        # Label tag
        label_text = f"{box.label}"
        self.canvas.create_rectangle(x1, y1 - 18, x1 + len(label_text) * 8 + 10, y1,
                                      fill=color, outline=color, tags="bbox")
        self.canvas.create_text(x1 + 5, y1 - 9,
                                 anchor="w", text=label_text,
                                 fill="#fff", font=("Arial", 10, "bold"), tags="bbox")

    def _prev_image(self):
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self._show_current_image()

    def _next_image(self):
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self._show_current_image()

    def _current_filename(self):
        if self.image_files and 0 <= self.current_index < len(self.image_files):
            return self.image_files[self.current_index]
        return None

    # ========================================
    # DRAWING BOUNDING BOXES
    # ========================================

    def _canvas_to_image_coords(self, cx, cy):
        """Convert canvas coordinates to original image coordinates."""
        ix = (cx - self.offset_x) * self.scale_x
        iy = (cy - self.offset_y) * self.scale_y
        if self.current_image:
            ix = max(0, min(ix, self.current_image.width))
            iy = max(0, min(iy, self.current_image.height))
        return ix, iy

    def _on_mouse_down(self, event):
        if not self.current_label:
            self._set_status("⚠️ Please select or create a label first!")
            return
        if not self.current_image:
            return

        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y

    def _on_mouse_drag(self, event):
        if not self.drawing:
            return
        if self.current_rect:
            self.canvas.delete(self.current_rect)

        color = self.label_colors.get(self.current_label, "#fff")
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline=color, width=2, dash=(4, 4)
        )

    def _on_mouse_up(self, event):
        if not self.drawing:
            return
        self.drawing = False
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None

        # Convert to image coordinates
        ix1, iy1 = self._canvas_to_image_coords(self.start_x, self.start_y)
        ix2, iy2 = self._canvas_to_image_coords(event.x, event.y)

        # Minimum size check
        if abs(ix2 - ix1) < 5 or abs(iy2 - iy1) < 5:
            return

        label_id = self.labels.index(self.current_label)
        bbox = BoundingBox(ix1, iy1, ix2, iy2, self.current_label, label_id)

        filename = self._current_filename()
        if filename:
            if filename not in self.annotations:
                self.annotations[filename] = []
            self.annotations[filename].append(bbox)
            self._redraw_canvas()
            self._set_status(f"Added bbox: {self.current_label} ({ix1:.0f},{iy1:.0f})-({ix2:.0f},{iy2:.0f})")

    def _on_right_click(self, event):
        """Delete the bbox under the cursor."""
        filename = self._current_filename()
        if not filename or filename not in self.annotations:
            return

        # Find clicked bbox (in canvas coords)
        cx, cy = event.x, event.y
        for i, box in reversed(list(enumerate(self.annotations[filename]))):
            x1 = box.x1 / self.scale_x + self.offset_x
            y1 = box.y1 / self.scale_y + self.offset_y
            x2 = box.x2 / self.scale_x + self.offset_x
            y2 = box.y2 / self.scale_y + self.offset_y
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                removed = self.annotations[filename].pop(i)
                self._redraw_canvas()
                self._set_status(f"Removed bbox: {removed.label}")
                return

    def _delete_last_bbox(self):
        filename = self._current_filename()
        if filename and filename in self.annotations and self.annotations[filename]:
            removed = self.annotations[filename].pop()
            self._redraw_canvas()
            self._set_status(f"Removed last bbox: {removed.label}")

    def _on_mouse_move(self, event):
        if self.current_image:
            ix, iy = self._canvas_to_image_coords(event.x, event.y)
            self.coords_label.configure(text=f"({ix:.0f}, {iy:.0f})")

    def _on_canvas_resize(self, event):
        self._fit_image_to_canvas()
        self._redraw_canvas()

    # ========================================
    # ANNOTATIONS SAVE/LOAD
    # ========================================

    def _get_annotations_file(self):
        return os.path.join(BASE_DIR, "annotations.json")

    def _save_annotations(self):
        data = {}
        for filename, boxes in self.annotations.items():
            data[filename] = [box.to_dict() for box in boxes]

        with open(self._get_annotations_file(), "w") as f:
            json.dump(data, f, indent=2)

        self._set_status(f"💾 Annotations saved ({len(data)} images)")
        messagebox.showinfo("Saved", f"Annotations saved for {len(data)} images.")

    def _load_annotations(self):
        ann_file = self._get_annotations_file()
        if os.path.exists(ann_file):
            try:
                with open(ann_file, "r") as f:
                    data = json.load(f)
                for filename, boxes_data in data.items():
                    self.annotations[filename] = [BoundingBox.from_dict(b) for b in boxes_data]
            except (json.JSONDecodeError, IOError):
                pass

    def _clear_current_annotations(self):
        filename = self._current_filename()
        if filename:
            if messagebox.askyesno("Confirm", f"Clear all annotations for {filename}?"):
                self.annotations[filename] = []
                self._redraw_canvas()
                self._set_status(f"Cleared annotations for {filename}")


    # ========================================
    # EXPORT YOLO DATASET
    # ========================================

    def _export_yolo_dataset(self):
        """
        Export all annotations as a YOLO format dataset.
        
        Each annotated image is split into per-label copies:
          img_01.png → img_01_person.png (only person bboxes)
                     → img_01_cap.png    (only cap bboxes)
                     → img_01_mask.png   (only mask bboxes)
        
        This matches the simple_dataset format and improves training accuracy.
        """
        if not self.labels:
            messagebox.showwarning("Warning", "No labels defined.")
            return

        if not self.annotations:
            messagebox.showwarning("Warning", "No annotations to export.")
            return

        # Save annotations first
        self._save_annotations_silent()

        # Clean output dir
        output_dir = self.output_dir
        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            d = os.path.join(output_dir, sub)
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        # Get annotated files
        annotated_files = [f for f, boxes in self.annotations.items() if boxes]
        if not annotated_files:
            messagebox.showwarning("Warning", "No annotated images found.")
            return

        # Build all image-label pairs: split each image by label
        # e.g. img_01.png with person+cap+mask boxes becomes 3 pairs
        pairs = []  # [(src_filename, label_name, label_id, [boxes]), ...]

        for filename in annotated_files:
            boxes = self.annotations[filename]
            base_name = os.path.splitext(filename)[0]

            # Group boxes by label
            boxes_by_label = {}
            for box in boxes:
                if box.label not in boxes_by_label:
                    boxes_by_label[box.label] = []
                boxes_by_label[box.label].append(box)

            # Create a pair for each label found in this image
            for label, label_boxes in boxes_by_label.items():
                label_id = self.labels.index(label)
                pairs.append((filename, base_name, label, label_id, label_boxes))

        if not pairs:
            messagebox.showwarning("Warning", "No label pairs to export.")
            return

        # Shuffle and split
        import random
        random.shuffle(pairs)
        try:
            split_ratio = float(self.train_split_var.get())
        except ValueError:
            split_ratio = 0.8
        split_idx = max(1, int(len(pairs) * split_ratio))
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:] if split_idx < len(pairs) else [pairs[-1]]

        if not val_pairs:
            val_pairs = [train_pairs.pop()]

        train_list = []
        val_list = []

        def process_pairs(pair_list, img_dir, lbl_dir, file_list):
            for filename, base_name, label, label_id, label_boxes in pair_list:
                src = os.path.join(self.images_dir, filename)
                if not os.path.exists(src):
                    continue

                # Prefixed output name: img_01_person.png, img_01_cap.png, etc.
                out_name = f"{base_name}_{label}"
                dest_img = os.path.join(img_dir, out_name + ".png")
                dest_lbl = os.path.join(lbl_dir, out_name + ".txt")

                # Copy image as PNG
                try:
                    img = Image.open(src).convert("RGB")
                    img.save(dest_img, "PNG")
                except Exception as e:
                    print(f"Error copying {filename}: {e}")
                    continue

                # Write YOLO label (only bboxes for this specific label)
                with open(dest_lbl, "w") as f:
                    for box in label_boxes:
                        f.write(box.to_yolo(img.width, img.height) + "\n")

                file_list.append(os.path.abspath(dest_img))

        process_pairs(train_pairs,
                      os.path.join(output_dir, "images", "train"),
                      os.path.join(output_dir, "labels", "train"),
                      train_list)
        process_pairs(val_pairs,
                      os.path.join(output_dir, "images", "val"),
                      os.path.join(output_dir, "labels", "val"),
                      val_list)

        # Write data.yaml
        yaml_content = "names:\n"
        for i, label in enumerate(self.labels):
            yaml_content += f"  {i}: {label}\n"
        yaml_content += f"path: {os.path.abspath(output_dir)}\n"
        yaml_content += "train: images/train\n"
        yaml_content += "val: images/val\n"
        yaml_content += f"nc: {len(self.labels)}\n"

        with open(os.path.join(output_dir, "data.yaml"), "w") as f:
            f.write(yaml_content)

        # Write Train.txt and Val.txt
        with open(os.path.join(output_dir, "Train.txt"), "w") as f:
            f.write("\n".join(train_list) + "\n")
        with open(os.path.join(output_dir, "Val.txt"), "w") as f:
            f.write("\n".join(val_list) + "\n")

        total_pairs = len(train_list) + len(val_list)
        self._set_status(f"📦 YOLO dataset exported: {total_pairs} pairs ({len(train_list)} train / {len(val_list)} val)")
        messagebox.showinfo("Export Complete",
                            f"YOLO dataset exported!\n\n"
                            f"📁 {output_dir}\n"
                            f"🏋️ Train: {len(train_list)} image-label pairs\n"
                            f"✅ Val: {len(val_list)} image-label pairs\n"
                            f"📊 Total: {total_pairs} pairs from {len(annotated_files)} images\n"
                            f"🏷️ Labels: {', '.join(self.labels)}\n\n"
                            f"Each image split by label:\n"
                            f"  img_XX_label.png + img_XX_label.txt\n\n"
                            f"data.yaml ready for YOLO training.")

    def _save_annotations_silent(self):
        data = {}
        for filename, boxes in self.annotations.items():
            data[filename] = [box.to_dict() for box in boxes]
        with open(self._get_annotations_file(), "w") as f:
            json.dump(data, f, indent=2)

    # ========================================
    # DIRECTORY MANAGEMENT
    # ========================================

    def _change_images_dir(self):
        d = filedialog.askdirectory(title="Select Images Directory",
                                     initialdir=self.images_dir)
        if d:
            self.images_dir = d
            self.config.set("images_dir", d)
            self.images_dir_label.configure(text=f"Images: {os.path.basename(d)}")
            self.current_index = 0
            self._load_images()
            self._show_current_image()

    def _change_output_dir(self):
        d = filedialog.askdirectory(title="Select Output Directory",
                                     initialdir=self.output_dir)
        if d:
            self.output_dir = d
            self.config.set("output_dir", d)
            self.output_dir_label.configure(text=f"Output: {os.path.basename(d)}")

    # ========================================
    # CONFIG
    # ========================================

    def _save_config(self):
        try:
            split = float(self.train_split_var.get())
            img_size = int(self.img_size_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid config values.")
            return

        self.config.set("train_split", split)
        self.config.set("image_size", img_size)
        self.config.set("labels", self.labels)
        self._set_status("⚙️ Config saved")
        messagebox.showinfo("Config Saved", f"Configuration saved to {self.config.config_path}")

    # ========================================
    # STATUS
    # ========================================

    def _set_status(self, text):
        self.status_label.configure(text=text)
        print(f"[Status] {text}")


# === MAIN ===
if __name__ == "__main__":
    app = ImageLabeler()
    app.mainloop()
