import sys
from collections import defaultdict

import numpy as np
from skimage import feature, color
from skimage import future  # for fit_segmenter / predict_segmenter
from skimage.morphology import remove_small_objects, remove_small_holes
from sklearn.ensemble import RandomForestClassifier

from PyQt5.QtGui import QPalette
from PyQt5.QtCore import Qt, QRect, QPoint, QSize
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QListWidget, QLineEdit, QSpinBox, QFileDialog, QMessageBox, QSizePolicy,
    QDialog, QTextEdit, QComboBox
)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Patch for old gwyfile + NumPy >= 2.0
_old_fromstring = np.fromstring

def _fromstring_compat(s, dtype=float, count=-1, sep=''):
    # this is the case used by gwyfile: binary buffer, no sep
    if isinstance(s, (bytes, bytearray)) and (sep == '' or sep is None):
        return np.frombuffer(s, dtype=dtype, count=count)
    # fallback to original behavior
    return _old_fromstring(s, dtype=dtype, count=count, sep=sep)

np.fromstring = _fromstring_compat


import gwyfile  # NEW

def load_gwy_topography_like_notebook(path, channel_name=None, scale=1e6):
    """
    Load a .gwy file exactly like in the user's working notebook:
      obj = gwyfile.load(path)
      channels = gwyfile.util.get_datafields(obj)
      channel = channels[first or chosen]
      img = channel.data * scale
    """
    obj = gwyfile.load(path)
    channels = gwyfile.util.get_datafields(obj)

    if not channels:
        raise ValueError("No datafields found in the GWY file.")

    if channel_name is None:
        channel_name = list(channels.keys())[0]  # first one

    channel = channels[channel_name]
    img = channel.data * scale
    return img.astype("float32")

def load_gwy_topography(path):
    """
    Load a .gwy file and return a 2D numpy array (topography).
    We take the first data field we find.
    """
    gwydoc = gwyfile.load(path)

    # data fields are usually under '/0/data', '/1/data', ...
    # This finds the first key that looks like a data field.
    data_keys = [k for k in gwydoc.keys() if k.endswith('/data')]
    if not data_keys:
        raise ValueError("No data field found in GWY file.")

    first_key = sorted(data_keys)[0]
    data_field = gwydoc[first_key]  # this is a GwyDataField-like object

    # data is usually in .data (flattened) with xres, yres
    arr = data_field.data.reshape(data_field.yres, data_field.xres)
    return arr

def numpy_to_qimage(arr, colormap='gray'):
    """Convert a grayscale or RGB numpy array to QImage safely (NumPy 2.0 compatible)."""
    import numpy as np

    arr = np.asarray(arr)

    if arr.ndim == 2:  # grayscale - apply colormap
        h, w = arr.shape
        rng = np.ptp(arr) if np.ptp(arr) > 0 else 1.0
        arr_norm = (arr - np.min(arr)) / rng  # normalize to [0, 1]
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        arr_rgb = cmap(arr_norm)  # returns RGBA in [0, 1]
        arr8 = (255 * arr_rgb[:, :, :3]).astype(np.uint8)  # take RGB, convert to 8-bit
        qimg = QImage(arr8.data, w, h, 3 * w, QImage.Format_RGB888)
        return qimg.copy()

    elif arr.ndim == 3 and arr.shape[2] == 3:
        h, w, ch = arr.shape
        rng = np.ptp(arr) if np.ptp(arr) > 0 else 1.0
        arr8 = (255 * (arr - np.min(arr)) / rng).astype(np.uint8)
        qimg = QImage(arr8.data, w, h, 3 * w, QImage.Format_RGB888)
        return qimg.copy()

    else:
        raise ValueError(f"Unsupported image shape {arr.shape} for display.")


class ImageLabel(QLabel):
    """
    QLabel that shows the image and lets the user draw rectangles.
    It notifies the main window when a rectangle is finished.
    """
    def __init__(self, parent=None, shared_rectangles=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)  # Don't auto-scale pixmap
        self.image = None           # numpy array
        self.pixmap_orig = None     # QPixmap
        self.drawing = False
        self.start_point = QPoint()
        self.current_point = QPoint()
        self.rectangles_by_class = shared_rectangles if shared_rectangles is not None else defaultdict(list)
        self.current_class = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        # zoom system: base_zoom is the fit-to-width scaling, zoom_multiplier
        # is the user-controlled zoom relative to the fitted view. total_zoom = base_zoom * zoom_multiplier
        self.base_zoom = 1.0
        self.zoom_multiplier = 1.0
        self.colormap = 'gray'  # default colormap
        # displayed pixmap metrics (updated in apply_zoom)
        self.display_w = 0
        self.display_h = 0
        self.offset_x = 0
        self.offset_y = 0

    def set_image(self, img_np):
        self.image = img_np
        qimg = numpy_to_qimage(img_np, colormap=self.colormap)
        self.pixmap_orig = QPixmap.fromImage(qimg)
        # reset zoom state and fit baseline
        self.base_zoom = 1.0
        self.zoom_multiplier = 1.0
        self.apply_zoom()

    def apply_zoom(self):
        if self.pixmap_orig is None:
            return
        # total zoom is base (fit) times user multiplier
        total_zoom = self.base_zoom * self.zoom_multiplier
        # Scale the displayed pixmap, keep aspect ratio, smooth
        w = max(1, int(self.pixmap_orig.width() * total_zoom))
        h = max(1, int(self.pixmap_orig.height() * total_zoom))
        disp = self.pixmap_orig.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(disp)
        # update displayed metrics for coordinate transforms
        pm = self.pixmap()
        if pm is not None:
            self.display_w = pm.width()
            self.display_h = pm.height()
        else:
            self.display_w = w
            self.display_h = h

        # compute offsets when pixmap is centered inside the label
        self.offset_x = max(0, (self.width() - self.display_w) // 2)
        self.offset_y = max(0, (self.height() - self.display_h) // 2)

        self.update_scale_factors()
        self.update()

    def zoom_in(self):
        self.zoom_multiplier = min(8.0, self.zoom_multiplier * 1.25)
        self.apply_zoom()

    def zoom_out(self):
        self.zoom_multiplier = max(0.125, self.zoom_multiplier / 1.25)
        self.apply_zoom()

    def fit_to_width(self, target_width: int):
        """Set zoom so displayed image width ≈ target_width, then redraw."""
        if self.pixmap_orig is None:
            return
        if target_width <= 0:
            return
        # base_zoom scales the original image to match the requested width
        new_base = target_width / float(self.pixmap_orig.width())
        self.base_zoom = max(0.01, min(8.0, new_base))
        # reset user multiplier so fit shows whole image
        self.zoom_multiplier = 1.0
        self.apply_zoom()

    def update_scale_factors(self):
        if self.pixmap_orig is None:
            return
        # scale from label coordinates (inside the widget) to original image coords
        # use displayed pixmap size and offsets
        if self.display_w and self.display_h:
            disp_w = self.display_w
            disp_h = self.display_h
        else:
            disp_w = self.pixmap_orig.width()
            disp_h = self.pixmap_orig.height()

        w_lab = self.width()
        h_lab = self.height()

        # factor to convert from displayed pixmap pixels to original image pixels
        self.scale_x = (self.pixmap_orig.width() / float(disp_w)) if disp_w else 1.0
        self.scale_y = (self.pixmap_orig.height() / float(disp_h)) if disp_h else 1.0

    def set_colormap(self, colormap_name):
        """Change the colormap and refresh the display."""
        self.colormap = colormap_name
        if self.image is not None:
            qimg = numpy_to_qimage(self.image, colormap=self.colormap)
            self.pixmap_orig = QPixmap.fromImage(qimg)
            self.apply_zoom()

    def sizeHint(self):
        """Return the preferred size for the label."""
        if self.pixmap_orig is None:
            return QSize(400, 300)
        # Return zoomed pixmap size (base fit * user multiplier)
        total_zoom = self.base_zoom * self.zoom_multiplier
        w = int(self.pixmap_orig.width() * total_zoom)
        h = int(self.pixmap_orig.height() * total_zoom)
        return QSize(w, h)

    def minimumSizeHint(self):
        """Return the minimum size for the label."""
        return QSize(100, 100)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_scale_factors()

    def mousePressEvent(self, event):
        if self.image is None:
            return
        if event.button() == Qt.LeftButton and self.current_class is not None:
            self.drawing = True
            self.start_point = event.pos()
            self.current_point = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.current_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            end_point = event.pos()
            rect = QRect(self.start_point, end_point).normalized()

            # map from label coords to image coords
            # account for pixmap offset (centered inside the label)
            lx1 = rect.left() - self.offset_x
            ly1 = rect.top() - self.offset_y
            lx2 = rect.right() - self.offset_x
            ly2 = rect.bottom() - self.offset_y

            # clamp to displayed pixmap area
            lx1 = max(0, min(lx1, self.display_w - 1))
            lx2 = max(0, min(lx2, self.display_w - 1))
            ly1 = max(0, min(ly1, self.display_h - 1))
            ly2 = max(0, min(ly2, self.display_h - 1))

            x1 = int(lx1 * self.scale_x)
            y1 = int(ly1 * self.scale_y)
            x2 = int(lx2 * self.scale_x)
            y2 = int(ly2 * self.scale_y)

            # store as [x1, x2, y1, y2]
            if self.current_class is not None:
                self.rectangles_by_class[self.current_class].append([x1, x2, y1, y2])

            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.image is None:
            return

        painter = QPainter(self)
        pen = QPen(Qt.red, 2, Qt.SolidLine)
        painter.setPen(pen)

        # draw existing rectangles
        for cls, rects in self.rectangles_by_class.items():
            if cls == self.current_class:
                painter.setPen(QPen(Qt.green, 2))
            else:
                painter.setPen(QPen(Qt.red, 2))
            for r in rects:
                # r = [x1, x2, y1, y2] in image coords; convert to label coords
                # map image -> displayed pixmap coords, then add offsets
                x1 = int(r[0] / self.scale_x) + self.offset_x
                x2 = int(r[1] / self.scale_x) + self.offset_x
                y1 = int(r[2] / self.scale_y) + self.offset_y
                y2 = int(r[3] / self.scale_y) + self.offset_y
                painter.drawRect(QRect(QPoint(x1, y1), QPoint(x2, y2)))

        # draw current rectangle
        if self.drawing:
            painter.setPen(QPen(Qt.yellow, 2, Qt.DashLine))
            painter.drawRect(QRect(self.start_point, self.current_point))


class DualImageLabel(QWidget):
    """
    Widget that displays two images side-by-side (topography and phase).
    Drawing on either image creates boxes on both simultaneously.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)  # Minimum size to avoid being too small
        
        # Shared rectangles between both labels
        self.rectangles_by_class = defaultdict(list)
        
        # Create two ImageLabel instances sharing the same rectangles
        self.label_left = ImageLabel(shared_rectangles=self.rectangles_by_class)
        self.label_right = ImageLabel(shared_rectangles=self.rectangles_by_class)
        
        # Layout: horizontal split
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Add the two ImageLabel widgets directly so they split the available
        # space into two equal halves (no header labels).
        self.label_left.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.label_left, 1)
        layout.addWidget(self.label_right, 1)
        self.setLayout(layout)
    
    def set_images(self, img_topo, img_phase):
        """Set both topography and phase images."""
        self.label_left.set_image(img_topo)
        self.label_right.set_image(img_phase)
    
    def set_image_topography(self, img):
        """Set only topography image."""
        self.label_left.set_image(img)
    
    def set_image_phase(self, img):
        """Set only phase image."""
        self.label_right.set_image(img)
    
    def set_current_class(self, class_name):
        """Set current class for both labels."""
        self.label_left.current_class = class_name
        self.label_right.current_class = class_name
    
    def set_colormap(self, colormap_name):
        """Set colormap for both labels."""
        self.label_left.set_colormap(colormap_name)
        self.label_right.set_colormap(colormap_name)
    
    def get_zoom(self) -> float:
        """Return current shared zoom (use left label as source)."""
        # Return the user zoom multiplier (relative to fitted/base zoom)
        return getattr(self.label_left, 'zoom_multiplier', 1.0)

    def set_zoom(self, z: float):
        """Set zoom for both labels and apply it."""
        # clamp zoom
        z = max(0.125, min(8.0, z))
        # set zoom multiplier on both children (preserve their base_zoom)
        self.label_left.zoom_multiplier = z
        self.label_right.zoom_multiplier = z
        self.label_left.apply_zoom()
        self.label_right.apply_zoom()

    def zoom_in(self):
        """Zoom both images synchronously."""
        newz = min(8.0, self.get_zoom() * 1.25)
        self.set_zoom(newz)

    def zoom_out(self):
        """Zoom out both images synchronously."""
        newz = max(0.125, self.get_zoom() / 1.25)
        self.set_zoom(newz)

    def fit_to_width(self, target_width: int):
        """Fit both images to target width (total width)."""
        # Split target width roughly in half for each image, subtract spacing
        spacing = max(1, self.layout().spacing())
        per_image = max(50, int((target_width - spacing) / 2))
        self.label_left.fit_to_width(per_image)
        self.label_right.fit_to_width(per_image)

        # After fitting children, update this widget's minimum size so the
        # scroll area and layouts allocate space properly (avoid wasted gaps).
        lw = getattr(self.label_left, 'display_w', 0)
        lh = getattr(self.label_left, 'display_h', 0)
        rw = getattr(self.label_right, 'display_w', 0)
        rh = getattr(self.label_right, 'display_h', 0)
        if lw and rw:
            total_w = lw + rw + spacing + 4
            total_h = max(lh, rh) + 4
            self.setMinimumSize(total_w, total_h)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Auto-fit when the widget is resized, but only when the user hasn't
        # zoomed (zoom_multiplier == 1.0). This keeps zoom centered on user actions.
        try:
            left_ok = getattr(self.label_left, 'zoom_multiplier', 1.0) == 1.0
            right_ok = getattr(self.label_right, 'zoom_multiplier', 1.0) == 1.0
            if left_ok and right_ok:
                self.fit_to_width(self.width())
        except Exception:
            pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Random Forest Image Labeler")
        self.image = None  # primary image (topography)
        self.image_secondary = None  # secondary image (phase)
        self.last_prediction = None  # store the last segmentation result

        # central widget
        central = QWidget()
        self.setCentralWidget(central)

        # left side: dual image viewer (topography and phase side-by-side)
        self.image_label = DualImageLabel()

        # right side: class controls
        self.class_list = QListWidget()
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("New class name")
        self.add_class_btn = QPushButton("Add class")
        self.add_class_btn.clicked.connect(self.add_class)
        self.clear_current_btn = QPushButton("Clear current class")
        self.clear_current_btn.clicked.connect(self.clear_current_class)

        self.clear_all_btn = QPushButton("Clear all boxes")
        self.clear_all_btn.clicked.connect(self.clear_all_boxes)

        self.zoom_in_btn = QPushButton("Zoom in")
        self.zoom_in_btn.clicked.connect(self.image_label.zoom_in)

        self.zoom_out_btn = QPushButton("Zoom out")
        self.zoom_out_btn.clicked.connect(self.image_label.zoom_out)

        self.fit_btn = QPushButton("Fit to window")
        self.fit_btn.clicked.connect(self.fit_image_to_window)

        self.load_btn = QPushButton("Load image (Topography)")
        self.load_btn.clicked.connect(self.load_image)

        self.load_secondary_btn = QPushButton("Load Phase image")
        self.load_secondary_btn.clicked.connect(self.load_secondary_image)

        self.view_features_btn = QPushButton("View Features")
        self.view_features_btn.clicked.connect(self.view_features)

        self.train_btn = QPushButton("Train RF & Segment")
        self.train_btn.clicked.connect(self.train_and_segment)

        self.save_prediction_btn = QPushButton("Save Prediction")
        self.save_prediction_btn.clicked.connect(self.save_prediction)

        # Post-processing controls (right panel)
        self.postproc_threshold_input = QSpinBox()
        self.postproc_threshold_input.setRange(1, 10_000_000)
        self.postproc_threshold_input.setValue(64)
        self.postproc_threshold_input.setSingleStep(8)

        self.remove_small_btn = QPushButton("Remove small objects")
        self.remove_small_btn.clicked.connect(self.on_remove_small_objects)

        self.fill_holes_btn = QPushButton("Fill small holes")
        self.fill_holes_btn.clicked.connect(self.on_fill_small_holes)

        # Colormap selector (right panel)
        self.colormap_combo = QComboBox()
        colormaps = ['gray', 'viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'copper', 'twilight', 'jet']
        self.colormap_combo.addItems(colormaps)
        self.colormap_combo.setCurrentText('gray')
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)

        self.scroll = QScrollArea()
        # Allow the scroll area to resize its child to the available viewport
        # so Fit to window can work with maximized dimensions.
        self.scroll.setWidgetResizable(True)
        self.scroll.setBackgroundRole(QPalette.Dark)
        self.scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.load_btn)
        right_layout.addWidget(self.load_secondary_btn)
        right_layout.addWidget(self.class_list)
        right_layout.addWidget(self.class_input)
        right_layout.addWidget(self.add_class_btn)
        right_layout.addWidget(self.clear_current_btn)
        right_layout.addWidget(self.clear_all_btn)
        # post-processing controls
        right_layout.addWidget(self.postproc_threshold_input)
        right_layout.addWidget(self.remove_small_btn)
        right_layout.addWidget(self.fill_holes_btn)
        # colormap selector
        right_layout.addWidget(self.colormap_combo)
        # view features remains on right panel
        right_layout.addWidget(self.view_features_btn)
        right_layout.addStretch()
        right_layout.addWidget(self.train_btn)
        right_layout.addWidget(self.save_prediction_btn)
        
        # Wrap right layout in a widget and set max width
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setMaximumWidth(280)
        right_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        # keep a reference so fitting can subtract the right panel width
        self.right_widget = right_widget

        # Left container: toolbar (horizontal) over the scroll area (image)
        left_container = QWidget()
        left_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(6, 6, 6, 6)
        toolbar_layout.setSpacing(6)
        # add a stretch so buttons are centered-ish, adjust as desired
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.zoom_in_btn)
        toolbar_layout.addWidget(self.zoom_out_btn)
        toolbar_layout.addWidget(self.fit_btn)
        toolbar_layout.addStretch()
        toolbar.setLayout(toolbar_layout)

        # place toolbar above the scroll area
        left_layout.addWidget(toolbar)
        self.scroll.setWidget(self.image_label)
        left_layout.addWidget(self.scroll)
        left_container.setLayout(left_layout)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(left_container, stretch=6)
        layout.addWidget(right_widget, stretch=1)

        central.setLayout(layout)

        self.class_list.currentTextChanged.connect(self.on_class_selected)

    def fit_image_to_window(self):
        # Compute available width for the image viewer using the current
        # central widget width minus the right control panel. Call
        # processEvents so that geometry updates from maximizing are applied.
        try:
            QApplication.processEvents()
        except Exception:
            pass

        try:
            central_w = self.centralWidget().width() if self.centralWidget() is not None else self.width()
            right_w = getattr(self, 'right_widget', None).width() if getattr(self, 'right_widget', None) is not None else 0
            # small margin to avoid touching the scroll frame
            margin = 16
            available = max(0, central_w - right_w - margin)
            # if scroll viewport is actually larger (rare), prefer its viewport
            try:
                if hasattr(self, 'scroll') and self.scroll is not None:
                    vp = self.scroll.viewport().width()
                    # prefer the larger of the two (most visible area)
                    available = max(available, vp - 8)
            except Exception:
                pass
        except Exception:
            available = self.image_label.width()

        # Subtract a small safety margin for borders
        target = max(50, int(available))
        # Delegate to the image label(s) to fit their content to the computed width.
        # DualImageLabel exposes a fit_to_width method on its children.
        try:
            # If using DualImageLabel, call its fit_to_width which forwards to both labels
            if hasattr(self.image_label, 'fit_to_width'):
                self.image_label.fit_to_width(target)
            else:
                # Single ImageLabel fallback
                self.image_label.fit_to_width(target)
        except Exception:
            # Final fallback: instruct left label directly
            try:
                self.image_label.label_left.fit_to_width(target)
                self.image_label.label_right.fit_to_width(target)
            except Exception:
                pass

    def clear_current_class(self):
        item = self.class_list.currentItem()
        if not item:
            return
        cls = item.text()
        self.image_label.rectangles_by_class.pop(cls, None)
        self.image_label.label_left.update()
        self.image_label.label_right.update()

    def clear_all_boxes(self):
        self.image_label.rectangles_by_class.clear()
        self.image_label.label_left.update()
        self.image_label.label_right.update()

    def add_class(self):
        name = self.class_input.text().strip()
        if not name:
            return
        # avoid duplicates
        for i in range(self.class_list.count()):
            if self.class_list.item(i).text() == name:
                self.class_list.setCurrentRow(i)
                return
        self.class_list.addItem(name)
        self.class_input.clear()
        self.class_list.setCurrentRow(self.class_list.count() - 1)

    def on_class_selected(self, name):
        self.image_label.set_current_class(name)
    
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Topography image",
            "",
            "Images (*.png *.jpg *.tif *.bmp);;Gwyddion (*.gwy)"
        )
        if not path:
            return

        import os
        ext = os.path.splitext(path)[1].lower()

        if ext == ".gwy":
            try:
                img = load_gwy_topography_like_notebook(path)
            except Exception as e:
                QMessageBox.critical(self, "GWY load error", f"Could not load {path}:\n{e}")
                return
        else:
            import imageio.v2 as iio
            img = iio.imread(path).astype("float32")
            if img.ndim == 3 and img.shape[2] >= 3:
                img = img[:, :, :3]

        self.image = img
        self.image_label.set_image_topography(img)
        self.image_label.rectangles_by_class.clear()   # NEW: clear boxes on new image
        self.image_label.label_left.update()
        self.image_label.label_right.update()

    def load_secondary_image(self):
        """Load phase image (secondary image for feature generation)."""
        if self.image is None:
            QMessageBox.warning(self, "Load primary first", "Load Topography image first.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Phase image",
            "",
            "Images (*.png *.jpg *.tif *.bmp);;Gwyddion (*.gwy)"
        )
        if not path:
            return

        import os
        ext = os.path.splitext(path)[1].lower()

        if ext == ".gwy":
            try:
                img = load_gwy_topography_like_notebook(path)
            except Exception as e:
                QMessageBox.critical(self, "GWY load error", f"Could not load {path}:\n{e}")
                return
        else:
            import imageio.v2 as iio
            img = iio.imread(path).astype("float32")
            if img.ndim == 3 and img.shape[2] >= 3:
                img = img[:, :, :3]

        # Check shape matches primary image
        if img.shape[:2] != self.image.shape[:2]:
            QMessageBox.critical(
                self,
                "Shape mismatch",
                f"Phase image shape {img.shape[:2]} does not match topography {self.image.shape[:2]}"
            )
            return

        self.image_secondary = img
        self.image_label.set_image_phase(img)
        self.image_label.label_left.update()
        self.image_label.label_right.update()
        QMessageBox.information(self, "Phase loaded", f"Phase image loaded. Shape: {img.shape}")

    def get_feature_names(self):
        """
        Compute the feature array and return a formatted list of feature names/shapes.
        If secondary image (phase) is loaded, concatenate features from both images.
        """
        if self.image is None:
            return None
        
        sigma_min = 1
        sigma_max = 6
        features_func = lambda img: feature.multiscale_basic_features(
            img,
            intensity=True,
            edges=True,
            texture=True,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        
        # Extract features from primary image (topography)
        features = features_func(self.image)
        
        # If secondary image (phase) is loaded, extract features and concatenate
        if self.image_secondary is not None:
            features_secondary = features_func(self.image_secondary)
            # Stack along the feature dimension (axis 2)
            features = np.concatenate([features, features_secondary], axis=2)
        
        return features

    def view_features(self):
        """Show a dialog with feature information."""
        if self.image is None:
            QMessageBox.warning(self, "No image", "Load an image first.")
            return
        
        try:
            features = self.get_feature_names()
            
            # Create a dialog to display feature info
            dialog = QDialog(self)
            dialog.setWindowTitle("Feature Information")
            dialog.setGeometry(100, 100, 500, 400)
            
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            
            # Format the feature information
            info_text = f"Feature Array Shape: {features.shape}\n"
            info_text += f"Number of features: {features.shape[2]}\n"
            info_text += f"Image dimensions: {features.shape[0]} x {features.shape[1]}\n\n"
            
            if self.image_secondary is not None:
                half_features = features.shape[2] // 2
                info_text += "DUAL-IMAGE MODE: Features from BOTH Topography and Phase\n"
                info_text += f"  - Topography features: {half_features}\n"
                info_text += f"  - Phase features: {half_features}\n"
                info_text += f"  - Total: {features.shape[2]} features\n"
                info_text += "\n"
            else:
                info_text += "SINGLE-IMAGE MODE: Features from Topography only\n\n"
            
            info_text += "Features created by multiscale_basic_features:\n"
            info_text += "=" * 50 + "\n"
            
            feature_types = [
                "Intensity (original image)",
                "Gaussian (multiple scales)",
                "Laplacian of Gaussian (multiple scales)",
                "Sobel edges (X and Y directions, multiple scales)",
                "Gaussian gradients (X and Y directions, multiple scales)",
                "Difference of Gaussians (multiple scales)",
                "Hessian matrix elements (multiple scales)",
                "Median filters (multiple scales)",
                "Bilateral filters (multiple scales)",
                "LBP (Local Binary Pattern)",
                "And more texture features..."
            ]
            
            for i, ftype in enumerate(feature_types, 1):
                info_text += f"{i}. {ftype}\n"
            
            info_text += "\n" + "=" * 50 + "\n"
            info_text += f"Total output channels: {features.shape[2]}\n"
            info_text += f"Total values in feature array: {np.prod(features.shape):,}"
            
            text_edit.setText(info_text)
            
            layout = QVBoxLayout()
            layout.addWidget(text_edit)
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not compute features:\n{e}")

    def train_and_segment(self):
        if self.image is None:
            QMessageBox.warning(self, "No image", "Load an image first.")
            return

        if self.class_list.count() == 0:
            QMessageBox.warning(self, "No classes", "Add at least one class.")
            return

        # Create training_labels like in your notebook
        h, w = self.image.shape[:2]
        training_labels = np.zeros((h, w), dtype=np.uint8)

        # Map class name -> label id (1..N)
        class_to_id = {}
        for i in range(self.class_list.count()):
            name = self.class_list.item(i).text()
            class_to_id[name] = i + 1  # start at 1

        for cls_name, rects in self.image_label.rectangles_by_class.items():
            if cls_name not in class_to_id:
                continue
            label_id = class_to_id[cls_name]
            for r in rects:
                x1, x2, y1, y2 = r
                # clip
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))
                training_labels[y1:y2, x1:x2] = label_id

        if np.count_nonzero(training_labels) == 0:
            QMessageBox.warning(self, "No labels", "You didn't draw any rectangles.")
            return

        # compute features like in your notebook
        sigma_min = 1
        sigma_max = 4
        features_func = lambda img: feature.multiscale_basic_features(
            img,
            intensity=True,
            edges=True,
            texture=True,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        # Extract features from primary image (topography)
        features = features_func(self.image)
        
        # If secondary image (phase) is loaded, extract and concatenate features
        if self.image_secondary is not None:
            features_secondary = features_func(self.image_secondary)
            features = np.concatenate([features, features_secondary], axis=2)
            print(f"Training with dual-image features: {features.shape[2]} total channels")
            print(f"  - Topography: ~{features_secondary.shape[2]} features")
            print(f"  - Phase: ~{features_secondary.shape[2]} features")
        else:
            print(f"Training with single-image features: {features.shape[2]} channels")

        # train RF
        clf = RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            max_depth=10,
            max_samples=0.05,
        )

        clf = future.fit_segmenter(training_labels, features, clf)
        pred = future.predict_segmenter(features, clf)

        # Store prediction for saving later
        self.last_prediction = pred

        # show result in matplotlib window (quickest)
        # normalize to 0..1, even if there are negatives
        img = self.image
        imin = np.min(img)
        imax = np.max(img)
        if imax > imin:
            img_norm = (img - imin) / (imax - imin)
        else:
            img_norm = np.zeros_like(img)
        rgb = color.label2rgb(pred, image=img_norm, bg_label=0)
        plt.figure("Segmentation result")
        plt.imshow(rgb)
        plt.title("Predicted classes")
        plt.axis("off")
        plt.show(block = False)

    def save_prediction(self):
        """Save the last prediction map to file."""
        if self.last_prediction is None:
            QMessageBox.warning(self, "No prediction", "Train and segment first to generate a prediction.")
            return
        
        # Open save dialog
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save prediction map",
            "",
            "NumPy binary (*.npy);;NumPy text (*.txt);;XYZ format (*.xyz)"
        )
        if not path:
            return
        
        try:
            if path.endswith('.npy'):
                # NumPy binary format (smallest, fastest)
                np.save(path, self.last_prediction)
                QMessageBox.information(self, "Saved", f"Prediction saved to {path}\nShape: {self.last_prediction.shape}")
            
            elif path.endswith('.txt'):
                # Text format (human-readable, larger file)
                np.savetxt(path, self.last_prediction, fmt='%d')
                QMessageBox.information(self, "Saved", f"Prediction saved to {path}\nShape: {self.last_prediction.shape}")
            
            elif path.endswith('.xyz'):
                # XYZ format: x y class_id (one point per line)
                h, w = self.last_prediction.shape
                with open(path, 'w') as f:
                    for y in range(h):
                        for x in range(w):
                            class_id = int(self.last_prediction[y, x])
                            f.write(f"{x} {y} {class_id}\n")
                QMessageBox.information(self, "Saved", f"Prediction saved to {path}\nShape: {self.last_prediction.shape}\n{h*w:,} points")
            
            else:
                QMessageBox.warning(self, "Unknown format", "Use .npy, .txt, or .xyz extension")
        
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Could not save:\n{e}")

    def _show_prediction(self, pred):
        """Utility to render a prediction array in a matplotlib window."""
        if self.image is None:
            return
        img = self.image
        imin = np.min(img)
        imax = np.max(img)
        if imax > imin:
            img_norm = (img - imin) / (imax - imin)
        else:
            img_norm = np.zeros_like(img)
        rgb = color.label2rgb(pred, image=img_norm, bg_label=0)
        plt.figure("Segmentation result")
        plt.imshow(rgb)
        plt.title("Predicted classes (post-processed)")
        plt.axis("off")
        plt.show(block=False)

    def on_remove_small_objects(self):
        """Remove small connected components per-class smaller than threshold."""
        if self.last_prediction is None:
            QMessageBox.warning(self, "No prediction", "No prediction available. Run segmentation first.")
            return
        thr = int(self.postproc_threshold_input.value())

        pred = self.last_prediction
        new = pred.copy()
        for label in np.unique(pred):
            if label == 0:
                continue
            mask = (pred == label)
            kept = remove_small_objects(mask.astype(bool), min_size=thr)
            # pixels removed become background (0)
            removed = mask & (~kept)
            new[removed] = 0

        self.last_prediction = new
        self._show_prediction(self.last_prediction)

    def on_fill_small_holes(self):
        """Fill small holes inside labeled objects up to the given area threshold."""
        if self.last_prediction is None:
            QMessageBox.warning(self, "No prediction", "No prediction available. Run segmentation first.")
            return
        thr = int(self.postproc_threshold_input.value())

        pred = self.last_prediction
        new = pred.copy()
        for label in np.unique(pred):
            if label == 0:
                continue
            mask = (pred == label)
            filled = remove_small_holes(mask.astype(bool), area_threshold=thr)
            # new pixels that were holes become the label
            add = filled & (~mask)
            new[add] = label

        self.last_prediction = new
        self._show_prediction(self.last_prediction)

    def on_colormap_changed(self, colormap_name):
        """Handle colormap selection change."""
        self.image_label.set_colormap(colormap_name)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 700)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
