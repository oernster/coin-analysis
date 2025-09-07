"""
Image Widget for Coin Recognition

This module provides a widget for displaying and manipulating images in the Coin Recognition Application.
"""

from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QComboBox, QFileDialog, QSizePolicy, QScrollArea,
    QFrame, QToolButton, QMenu
)
from PySide6.QtCore import Qt, Signal, Slot, QSize, QRectF, QPointF
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush, QTransform, QAction

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, Union, List

from app.utils.image_utils import cv_to_qpixmap, qimage_to_cv


class ImageWidget(QWidget):
    """Widget for displaying and manipulating images."""
    
    # Signals
    image_loaded = Signal(np.ndarray)
    roi_selected = Signal(np.ndarray)
    roi_cleared = Signal()  # Add signal for ROI clearing
    zoom_changed = Signal(float)
    
    def __init__(self, parent=None):
        """Initialize the image widget."""
        super().__init__(parent)
        
        # Image data
        self.cv_image = None
        self.pixmap = None
        self.zoom_factor = 1.0
        self.roi_rect = None
        
        # UI setup
        self._setup_ui()
        
        # Connect signals
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area for image
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        
        # Image label
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setFrameStyle(QFrame.StyledPanel)
        self.image_label.setMouseTracking(True)
        
        # Add image label to scroll area
        self.scroll_area.setWidget(self.image_label)
        
        # Add scroll area to main layout
        main_layout.addWidget(self.scroll_area)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        
        self.zoom_out_button = QToolButton()
        self.zoom_out_button.setText("-")
        self.zoom_out_button.setToolTip("Zoom Out")
        zoom_layout.addWidget(self.zoom_out_button)
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setToolTip("Zoom")
        zoom_layout.addWidget(self.zoom_slider)
        
        self.zoom_in_button = QToolButton()
        self.zoom_in_button.setText("+")
        self.zoom_in_button.setToolTip("Zoom In")
        zoom_layout.addWidget(self.zoom_in_button)
        
        self.zoom_reset_button = QToolButton()
        self.zoom_reset_button.setText("1:1")
        self.zoom_reset_button.setToolTip("Reset Zoom")
        zoom_layout.addWidget(self.zoom_reset_button)
        
        controls_layout.addLayout(zoom_layout)
        
        # ROI controls
        roi_layout = QVBoxLayout()
        
        # Add help label
        roi_help_label = QLabel("Manual ROI Selection: Use these buttons to manually select a region of interest "
                               "containing the year and mint mark. This is optional and can be used instead of "
                               "automatic ROI extraction.")
        roi_help_label.setWordWrap(True)
        roi_layout.addWidget(roi_help_label)
        
        roi_buttons_layout = QHBoxLayout()
        
        self.select_roi_button = QPushButton("Manually Select ROI")
        self.select_roi_button.setToolTip("Draw a rectangle around the year and mint mark")
        roi_buttons_layout.addWidget(self.select_roi_button)
        
        self.clear_roi_button = QPushButton("Clear Manual ROI")
        self.clear_roi_button.setToolTip("Remove the manually selected region and revert to the original image")
        self.clear_roi_button.setEnabled(False)
        roi_buttons_layout.addWidget(self.clear_roi_button)
        
        roi_layout.addLayout(roi_buttons_layout)
        controls_layout.addLayout(roi_layout)
        
        # Add controls to main layout
        main_layout.addLayout(controls_layout)
    
    def _connect_signals(self):
        """Connect signals and slots."""
        # Zoom controls
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        self.zoom_in_button.clicked.connect(self._on_zoom_in)
        self.zoom_out_button.clicked.connect(self._on_zoom_out)
        self.zoom_reset_button.clicked.connect(self._on_zoom_reset)
        
        # ROI controls
        self.select_roi_button.clicked.connect(self._on_select_roi)
        self.clear_roi_button.clicked.connect(self._on_clear_roi)
    
    def set_image(self, image: Union[np.ndarray, QImage, QPixmap, str, Path]):
        """
        Set the image to display.
        
        Args:
            image: Image to display (numpy array, QImage, QPixmap, or file path)
        """
        if image is None:
            self.cv_image = None
            self.pixmap = None
            self.image_label.setText("No image loaded")
            return
        
        # Convert image to OpenCV format (numpy array)
        if isinstance(image, np.ndarray):
            self.cv_image = image.copy()
        elif isinstance(image, QImage):
            self.cv_image = qimage_to_cv(image)
        elif isinstance(image, QPixmap):
            qimage = image.toImage()
            self.cv_image = qimage_to_cv(qimage)
        elif isinstance(image, (str, Path)):
            self.cv_image = cv2.imread(str(image))
            if self.cv_image is None:
                self.image_label.setText(f"Failed to load image: {image}")
                return
        else:
            self.image_label.setText(f"Unsupported image type: {type(image)}")
            return
        
        # Convert to QPixmap for display
        self.pixmap = cv_to_qpixmap(self.cv_image)
        
        # Calculate zoom factor to fit the image in the view
        self._fit_to_view()
        
        # Update display
        self._update_display()
        
        # Emit signal
        self.image_loaded.emit(self.cv_image)
    
    def get_image(self) -> Optional[np.ndarray]:
        """
        Get the current image.
        
        Returns:
            Current image as a numpy array, or None if no image is loaded
        """
        return self.cv_image
    
    def get_roi(self) -> Optional[np.ndarray]:
        """
        Get the selected region of interest.
        
        Returns:
            Selected region of interest as a numpy array, or None if no ROI is selected
        """
        if self.cv_image is None or self.roi_rect is None:
            return None
        
        x, y, w, h = self.roi_rect
        return self.cv_image[y:y+h, x:x+w]
    
    def _fit_to_view(self):
        """Calculate and set the zoom factor to fit the image in the view."""
        if self.pixmap is None:
            return
        
        # Get the size of the scroll area viewport
        viewport_width = self.scroll_area.viewport().width()
        viewport_height = self.scroll_area.viewport().height()
        
        # Get the size of the image
        image_width = self.pixmap.width()
        image_height = self.pixmap.height()
        
        # Calculate the zoom factor to fit the image in the view
        width_ratio = viewport_width / image_width
        height_ratio = viewport_height / image_height
        
        # Use the smaller ratio to ensure the entire image fits
        fit_zoom_factor = min(width_ratio, height_ratio)
        
        # Apply a small margin
        fit_zoom_factor *= 0.95
        
        # Set the zoom factor
        self.zoom_factor = fit_zoom_factor
        
        # Update the zoom slider
        zoom_value = int(self.zoom_factor * 100)
        self.zoom_slider.setValue(zoom_value)
    
    def _update_display(self):
        """Update the image display."""
        if self.pixmap is None:
            return
        
        # Create a copy of the pixmap to draw on
        display_pixmap = QPixmap(self.pixmap)
        
        # Apply zoom
        if self.zoom_factor != 1.0:
            width = int(display_pixmap.width() * self.zoom_factor)
            height = int(display_pixmap.height() * self.zoom_factor)
            display_pixmap = display_pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Draw ROI if selected
        if self.roi_rect is not None:
            painter = QPainter(display_pixmap)
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            
            x, y, w, h = self.roi_rect
            x = int(x * self.zoom_factor)
            y = int(y * self.zoom_factor)
            w = int(w * self.zoom_factor)
            h = int(h * self.zoom_factor)
            
            painter.drawRect(x, y, w, h)
            painter.end()
        
        # Update label
        self.image_label.setPixmap(display_pixmap)
        self.image_label.setMinimumSize(1, 1)  # Allow the label to shrink
        self.image_label.resize(display_pixmap.size())
    
    @Slot(int)
    def _on_zoom_slider_changed(self, value):
        """Handle zoom slider value changed."""
        self.zoom_factor = value / 100.0
        self._update_display()
        self.zoom_changed.emit(self.zoom_factor)
    
    @Slot()
    def _on_zoom_in(self):
        """Handle zoom in button clicked."""
        value = min(self.zoom_slider.value() + 10, self.zoom_slider.maximum())
        self.zoom_slider.setValue(value)
    
    @Slot()
    def _on_zoom_out(self):
        """Handle zoom out button clicked."""
        value = max(self.zoom_slider.value() - 10, self.zoom_slider.minimum())
        self.zoom_slider.setValue(value)
    
    @Slot()
    def _on_zoom_reset(self):
        """Handle zoom reset button clicked."""
        self.zoom_slider.setValue(100)
    
    @Slot()
    def _on_select_roi(self):
        """Handle select ROI button clicked."""
        if self.cv_image is None:
            return
        
        # Use OpenCV's selectROI function with better instructions
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.setWindowTitle("Select ROI", "Select ROI - Draw a rectangle around the year/mint mark and press ENTER")
        roi = cv2.selectROI("Select ROI", self.cv_image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        
        if roi[2] > 0 and roi[3] > 0:
            self.roi_rect = roi
            self._update_display()
            self.clear_roi_button.setEnabled(True)
            
            # Emit signal with ROI image
            roi_image = self.get_roi()
            if roi_image is not None:
                self.roi_selected.emit(roi_image)
    
    @Slot()
    def _on_clear_roi(self):
        """Handle clear ROI button clicked."""
        self.roi_rect = None
        self._update_display()
        self.clear_roi_button.setEnabled(False)
        
        # Emit signal
        self.roi_cleared.emit()


class ImageGalleryWidget(QWidget):
    """Widget for displaying multiple images in a gallery."""
    
    # Signals
    image_selected = Signal(int, np.ndarray)
    
    def __init__(self, parent=None):
        """Initialize the image gallery widget."""
        super().__init__(parent)
        
        # Image data
        self.images = []
        self.selected_index = -1
        
        # UI setup
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Scroll area for thumbnails
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Container widget for thumbnails
        self.thumbnails_widget = QWidget()
        self.thumbnails_layout = QVBoxLayout(self.thumbnails_widget)
        self.thumbnails_layout.setAlignment(Qt.AlignTop)
        self.thumbnails_layout.setContentsMargins(5, 5, 5, 5)
        self.thumbnails_layout.setSpacing(10)
        
        # Add thumbnails widget to scroll area
        self.scroll_area.setWidget(self.thumbnails_widget)
        
        # Add scroll area to main layout
        main_layout.addWidget(self.scroll_area)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Add image button
        self.add_image_button = QPushButton("Add Image")
        controls_layout.addWidget(self.add_image_button)
        
        # Remove image button
        self.remove_image_button = QPushButton("Remove Image")
        self.remove_image_button.setEnabled(False)
        controls_layout.addWidget(self.remove_image_button)
        
        # Add controls to main layout
        main_layout.addLayout(controls_layout)
        
        # Connect signals
        self.add_image_button.clicked.connect(self._on_add_image)
        self.remove_image_button.clicked.connect(self._on_remove_image)
    
    def add_image(self, image: Union[np.ndarray, QImage, QPixmap, str, Path], title: str = ""):
        """
        Add an image to the gallery.
        
        Args:
            image: Image to add (numpy array, QImage, QPixmap, or file path)
            title: Title for the image
        """
        # Convert image to OpenCV format (numpy array)
        if isinstance(image, np.ndarray):
            cv_image = image.copy()
        elif isinstance(image, QImage):
            cv_image = qimage_to_cv(image)
        elif isinstance(image, QPixmap):
            qimage = image.toImage()
            cv_image = qimage_to_cv(qimage)
        elif isinstance(image, (str, Path)):
            cv_image = cv2.imread(str(image))
            if cv_image is None:
                return
            if not title:
                title = Path(image).name
        else:
            return
        
        # Add image to list
        self.images.append((cv_image, title))
        
        # Add thumbnail to layout
        self._add_thumbnail(len(self.images) - 1)
        
        # Enable remove button if there are images
        self.remove_image_button.setEnabled(True)
    
    def _add_thumbnail(self, index: int):
        """
        Add a thumbnail for the image at the given index.
        
        Args:
            index: Index of the image
        """
        cv_image, title = self.images[index]
        
        # Create thumbnail widget
        thumbnail_widget = QWidget()
        thumbnail_layout = QVBoxLayout(thumbnail_widget)
        thumbnail_layout.setContentsMargins(5, 5, 5, 5)
        thumbnail_layout.setSpacing(5)
        
        # Create thumbnail label
        thumbnail_label = QLabel()
        thumbnail_label.setAlignment(Qt.AlignCenter)
        thumbnail_label.setFixedSize(150, 150)
        thumbnail_label.setFrameStyle(QFrame.StyledPanel)
        
        # Create thumbnail pixmap
        thumbnail_pixmap = cv_to_qpixmap(cv_image)
        thumbnail_pixmap = thumbnail_pixmap.scaled(
            thumbnail_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        thumbnail_label.setPixmap(thumbnail_pixmap)
        
        # Create title label
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        
        # Add labels to thumbnail layout
        thumbnail_layout.addWidget(thumbnail_label)
        thumbnail_layout.addWidget(title_label)
        
        # Add thumbnail widget to thumbnails layout
        self.thumbnails_layout.addWidget(thumbnail_widget)
        
        # Connect thumbnail click
        thumbnail_widget.mouseReleaseEvent = lambda event: self._on_thumbnail_clicked(index)
    
    def _on_thumbnail_clicked(self, index: int):
        """
        Handle thumbnail clicked.
        
        Args:
            index: Index of the clicked thumbnail
        """
        self.selected_index = index
        
        # Emit signal
        cv_image, _ = self.images[index]
        self.image_selected.emit(index, cv_image)
    
    def _on_add_image(self):
        """Handle add image button clicked."""
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            for file_path in file_paths:
                self.add_image(file_path)
    
    def _on_remove_image(self):
        """Handle remove image button clicked."""
        if self.selected_index < 0 or self.selected_index >= len(self.images):
            return
        
        # Remove image from list
        self.images.pop(self.selected_index)
        
        # Remove thumbnail from layout
        item = self.thumbnails_layout.takeAt(self.selected_index)
        if item.widget():
            item.widget().deleteLater()
        
        # Update selected index
        if self.selected_index >= len(self.images):
            self.selected_index = len(self.images) - 1
        
        # Disable remove button if there are no images
        self.remove_image_button.setEnabled(len(self.images) > 0)