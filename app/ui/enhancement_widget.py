"""
Enhancement Widget for Coin Recognition

This module provides a widget for configuring image enhancement options in the Coin Recognition Application.
"""

from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QComboBox, QCheckBox, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QTabWidget, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot, QSize

import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple

from app.processing.preprocessor import (
    convert_to_grayscale, enhance_contrast, reduce_noise, 
    detect_edges, extract_roi
)


class EnhancementWidget(QWidget):
    """Widget for configuring image enhancement options."""
    
    # Signals
    parameters_changed = Signal(dict)
    apply_requested = Signal()
    reset_requested = Signal()
    
    def __init__(self, parent=None):
        """Initialize the enhancement widget."""
        super().__init__(parent)
        
        # Default parameters
        self.parameters = {
            'grayscale': True,
            'contrast': {
                'enabled': True,
                'clip_limit': 2.0,
                'tile_grid_size': (8, 8)
            },
            'noise_reduction': {
                'enabled': True,
                'method': 'gaussian',
                'kernel_size': 5
            },
            'roi': {
                'enabled': True,
                'type': 'right_half'
            }
        }
        
        # Store image states for reverting
        self.original_image = None
        self.pre_edge_detection_image = None
        
        # UI setup
        self._setup_ui()
        
        # Initialize UI with default parameters
        self._update_ui_from_parameters()
        
        # Connect signals
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Grayscale group
        grayscale_group = QGroupBox("Grayscale Conversion")
        grayscale_layout = QVBoxLayout(grayscale_group)
        
        self.grayscale_checkbox = QCheckBox("Convert to Grayscale")
        grayscale_layout.addWidget(self.grayscale_checkbox)
        
        main_layout.addWidget(grayscale_group)
        
        # Contrast enhancement group
        contrast_group = QGroupBox("Contrast Enhancement")
        contrast_layout = QFormLayout(contrast_group)
        
        self.contrast_checkbox = QCheckBox("Enable Contrast Enhancement")
        contrast_layout.addRow(self.contrast_checkbox)
        
        self.clip_limit_slider = QDoubleSpinBox()
        self.clip_limit_slider.setRange(0.5, 10.0)
        self.clip_limit_slider.setSingleStep(0.1)
        self.clip_limit_slider.setDecimals(1)
        contrast_layout.addRow("Clip Limit:", self.clip_limit_slider)
        
        self.tile_size_combo = QComboBox()
        self.tile_size_combo.addItem("4x4", (4, 4))
        self.tile_size_combo.addItem("8x8", (8, 8))
        self.tile_size_combo.addItem("16x16", (16, 16))
        contrast_layout.addRow("Tile Size:", self.tile_size_combo)
        
        main_layout.addWidget(contrast_group)
        
        # Noise reduction group
        noise_group = QGroupBox("Noise Reduction")
        noise_layout = QFormLayout(noise_group)
        
        self.noise_checkbox = QCheckBox("Enable Noise Reduction")
        noise_layout.addRow(self.noise_checkbox)
        
        self.noise_method_combo = QComboBox()
        self.noise_method_combo.addItem("Gaussian Blur", "gaussian")
        self.noise_method_combo.addItem("Median Blur", "median")
        self.noise_method_combo.addItem("Bilateral Filter", "bilateral")
        noise_layout.addRow("Method:", self.noise_method_combo)
        
        self.kernel_size_combo = QComboBox()
        self.kernel_size_combo.addItem("3x3", 3)
        self.kernel_size_combo.addItem("5x5", 5)
        self.kernel_size_combo.addItem("7x7", 7)
        self.kernel_size_combo.addItem("9x9", 9)
        noise_layout.addRow("Kernel Size:", self.kernel_size_combo)
        
        main_layout.addWidget(noise_group)
        
        # ROI group
        roi_group = QGroupBox("Region of Interest (Optional)")
        roi_layout = QFormLayout(roi_group)
        
        self.roi_checkbox = QCheckBox("Extract Region of Interest")
        roi_layout.addRow(self.roi_checkbox)
        
        # Add help text
        roi_help_label = QLabel("ROI extraction is optional. You can either:\n"
                               "1. Use automatic extraction (select type below)\n"
                               "2. Manually select a region using the 'Select ROI' button\n"
                               "3. Disable ROI extraction entirely")
        roi_help_label.setWordWrap(True)
        roi_layout.addRow(roi_help_label)
        
        self.roi_type_combo = QComboBox()
        self.roi_type_combo.addItem("Right Half (Year/Mint)", "right_half")
        self.roi_type_combo.addItem("Left Half (Face)", "left_half")
        self.roi_type_combo.addItem("Center", "center")
        roi_layout.addRow("Automatic ROI Type:", self.roi_type_combo)
        
        main_layout.addWidget(roi_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.apply_button = QPushButton("Apply")
        buttons_layout.addWidget(self.apply_button)
        
        self.reset_button = QPushButton("Reset")
        buttons_layout.addWidget(self.reset_button)
        
        main_layout.addLayout(buttons_layout)
    
    def _connect_signals(self):
        """Connect signals and slots."""
        # Grayscale
        self.grayscale_checkbox.stateChanged.connect(self._on_parameter_changed)
        
        # Contrast enhancement
        self.contrast_checkbox.stateChanged.connect(self._on_parameter_changed)
        self.clip_limit_slider.valueChanged.connect(self._on_parameter_changed)
        self.tile_size_combo.currentIndexChanged.connect(self._on_parameter_changed)
        
        # Noise reduction
        self.noise_checkbox.stateChanged.connect(self._on_parameter_changed)
        self.noise_method_combo.currentIndexChanged.connect(self._on_parameter_changed)
        self.kernel_size_combo.currentIndexChanged.connect(self._on_parameter_changed)
        
        # ROI
        self.roi_checkbox.stateChanged.connect(self._on_parameter_changed)
        self.roi_type_combo.currentIndexChanged.connect(self._on_parameter_changed)
        
        # Buttons
        self.apply_button.clicked.connect(self.apply_requested)
        self.reset_button.clicked.connect(self._on_reset)
    
    def _update_ui_from_parameters(self):
        """Update UI components from parameters."""
        # Grayscale
        self.grayscale_checkbox.setChecked(self.parameters['grayscale'])
        
        # Contrast enhancement
        self.contrast_checkbox.setChecked(self.parameters['contrast']['enabled'])
        self.clip_limit_slider.setValue(self.parameters['contrast']['clip_limit'])
        
        # Find the index of the tile size in the combo box
        tile_size = self.parameters['contrast']['tile_grid_size']
        for i in range(self.tile_size_combo.count()):
            if self.tile_size_combo.itemData(i) == tile_size:
                self.tile_size_combo.setCurrentIndex(i)
                break
        
        # Noise reduction
        self.noise_checkbox.setChecked(self.parameters['noise_reduction']['enabled'])
        
        # Find the index of the noise method in the combo box
        noise_method = self.parameters['noise_reduction']['method']
        for i in range(self.noise_method_combo.count()):
            if self.noise_method_combo.itemData(i) == noise_method:
                self.noise_method_combo.setCurrentIndex(i)
                break
        
        # Find the index of the kernel size in the combo box
        kernel_size = self.parameters['noise_reduction']['kernel_size']
        for i in range(self.kernel_size_combo.count()):
            if self.kernel_size_combo.itemData(i) == kernel_size:
                self.kernel_size_combo.setCurrentIndex(i)
                break
        
        # ROI
        self.roi_checkbox.setChecked(self.parameters['roi']['enabled'])
        
        # Find the index of the ROI type in the combo box
        roi_type = self.parameters['roi']['type']
        for i in range(self.roi_type_combo.count()):
            if self.roi_type_combo.itemData(i) == roi_type:
                self.roi_type_combo.setCurrentIndex(i)
                break
    
    def _update_parameters_from_ui(self):
        """Update parameters from UI components."""
        # Grayscale
        self.parameters['grayscale'] = self.grayscale_checkbox.isChecked()
        
        # Contrast enhancement
        self.parameters['contrast']['enabled'] = self.contrast_checkbox.isChecked()
        self.parameters['contrast']['clip_limit'] = self.clip_limit_slider.value()
        self.parameters['contrast']['tile_grid_size'] = self.tile_size_combo.currentData()
        
        # Noise reduction
        self.parameters['noise_reduction']['enabled'] = self.noise_checkbox.isChecked()
        self.parameters['noise_reduction']['method'] = self.noise_method_combo.currentData()
        self.parameters['noise_reduction']['kernel_size'] = self.kernel_size_combo.currentData()
        
        # ROI
        self.parameters['roi']['enabled'] = self.roi_checkbox.isChecked()
        self.parameters['roi']['type'] = self.roi_type_combo.currentData()
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current enhancement parameters.
        
        Returns:
            Dictionary of enhancement parameters
        """
        return self.parameters
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """
        Set the enhancement parameters.
        
        Args:
            parameters: Dictionary of enhancement parameters
        """
        self.parameters = parameters
        self._update_ui_from_parameters()
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process an image using the current enhancement parameters.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        # Store the original image
        self.original_image = image.copy()
        
        # Make a copy of the input image
        result = image.copy()
        
        # Apply grayscale conversion
        if self.parameters['grayscale']:
            result = convert_to_grayscale(result)
        
        # Apply ROI extraction
        if self.parameters['roi']['enabled']:
            result = extract_roi(result, self.parameters['roi']['type'])
        
        # Apply contrast enhancement
        if self.parameters['contrast']['enabled']:
            result = enhance_contrast(
                result,
                clip_limit=self.parameters['contrast']['clip_limit'],
                tile_grid_size=self.parameters['contrast']['tile_grid_size'],
                force_grayscale=False  # Never force grayscale
            )
        
        # Apply noise reduction
        if self.parameters['noise_reduction']['enabled']:
            result = reduce_noise(
                result,
                method=self.parameters['noise_reduction']['method'],
                kernel_size=self.parameters['noise_reduction']['kernel_size']
            )
        
        return result
    
    @Slot()
    def _on_parameter_changed(self):
        """Handle parameter changed."""
        # Update parameters from UI
        self._update_parameters_from_ui()
        
        # Emit signal
        self.parameters_changed.emit(self.parameters)
    
    @Slot(int)
    def _on_threshold1_changed(self, value):
        """Handle threshold1 slider value changed."""
        self.threshold1_label.setText(str(value))
        self._on_parameter_changed()
    
    @Slot(int)
    def _on_threshold2_changed(self, value):
        """Handle threshold2 slider value changed."""
        self.threshold2_label.setText(str(value))
        self._on_parameter_changed()
    
    @Slot()
    def _on_reset(self):
        """Handle reset button clicked."""
        # Reset parameters to defaults
        self.parameters = {
            'grayscale': True,
            'contrast': {
                'enabled': True,
                'clip_limit': 2.0,
                'tile_grid_size': (8, 8)
            },
            'noise_reduction': {
                'enabled': True,
                'method': 'gaussian',
                'kernel_size': 5
            },
            'edge_detection': {
                'enabled': False,
                'method': 'canny',
                'threshold1': 50,
                'threshold2': 150
            },
            'roi': {
                'enabled': True,
                'type': 'right_half'
            }
        }
        
        # Update UI
        self._update_ui_from_parameters()
        
        # Emit signals
        self.parameters_changed.emit(self.parameters)
        self.reset_requested.emit()