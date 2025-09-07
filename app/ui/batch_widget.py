"""
Batch Processing Widget for Coin Recognition

This module provides a widget for batch processing multiple images in the Coin Recognition Application.
"""

from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QProgressBar, QListWidget, QListWidgetItem, QFileDialog,
    QGroupBox, QFormLayout, QSpinBox, QCheckBox, QComboBox,
    QSplitter, QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot, QDir, QThread, QObject
from PySide6.QtGui import QIcon, QPixmap, QFont

import os
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from app.utils.image_utils import is_supported_image


class BatchProcessingWorker(QObject):
    """Worker for batch processing images in a separate thread."""
    
    # Signals
    progress_updated = Signal(int, int)  # current, total
    image_processed = Signal(str, dict)  # image_path, results
    processing_finished = Signal()
    processing_error = Signal(str, str)  # image_path, error_message
    
    def __init__(self, image_paths: List[str], processor_func):
        """
        Initialize the batch processing worker.
        
        Args:
            image_paths: List of paths to images to process
            processor_func: Function to process each image
        """
        super().__init__()
        self.image_paths = image_paths
        self.processor_func = processor_func
        self.stop_requested = False
    
    def process(self):
        """Process all images."""
        total = len(self.image_paths)
        
        for i, image_path in enumerate(self.image_paths):
            if self.stop_requested:
                break
            
            try:
                # Update progress
                self.progress_updated.emit(i + 1, total)
                
                # Process image
                results = self.processor_func(image_path)
                
                # Emit results
                self.image_processed.emit(image_path, results)
                
                # Small delay to prevent UI freezing
                time.sleep(0.01)
            
            except Exception as e:
                self.processing_error.emit(image_path, str(e))
        
        # Emit finished signal
        self.processing_finished.emit()
    
    def stop(self):
        """Request to stop processing."""
        self.stop_requested = True


class BatchWidget(QWidget):
    """Widget for batch processing multiple images."""
    
    # Signals
    batch_started = Signal(List[str])
    batch_finished = Signal()
    image_processed = Signal(str, dict)  # image_path, results
    
    def __init__(self, parent=None):
        """Initialize the batch widget."""
        super().__init__(parent)
        
        # Batch processing state
        self.image_paths = []
        self.results = {}  # {image_path: results}
        self.processing_thread = None
        self.processing_worker = None
        
        # UI setup
        self._setup_ui()
        
        # Connect signals
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Input group
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)
        
        # Input controls
        input_controls_layout = QHBoxLayout()
        
        self.select_folder_button = QPushButton("Select Folder")
        input_controls_layout.addWidget(self.select_folder_button)
        
        self.select_files_button = QPushButton("Select Files")
        input_controls_layout.addWidget(self.select_files_button)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.setEnabled(False)
        input_controls_layout.addWidget(self.clear_button)
        
        input_layout.addLayout(input_controls_layout)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        input_layout.addWidget(self.file_list)
        
        # Add input group to main layout
        main_layout.addWidget(input_group)
        
        # Options group
        options_group = QGroupBox("Options")
        options_layout = QFormLayout(options_group)
        
        # Processing method
        self.processing_method_combo = QComboBox()
        self.processing_method_combo.addItem("Default", "default")
        self.processing_method_combo.addItem("High Accuracy", "high_accuracy")
        self.processing_method_combo.addItem("Fast", "fast")
        options_layout.addRow("Processing Method:", self.processing_method_combo)
        
        # Save results
        self.save_results_checkbox = QCheckBox("Save Results")
        self.save_results_checkbox.setChecked(True)
        options_layout.addRow("", self.save_results_checkbox)
        
        # Add options group to main layout
        main_layout.addWidget(options_group)
        
        # Results group
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        # Results table
        self.results_table = QTableWidget(0, 4)  # 0 rows, 4 columns
        self.results_table.setHorizontalHeaderLabels(["Image", "Year", "Mint", "Confidence"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        results_layout.addWidget(self.results_table)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.export_csv_button = QPushButton("Export CSV")
        self.export_csv_button.setEnabled(False)
        export_layout.addWidget(self.export_csv_button)
        
        self.export_json_button = QPushButton("Export JSON")
        self.export_json_button.setEnabled(False)
        export_layout.addWidget(self.export_json_button)
        
        results_layout.addLayout(export_layout)
        
        # Add results group to main layout
        main_layout.addWidget(results_group)
        
        # Progress group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        # Process buttons
        process_layout = QHBoxLayout()
        
        self.process_button = QPushButton("Process")
        self.process_button.setEnabled(False)
        process_layout.addWidget(self.process_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        process_layout.addWidget(self.stop_button)
        
        progress_layout.addLayout(process_layout)
        
        # Add progress group to main layout
        main_layout.addWidget(progress_group)
        
        # Set stretch factors
        main_layout.setStretchFactor(input_group, 3)
        main_layout.setStretchFactor(options_group, 1)
        main_layout.setStretchFactor(results_group, 3)
        main_layout.setStretchFactor(progress_group, 1)
    
    def _connect_signals(self):
        """Connect signals and slots."""
        # Input controls
        self.select_folder_button.clicked.connect(self._on_select_folder)
        self.select_files_button.clicked.connect(self._on_select_files)
        self.clear_button.clicked.connect(self._on_clear)
        
        # Process controls
        self.process_button.clicked.connect(self._on_process)
        self.stop_button.clicked.connect(self._on_stop)
        
        # Export controls
        self.export_csv_button.clicked.connect(lambda: self._on_export("csv"))
        self.export_json_button.clicked.connect(lambda: self._on_export("json"))
    
    def set_processor_func(self, func):
        """
        Set the function to process each image.
        
        Args:
            func: Function that takes an image path and returns results
        """
        self.processor_func = func
    
    @Slot()
    def _on_select_folder(self):
        """Handle select folder button clicked."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder", QDir.homePath()
        )
        
        if not folder_path:
            return
        
        # Find all image files in the folder
        image_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if is_supported_image(file_path):
                    image_paths.append(file_path)
        
        # Add to list
        self._add_image_paths(image_paths)
    
    @Slot()
    def _on_select_files(self):
        """Handle select files button clicked."""
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            self._add_image_paths(file_paths)
    
    def _add_image_paths(self, paths: List[str]):
        """
        Add image paths to the list.
        
        Args:
            paths: List of paths to add
        """
        # Add to internal list
        for path in paths:
            if path not in self.image_paths:
                self.image_paths.append(path)
        
        # Update UI
        self._update_file_list()
        
        # Enable buttons
        self.clear_button.setEnabled(len(self.image_paths) > 0)
        self.process_button.setEnabled(len(self.image_paths) > 0)
    
    def _update_file_list(self):
        """Update the file list widget."""
        # Clear list
        self.file_list.clear()
        
        # Add items
        for path in self.image_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            self.file_list.addItem(item)
    
    @Slot()
    def _on_clear(self):
        """Handle clear button clicked."""
        # Clear lists
        self.image_paths.clear()
        self.results.clear()
        
        # Update UI
        self.file_list.clear()
        self.results_table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
        
        # Disable buttons
        self.clear_button.setEnabled(False)
        self.process_button.setEnabled(False)
        self.export_csv_button.setEnabled(False)
        self.export_json_button.setEnabled(False)
    
    @Slot()
    def _on_process(self):
        """Handle process button clicked."""
        if not hasattr(self, 'processor_func'):
            QMessageBox.warning(
                self,
                "Error",
                "No processor function set. Please set a processor function before processing."
            )
            return
        
        if not self.image_paths:
            return
        
        # Clear previous results
        self.results.clear()
        self.results_table.setRowCount(0)
        
        # Update UI
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing...")
        self.process_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.select_folder_button.setEnabled(False)
        self.select_files_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        
        # Create worker and thread
        self.processing_worker = BatchProcessingWorker(self.image_paths, self.processor_func)
        self.processing_thread = QThread()
        
        # Move worker to thread
        self.processing_worker.moveToThread(self.processing_thread)
        
        # Connect signals
        self.processing_thread.started.connect(self.processing_worker.process)
        self.processing_worker.progress_updated.connect(self._on_progress_updated)
        self.processing_worker.image_processed.connect(self._on_image_processed)
        self.processing_worker.processing_finished.connect(self._on_processing_finished)
        self.processing_worker.processing_error.connect(self._on_processing_error)
        
        # Start thread
        self.processing_thread.start()
        
        # Emit signal
        self.batch_started.emit(self.image_paths)
    
    @Slot()
    def _on_stop(self):
        """Handle stop button clicked."""
        if self.processing_worker:
            self.processing_worker.stop()
            self.status_label.setText("Stopping...")
            self.stop_button.setEnabled(False)
    
    @Slot(int, int)
    def _on_progress_updated(self, current: int, total: int):
        """
        Handle progress updated.
        
        Args:
            current: Current progress
            total: Total progress
        """
        # Update progress bar
        progress = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        
        # Update status label
        self.status_label.setText(f"Processing {current}/{total}...")
    
    @Slot(str, dict)
    def _on_image_processed(self, image_path: str, results: Dict[str, Any]):
        """
        Handle image processed.
        
        Args:
            image_path: Path to the processed image
            results: Processing results
        """
        # Add to results
        self.results[image_path] = results
        
        # Add to results table
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # Image path
        image_item = QTableWidgetItem(os.path.basename(image_path))
        image_item.setToolTip(image_path)
        self.results_table.setItem(row, 0, image_item)
        
        # Year
        year = results.get('year')
        year_item = QTableWidgetItem(str(year) if year is not None else "Unknown")
        self.results_table.setItem(row, 1, year_item)
        
        # Mint
        mint = results.get('mint')
        mint_item = QTableWidgetItem(mint if mint else "Unknown")
        self.results_table.setItem(row, 2, mint_item)
        
        # Confidence
        confidence = results.get('confidence')
        confidence_item = QTableWidgetItem(f"{confidence*100:.1f}%" if confidence is not None else "N/A")
        self.results_table.setItem(row, 3, confidence_item)
        
        # Emit signal
        self.image_processed.emit(image_path, results)
    
    @Slot(str, str)
    def _on_processing_error(self, image_path: str, error_message: str):
        """
        Handle processing error.
        
        Args:
            image_path: Path to the image that caused the error
            error_message: Error message
        """
        # Add to results table
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # Image path
        image_item = QTableWidgetItem(os.path.basename(image_path))
        image_item.setToolTip(image_path)
        self.results_table.setItem(row, 0, image_item)
        
        # Error
        error_item = QTableWidgetItem("Error")
        error_item.setToolTip(error_message)
        self.results_table.setItem(row, 1, error_item)
        
        # Empty cells
        self.results_table.setItem(row, 2, QTableWidgetItem(""))
        self.results_table.setItem(row, 3, QTableWidgetItem(""))
    
    @Slot()
    def _on_processing_finished(self):
        """Handle processing finished."""
        # Clean up thread
        if self.processing_thread:
            self.processing_thread.quit()
            self.processing_thread.wait()
            self.processing_thread = None
            self.processing_worker = None
        
        # Update UI
        self.status_label.setText("Finished")
        self.process_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.select_folder_button.setEnabled(True)
        self.select_files_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        
        # Enable export buttons if there are results
        self.export_csv_button.setEnabled(len(self.results) > 0)
        self.export_json_button.setEnabled(len(self.results) > 0)
        
        # Emit signal
        self.batch_finished.emit()
    
    @Slot(str)
    def _on_export(self, format: str):
        """
        Handle export button clicked.
        
        Args:
            format: Export format ('csv' or 'json')
        """
        if not self.results:
            return
        
        # Get save path
        file_dialog = QFileDialog(self)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        
        if format == 'csv':
            file_dialog.setNameFilter("CSV Files (*.csv)")
            file_dialog.setDefaultSuffix("csv")
        else:  # json
            file_dialog.setNameFilter("JSON Files (*.json)")
            file_dialog.setDefaultSuffix("json")
        
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            
            try:
                if format == 'csv':
                    self._export_csv(file_path)
                else:  # json
                    self._export_json(file_path)
                
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Results exported to {file_path}"
                )
            
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Export Error",
                    f"Error exporting results: {str(e)}"
                )
    
    def _export_csv(self, file_path: str):
        """
        Export results to a CSV file.
        
        Args:
            file_path: Path to save the CSV file
        """
        import csv
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Image', 'Year', 'Mint', 'Confidence'])
            
            # Write data
            for image_path, results in self.results.items():
                year = results.get('year')
                mint = results.get('mint')
                confidence = results.get('confidence')
                
                writer.writerow([
                    image_path,
                    year if year is not None else '',
                    mint if mint else '',
                    f"{confidence*100:.1f}%" if confidence is not None else ''
                ])
    
    def _export_json(self, file_path: str):
        """
        Export results to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        import json
        
        # Convert results to serializable format
        serializable_results = {}
        for image_path, results in self.results.items():
            serializable_results[image_path] = {
                'year': results.get('year'),
                'mint': results.get('mint'),
                'confidence': results.get('confidence')
            }
        
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)