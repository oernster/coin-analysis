"""
Results Widget for Coin Recognition

This module provides a widget for displaying the results of coin recognition in the Coin Recognition Application.
"""

from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QFormLayout, QProgressBar, QSplitter, QFrame, QTextEdit
)
from PySide6.QtCore import Qt, Signal, Slot, QSize
from PySide6.QtGui import QFont, QColor, QBrush, QPalette

import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class ResultsWidget(QWidget):
    """Widget for displaying coin recognition results."""
    
    # Signals
    export_requested = Signal(str)  # Export format
    open_image_clicked = Signal()  # Open image button clicked
    process_image_clicked = Signal()  # Process image button clicked
    batch_process_clicked = Signal()  # Batch process button clicked
    single_mode_clicked = Signal()  # Single image mode button clicked
    batch_mode_clicked = Signal()  # Batch mode button clicked
    
    def __init__(self, parent=None):
        """Initialize the results widget."""
        super().__init__(parent)
        
        # Results data
        self.year = None
        self.mint = None
        self.year_confidence = None
        self.mint_confidence = None
        self.history = []  # List of (image_path, year, mint, year_confidence, mint_confidence)
        
        # UI setup
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Current result group
        current_group = QGroupBox("Current Result")
        current_layout = QFormLayout(current_group)
        
        # Year result
        year_layout = QHBoxLayout()
        self.year_label = QLabel("Unknown")
        self.year_label.setFont(QFont("Arial", 16, QFont.Bold))
        year_layout.addWidget(self.year_label)
        
        self.year_confidence_bar = QProgressBar()
        self.year_confidence_bar.setRange(0, 100)
        self.year_confidence_bar.setValue(0)
        self.year_confidence_bar.setTextVisible(True)
        self.year_confidence_bar.setFormat("%p%")
        year_layout.addWidget(self.year_confidence_bar)
        
        current_layout.addRow("Year:", year_layout)
        
        # Mint result
        mint_layout = QHBoxLayout()
        self.mint_label = QLabel("Unknown")
        self.mint_label.setFont(QFont("Arial", 16, QFont.Bold))
        mint_layout.addWidget(self.mint_label)
        
        self.mint_confidence_bar = QProgressBar()
        self.mint_confidence_bar.setRange(0, 100)
        self.mint_confidence_bar.setValue(0)
        self.mint_confidence_bar.setTextVisible(True)
        self.mint_confidence_bar.setFormat("%p%")
        mint_layout.addWidget(self.mint_confidence_bar)
        
        current_layout.addRow("Mint:", mint_layout)
        
        # Add action buttons under mint mark
        buttons_layout = QHBoxLayout()
        
        # Create styled buttons with blue background and white text
        button_style = """
            QPushButton {
                background-color: #0000AA;
                color: white;
                border-radius: 4px;
                padding: 6px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #0000FF;
            }
            QPushButton:pressed {
                background-color: #000088;
            }
        """
        
        # Open Image button
        self.open_image_button = QPushButton("Open Image")
        self.open_image_button.setStyleSheet(button_style)
        buttons_layout.addWidget(self.open_image_button)
        
        # Process Image button
        self.process_image_button = QPushButton("Process Image")
        self.process_image_button.setStyleSheet(button_style)
        buttons_layout.addWidget(self.process_image_button)
        
        # Batch Process button
        self.batch_process_button = QPushButton("Batch Process")
        self.batch_process_button.setStyleSheet(button_style)
        buttons_layout.addWidget(self.batch_process_button)
        
        # Add second row of buttons
        buttons_layout2 = QHBoxLayout()
        
        # Single Image button
        self.single_mode_button = QPushButton("Single Image")
        self.single_mode_button.setStyleSheet(button_style)
        buttons_layout2.addWidget(self.single_mode_button)
        
        # Batch Mode button
        self.batch_mode_button = QPushButton("Batch Mode")
        self.batch_mode_button.setStyleSheet(button_style)
        buttons_layout2.addWidget(self.batch_mode_button)
        
        # Add buttons to the current layout
        current_layout.addRow("", buttons_layout)
        current_layout.addRow("", buttons_layout2)
        
        # Add current result group to main layout
        main_layout.addWidget(current_group)
        
        # History group
        history_group = QGroupBox("History")
        history_layout = QVBoxLayout(history_group)
        
        # History table
        self.history_table = QTableWidget(0, 5)  # 0 rows, 5 columns
        self.history_table.setHorizontalHeaderLabels(["Image", "Year", "Mint", "Year Conf.", "Mint Conf."])
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        history_layout.addWidget(self.history_table)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.export_csv_button = QPushButton("Export CSV")
        self.export_csv_button.setEnabled(False)
        export_layout.addWidget(self.export_csv_button)
        
        self.export_json_button = QPushButton("Export JSON")
        self.export_json_button.setEnabled(False)
        export_layout.addWidget(self.export_json_button)
        
        self.clear_history_button = QPushButton("Clear History")
        self.clear_history_button.setEnabled(False)
        export_layout.addWidget(self.clear_history_button)
        
        history_layout.addLayout(export_layout)
        
        # Add history group to main layout
        main_layout.addWidget(history_group)
        
        # Details group
        details_group = QGroupBox("Details")
        details_layout = QVBoxLayout(details_group)
        
        # Details text
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        details_layout.addWidget(self.details_text)
        
        # Add details group to main layout
        main_layout.addWidget(details_group)
        
        # Set stretch factors
        main_layout.setStretchFactor(current_group, 1)
        main_layout.setStretchFactor(history_group, 2)
        main_layout.setStretchFactor(details_group, 1)
        
        # Connect signals
        self.export_csv_button.clicked.connect(lambda: self.export_requested.emit("csv"))
        self.export_json_button.clicked.connect(lambda: self.export_requested.emit("json"))
        self.clear_history_button.clicked.connect(self._on_clear_history)
        self.history_table.itemSelectionChanged.connect(self._on_history_selection_changed)
        
        # Connect action buttons
        self.open_image_button.clicked.connect(self.open_image_clicked)
        self.process_image_button.clicked.connect(self.process_image_clicked)
        self.batch_process_button.clicked.connect(self.batch_process_clicked)
        self.single_mode_button.clicked.connect(self.single_mode_clicked)
        self.batch_mode_button.clicked.connect(self.batch_mode_clicked)
    
    def set_result(self, year: Optional[int], mint: Optional[str], 
                  year_confidence: Optional[float] = None, mint_confidence: Optional[float] = None,
                  image_path: Optional[str] = None):
        """
        Set the current recognition result.
        
        Args:
            year: Recognized year
            mint: Recognized mint mark
            year_confidence: Confidence score for year recognition (0-1)
            mint_confidence: Confidence score for mint recognition (0-1)
            image_path: Path to the image
        """
        self.year = year
        self.mint = mint
        self.year_confidence = year_confidence
        self.mint_confidence = mint_confidence
        
        # Update UI
        if year is not None:
            self.year_label.setText(str(year))
        else:
            self.year_label.setText("Unknown")
        
        if mint is not None:
            self.mint_label.setText(mint)
        else:
            self.mint_label.setText("Unknown")
        
        if year_confidence is not None:
            self.year_confidence_bar.setValue(int(year_confidence * 100))
        else:
            self.year_confidence_bar.setValue(0)
        
        if mint_confidence is not None:
            self.mint_confidence_bar.setValue(int(mint_confidence * 100))
        else:
            self.mint_confidence_bar.setValue(0)
        
        # Add to history
        if year is not None or mint is not None:
            self._add_to_history(image_path, year, mint, year_confidence, mint_confidence)
        
        # Update details
        self._update_details()
    
    def _add_to_history(self, image_path: Optional[str], year: Optional[int], mint: Optional[str],
                       year_confidence: Optional[float], mint_confidence: Optional[float]):
        """
        Add a result to the history.
        
        Args:
            image_path: Path to the image
            year: Recognized year
            mint: Recognized mint mark
            year_confidence: Confidence score for year recognition
            mint_confidence: Confidence score for mint recognition
        """
        # Add to history list
        self.history.append((image_path, year, mint, year_confidence, mint_confidence))
        
        # Add to history table
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        
        # Image path
        image_item = QTableWidgetItem(image_path if image_path else "Unknown")
        self.history_table.setItem(row, 0, image_item)
        
        # Year
        year_item = QTableWidgetItem(str(year) if year is not None else "Unknown")
        self.history_table.setItem(row, 1, year_item)
        
        # Mint
        mint_item = QTableWidgetItem(mint if mint else "Unknown")
        self.history_table.setItem(row, 2, mint_item)
        
        # Year confidence
        year_conf_item = QTableWidgetItem(f"{year_confidence*100:.1f}%" if year_confidence is not None else "N/A")
        self.history_table.setItem(row, 3, year_conf_item)
        
        # Mint confidence
        mint_conf_item = QTableWidgetItem(f"{mint_confidence*100:.1f}%" if mint_confidence is not None else "N/A")
        self.history_table.setItem(row, 4, mint_conf_item)
        
        # Color code by confidence
        if year_confidence is not None:
            self._color_code_item(year_item, year_confidence)
            self._color_code_item(year_conf_item, year_confidence)
        
        if mint_confidence is not None:
            self._color_code_item(mint_item, mint_confidence)
            self._color_code_item(mint_conf_item, mint_confidence)
        
        # Enable export buttons
        self.export_csv_button.setEnabled(True)
        self.export_json_button.setEnabled(True)
        self.clear_history_button.setEnabled(True)
        
        # Select the new row
        self.history_table.selectRow(row)
    
    def _color_code_item(self, item: QTableWidgetItem, confidence: float):
        """
        Color code a table item based on confidence.
        
        Args:
            item: Table item to color
            confidence: Confidence score (0-1)
        """
        if confidence >= 0.9:
            item.setBackground(QBrush(QColor(0, 0, 150)))  # Dark blue
            item.setForeground(QBrush(QColor(255, 255, 255)))  # White text for dark background
        elif confidence >= 0.7:
            item.setBackground(QBrush(QColor(255, 255, 200)))  # Light yellow
        elif confidence >= 0.5:
            item.setBackground(QBrush(QColor(255, 230, 200)))  # Light orange
        else:
            item.setBackground(QBrush(QColor(255, 200, 200)))  # Light red
    
    def _update_details(self):
        """Update the details text."""
        details = ""
        
        if self.year is not None:
            details += f"<h3>Year: {self.year}</h3>\n"
            if self.year_confidence is not None:
                details += f"<p>Confidence: {self.year_confidence*100:.1f}%</p>\n"
        
        if self.mint is not None:
            details += f"<h3>Mint: {self.mint}</h3>\n"
            if self.mint_confidence is not None:
                details += f"<p>Confidence: {self.mint_confidence*100:.1f}%</p>\n"
        
        if self.year is not None and self.mint is not None:
            details += f"<h3>Combined: {self.year} {self.mint}</h3>\n"
            if self.year_confidence is not None and self.mint_confidence is not None:
                combined_confidence = (self.year_confidence + self.mint_confidence) / 2
                details += f"<p>Combined Confidence: {combined_confidence*100:.1f}%</p>\n"
        
        self.details_text.setHtml(details)
    
    def get_history(self) -> List[Tuple[Optional[str], Optional[int], Optional[str], 
                                      Optional[float], Optional[float]]]:
        """
        Get the history of recognition results.
        
        Returns:
            List of tuples of (image_path, year, mint, year_confidence, mint_confidence)
        """
        return self.history
    
    @Slot()
    def _on_clear_history(self):
        """Handle clear history button clicked."""
        # Clear history list
        self.history.clear()
        
        # Clear history table
        self.history_table.setRowCount(0)
        
        # Disable export buttons
        self.export_csv_button.setEnabled(False)
        self.export_json_button.setEnabled(False)
        self.clear_history_button.setEnabled(False)
    
    @Slot()
    def _on_history_selection_changed(self):
        """Handle history selection changed."""
        selected_rows = self.history_table.selectionModel().selectedRows()
        
        if not selected_rows:
            return
        
        # Get the selected row
        row = selected_rows[0].row()
        
        # Get the result from history
        image_path, year, mint, year_confidence, mint_confidence = self.history[row]
        
        # Update current result
        self.year = year
        self.mint = mint
        self.year_confidence = year_confidence
        self.mint_confidence = mint_confidence
        
        # Update UI
        if year is not None:
            self.year_label.setText(str(year))
        else:
            self.year_label.setText("Unknown")
        
        if mint is not None:
            self.mint_label.setText(mint)
        else:
            self.mint_label.setText("Unknown")
        
        if year_confidence is not None:
            self.year_confidence_bar.setValue(int(year_confidence * 100))
        else:
            self.year_confidence_bar.setValue(0)
        
        if mint_confidence is not None:
            self.mint_confidence_bar.setValue(int(mint_confidence * 100))
        else:
            self.mint_confidence_bar.setValue(0)
        
        # Update details
        self._update_details()