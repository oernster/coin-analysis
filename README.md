# Coin Recognition Application

A desktop application for recognizing years and mint marks on US cents using computer vision and machine learning techniques.

## Overview

This application is designed to analyze images of US cents and identify the year and mint mark on the right side of the coin. It works with partial coin images and can process both individual images and batches of images.

## Features

- **Image Processing**: Preprocess coin images to enhance features for recognition
- **Feature Extraction**: Extract relevant features from coin images
- **Machine Learning**: Recognize years and mint marks using trained models
- **User Interface**: Intuitive desktop interface built with PySide6
- **Batch Processing**: Process multiple images at once
- **Result Export**: Export recognition results to CSV or JSON formats
- **Image Enhancement**: Tools for adjusting image preprocessing parameters

## Installation

### Prerequisites

- Python 3.8 or higher
- PIP package manager

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/oernster/coin-analysis.git
   cd coin-analysis
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

Run the application using:

```
python run.py
```

### Single Image Processing

1. Click "Open Image" to load a coin image
2. Adjust enhancement parameters if needed
3. Click "Process" to recognize the year and mint mark
4. View results in the results panel

### Batch Processing

1. Switch to the "Batch Processing" tab
2. Click "Select Folder" or "Select Files" to choose images
3. Click "Process" to start batch processing
4. View results in the results table
5. Export results to CSV or JSON if needed

## Image Requirements

- Images should show the right side of US cents
- The year and mint mark should be visible
- Images can be partial (don't need to show the entire coin)
- Supported formats: PNG, JPG, JPEG, BMP, TIF, TIFF

## Project Structure

```
coin-analysis/
├── app/                      # Main application package
│   ├── ui/                   # User interface components
│   ├── processing/           # Image processing modules
│   ├── models/               # Machine learning models
│   └── utils/                # Utility functions
├── data/                     # Data directory
│   ├── raw/                  # Raw training images
│   ├── processed/            # Preprocessed images
│   └── models/               # Saved model weights
├── tests/                    # Test suite
├── resources/                # Application resources
├── requirements.txt          # Dependencies
├── implementation_plan.md    # Implementation plan
├── run.py                    # Application entry point
└── README.md                 # This file
```

## Training Your Own Model

The application comes with a basic model, but you can train your own model using your own dataset:

1. Collect images of US cents with visible years and mint marks
2. Place the images in the `data/raw` directory
3. Use the "Train Model" option in the Tools menu
4. Follow the training wizard to create and train your model

## Future Improvements

- Transfer learning with deep neural networks for improved accuracy
- OCR integration for direct text recognition
- Support for other coin types
- Mobile application version

## License

This project is licensed under the MIT License - see the LICENSE file for details.