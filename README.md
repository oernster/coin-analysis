# Coin Recognition Application

[Friendly coffee donation here](https://www.paypal.com/ncp/payment/Z36XJEEA4MNV6) 

A desktop application for recognizing years and mint marks on US cents using computer vision and machine learning techniques.

<img width="1758" height="982" alt="{BCC4B92F-0458-4507-BBCD-B19E4C25BB4E}" src="https://github.com/user-attachments/assets/fbe8cf7a-c538-4709-b56d-f238ede69170" />

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

## Documentation

- [TECH_DEBT.md](TECH_DEBT.md): what is still open, what is deliberately left and what only
  looks like debt, including the settled decision on this project's direction.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE)
file for details.
