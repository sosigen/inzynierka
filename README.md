# Facial Expression Recognition (FER) Research Project

This repository contains code and resources for a engineer's thesis focused on Facial Expression Recognition (FER) using various deep learning models.

## Project Structure

```
src/
├── models/
│   ├── CNN/
│   │   ├── raw/
│   │   └── transfer/
│   ├── GCNN/
│   └── transformer/
│       ├── raw/
│       └── transfer/
├── utils/
└── video/
```

## Models

The project explores different types of deep learning models for FER:

1. Convolutional Neural Networks (CNN)
   - Raw implementations
   - Transfer learning approaches
2. Graph Convolutional Neural Networks (GCNN)
   - Raw Implementations
3. Transformers
   - Raw implementations
   - Transfer learning approaches (ViT, DeiT, and Swin)

## Dataset

The project uses the FER2013+ dataset for training and evaluation. Data loading utilities can be found in `src/utils/fer2013plus_data_load.ipynb`.

## Utilities

- `gradient_logger.ipynb`: Utility for logging gradient information during training
- `metrics_generator.ipynb`: Tool for generating performance metrics based on model's training history and validation run
- `unzip.ipynb`: Utility for unzipping dataset files

## Video Processing

The `src/video/` directory contains scripts and resources for annotating videos with FER predictions:

- `annotate_video_with_FER.py`: Script for applying FER models to video input

## Requirements

For a list of required packages and dependencies, please refer to the `requirements.txt` file in the root directory.

## Usage

To use this project:

1. Clone the repository
2. Install the required dependencies: `pip install -r requirements.txt`
3. Explore the Jupyter notebooks in the `src/models/` directory to train and evaluate different FER models
4. Use the `src/video/annotate_video_with_FER.py` script to apply trained models to video input
