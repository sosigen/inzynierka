# Facial Expression Recognition (FER) Research Project

This repository contains code and resources for a engineer's thesis focused on Facial Expression Recognition (FER) using various deep learning models.

# Project Structure

## Usage
Modules needed to train a model are split between individual files based on the purpose they serve. To train a model, you need to:
1. Unzip dataset (optional, depending on what source do you use)
2. Prepare generators for train, validation and test
3. Train the model
4. Generate metrics for the trained model

If you are using Google Colab or a jupyter notebook it's as simple as adding the scripts as a separate cells in the correct order. Some architectures require different input shape and this fact has to be reflected in the copy of fer2013plus_data_load. It can be easily determined by looking at a model input layer's shape. Possible shapes are: (48, 48, 1), (48, 48, 3) and (224, 224, 3).

## Models (src/models)

This directory holds code responsible for creating models (3.). There are 3 architectures available:

1. Convolutional Neural Networks (CNN)
   - Raw implementations
   - Transfer learning approaches
2. Graph Convolutional Neural Networks (GCNN)
   - Raw Implementations
3. Transformers
   - Raw implementations
   - Transfer learning approaches (ViT, DeiT, and Swin)

## Video Processing (src/video)
This directory contains files necessary for the actual usage of the models. 2 python scripts able to annotate either a input video (annotate_video_with_FER.py) or live camera feed (annotate_camera_with_FER.py) with model's classifications. The scripts require haarcascade XML file and .keras or .h5 model file. Requirements needed for those scripts to run are defined in requirements.txt. To install run:
```shell pip install -r ./src/video/requirements.txt```

## Utils
The project uses the FER2013+ dataset for training and evaluation. Data loading utilities can be found in `src/utils/fer2013plus_data_load.ipynb`.

## Utilities

- `gradient_logger.ipynb`: Utility for logging gradient information during training
- `metrics_generator.ipynb`: Tool for generating performance metrics based on model's training history and validation run (4.)
- `unzip.ipynb`: Utility for unzipping dataset files (1.)
