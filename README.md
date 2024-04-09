# Image-Based Regression Analysis with Neural Networks

This repository explores regression analysis on images using the ResNet50 model with different loss functions: Mean Absolute Error (MAE), Mean Squared Error (MSE), and Huber Loss. The goal is to determine how these loss functions affect the accuracy of predicting correlation coefficients from images.

## Files Description
- `train_MAE.py`: Trains a model using MAE loss.
- `train_MSE.py`: Trains a model using MSE loss.
- `train_huber.py`: Trains a model using Huber loss.
- `test.py`: Evaluates the model on a test dataset.
- `split_data.py`: Splits the dataset into training, validation, and testing sets.
- `test_1image.py`: Demonstrates model prediction on a single image.
- `*.png`: Contains loss visualization graphs.

## Getting Started

```bash
pip install -r requirements.txt
python train_MAE.py # Or any other training script
python test.py
