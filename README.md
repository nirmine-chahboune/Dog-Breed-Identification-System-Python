Dog Breed Identification using PyTorch

This repository contains Python scripts for a dog breed identification project using PyTorch. The project focuses on training a deep learning model (ResNet-18) to classify dog breeds based on images. Here's a breakdown of what you'll find:

Data Handling: Loads dog breed images from the provided dataset using PyTorch's DataLoader and handles data transformations.
Model Definition: Utilizes a pre-trained ResNet-18 model and fine-tunes the final fully connected layer for dog breed classification.
Training: Implements a training pipeline that includes epochs, learning rate adjustment, and early stopping based on validation accuracy.
Evaluation: Evaluates the trained model on a test set and provides accuracy metrics.
Visualization: Plots training and validation accuracy/loss curves to monitor model performance over epochs.
Prediction: Includes a function for quick test predictions on sample images from the dataset.
Usage
To use this repository, ensure you have Python 3 and PyTorch installed. Clone the repository and follow the instructions in the Jupyter notebook or Python scripts to train and evaluate the model.

Requirements

Python 3

PyTorch

torchvision

pandas

numpy

matplotlib

scikit-image

Dataset

The dataset used is the dog breed identification dataset, which includes images of various dog breeds labeled with their respective classes.

Credits

This project was developed as part of a machine learning course and leverages the torchvision library for deep learning tasks.
