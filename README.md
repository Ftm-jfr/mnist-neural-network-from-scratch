# MNIST Neural Network from Scratch

A fully connected neural network for handwritten digit recognition (MNIST) implemented **from scratch with NumPy**: forward pass, backpropagation, optimizers, regularization, and training loop, with no TensorFlow or PyTorch. scikit-learn is used only for the train/validation split, and matplotlib for plots.

The main model reaches **97.69% test accuracy** on the 10,000-image MNIST test set.

> Course project for **Fundamentals of Computational Intelligence**, University of Isfahan.

## Features

- **Network architecture:** configurable number of layers and neurons, with a softmax output layer.
- **Activation functions:** ReLU, Sigmoid, Tanh (and Softmax for the output).
- **Weight initialization:** He initialization for ReLU, Xavier initialization for Sigmoid/Tanh.
- **Loss functions:** cross-entropy and mean squared error.
- **Optimizers:** mini-batch SGD and **Adam**, implemented by hand.
- **Regularization:** **Dropout**, plus **early stopping** on validation loss.
- **Utilities:** save/load model weights (`.npz`), training curves, confusion matrix, and misclassified-example plots.
- **Experiments:** architecture comparison, hyperparameter analysis, and Adam vs SGD.

## Dataset

The notebook uses MNIST in CSV format (60,000 training and 10,000 test images; first column is the label, followed by 784 pixel values).

I uploaded the dataset to Kaggle as a dataset of my own, and the notebook reads it from there by default:
`/kaggle/input/datasets/ftmjfr/mnistdataset/` (files `mnist_train.csv` and `mnist_test.csv`).

Preprocessing: images are flattened to 784 features, pixel values are scaled to [0, 1], and 10% of the training set (stratified) is held out for validation.

## Setup

**On Kaggle:** create a notebook, add the MNIST CSV dataset as input (see above), upload `mnist_neural_network.ipynb`, and run all cells.


## Model and results

Main model: `784 → 256 → 128 → 10`, ReLU, Adam (lr = 0.001), batch size 64, up to 50 epochs with early stopping.

| Experiment | Test accuracy |
|---|---|
| **Main model** (256-128, ReLU, Adam, full training set) | **97.69%** (train 99.36%) |
| Dropout model (rate 0.3, 20k training samples) | 95.04% |
| SGD baseline (128-64, 20k samples) | 94.74% |
| Adam (128-64, 20k samples) | 96.58% |

Architecture comparison (Adam, 20k training samples, up to 20 epochs):

| Architecture | Activation | Test accuracy |
|---|---|---|
| 784-32-10 | ReLU | 95.16% |
| 784-128-10 | ReLU | 96.79% |
| 784-256-128-10 | ReLU | 96.80% |
| 784-256-128-10 | Sigmoid | 96.92% |
| 784-256-128-10 | Tanh | 97.04% |
| 784-512-256-10 | ReLU | 97.21% |

Hyperparameter analysis (Adam, 10k training samples, 15 epochs) covers learning rate, hidden-layer width, activation, and batch size. Highlights: lr = 0.001 worked best (95.49%), while lr = 0.1 diverged to 84.74%; wider layers helped up to about 256 neurons; very large batches (1024) hurt accuracy (93.62%).

Note: the smaller experiments train on a subset of the data for speed, so their numbers are not comparable to the main model.

## Repository contents

| File | Description |
|---|---|
| `mnist_neural_network.ipynb` | Full implementation, experiments, and outputs |
| `best_model_weights.npz` | Saved weights of the main model |
| `requirements.txt` | Python dependencies |

## Author

[@Ftm-jfr](https://github.com/Ftm-jfr)
