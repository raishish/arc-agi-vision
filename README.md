# arc-agi-vision

A vision-based approach for ARC-AGI.

## Table of Contents

- [Introduction](#introduction)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project implements a vision-based approach for the ARC-AGI challenge. It includes various vision-based models such as PixelCNN, VAE, GAN, UNet, and Segmenter. The goal is to evaluate the performance of vision-based models on the challenge.

**Note: All models are trained and evaluated on the public training and evaluation datasets uploaded at [fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI/tree/master/data).**

## Models

| Status| Model                                          | Grid Accuracy | Foreground Accuracy | Background Accuracy | mIOU |
|-------|------------------------------------------------|---------------|---------------------|---------------------|------|
| ✅    | [PixelCNN](https://arxiv.org/abs/1601.06759)   |               |                     |                     |      |
| ⬜    | [PixelSNAIL](https://arxiv.org/abs/1712.09763) |               |                     |                     |      |
| ⬜    | [VAE](https://arxiv.org/abs/1312.6114)         |               |                     |                     |      |
| ⬜    | [GAN](https://arxiv.org/abs/1406.2661)         |               |                     |                     |      |
| ⬜    | [UNet](https://arxiv.org/abs/1505.04597)       |               |                     |                     |      |
| ✅    | [Segmenter](https://arxiv.org/abs/2105.05633)  |               |                     |                     |      |

where
- `Grid Accuracy`: the percentage of grids correctly predicted
- `Foreground Accuracy`: the percentage of foreground pixels correctly predicted
- `Background Accuracy`: the percentage of background pixels correctly predicted
- `mIOU`: the mean Intersection over Union (IoU) score

## Installation

To get started, clone the repository and install the required dependencies:

```sh
git clone https://github.com/yourusername/arc-agi-vision.git
cd arc-agi-vision
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```

For wandb support, you will also need to install the `wandb` package and log in to your account:

```sh
pip install wandb

wandb login
```

## Usage

### Training

To train a model, you can use the `train.ipynb` notebook. This notebook includes code for loading data, training models, and evaluating their performance.

### Evaluation

To evaluate the models, use the eval_model function in the utils.py file. This function calculates various metrics such as average loss, pixel accuracy, grid accuracy, background accuracy, and foreground accuracy.

Example usage:

```python
from utils import eval_model
from segmenter import PixelCNN

# Load your model and dataloader
model = PixelCNN(...)
dataloader = ...

# Evaluate the model
metrics = eval_model(model, dataloader)
print(metrics)
```

### Plotting

To plot the predicted grids, use the plot_batch function in the utils.py file. This function takes in a list of models and a batch of inputs and targets, and plots the predicted grids for each model.

Example usage:

```python
from utils import plot_batch
from segmenter import PixelCNN

# Load your model and dataloader
model = PixelCNN(...)
dataloader = ...

# Plot the predicted grids
plot_batch([model], dataloader)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
