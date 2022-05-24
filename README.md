# afrnn_code

This Repository contains the code related to the paper "AFRNN: Stable RNN with Top Down Feedback and Antisymmetry".

## Installation 

1. Install python 3.8.10
2. Install the following packages:

- torch                1.11.0+cu113
- torchaudio           0.11.0+cu113
- torchvision          0.12.0+cu113
- scikit-learn 1.1.0
- scipy 1.8.0
- tqdm
- numpy 1.22
- matplotlib 3.5

## Experiments

The different experiments from the paper can be reproduced with Jupyter Notebooks. The file `models.py` includes the Pytorch model definitions of the
presented architectures.

### pixel-by-pixel mnist

This experiment can be reproduced using the file `pixel_by_pixel_mnist.ipynb`.

### pixel-by-pixel cifar10

This experiment can be reproduced using the file `pixel_by_pixel_cifar10.ipynb`.

### Double Pendulum

This experiment can be reproduced using the file `double_pendulum.ipynb`.

### Jacobian Visualizations

The eigenvalues of the jacobian of the system over time can be calculated using the file `jacobian_plots.ipynb`.

### Trajectories

The trajectories of individual neurons of the system can be visualized with the help of `model_trajectories.ipynb`.

