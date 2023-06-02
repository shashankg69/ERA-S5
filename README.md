# ERA - MNIST Classification using Pytorch

This project demonstrates image classification on the MNIST dataset using PyTorch deep learning library. The MNIST dataset consists of handwritten digit images and their corresponding labels.

## Requirements

- Python (>=3.6)
- PyTorch (>=1.7)
- torchvision
- matplotlib
- tqdm


## Install the Required Packages on your Environment

**For CUDA Enabled Devices**:
```shell
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda install matplotlib tqdm
```
**For CPU Only Devices**:
```shell
conda install pytorch torchvision cpuonly -c pytorch
conda install matplotlib tqdm
```
## Install Jupyter Notebook (if not already installed):
```
conda install jupyter
```
## Project Structure
 **The project has the following structure**:

1. [`S5.ipynb`](https://github.com/Shashank-Gottumukkala/ERA/blob/main/S5.ipynb) : This is the main notebook where you execute the code. It imports functions and classes from other files and contains the main logic for training and testing the model.
   - `train_transforms` : A data transformation pipeline for the training data.
   - `test_transforms` : A data transformation pipeline for the test data.

2. [`utils.py`](https://github.com/Shashank-Gottumukkala/ERA/blob/main/utils.py) : This file contains utility functions and data loading functions that are commonly used across different modules. It provides the following functionalities:

   - `train`: A function for training the model.
   - `test` : A function for testing the model.


3. [`model.py`](https://github.com/Shashank-Gottumukkala/ERA/blob/main/model.py) : This file contains the definitions of the neural network models. It provides the following classes:
   - `Net`: Neural network model with a specific architecture. 


## Usage
1. Open the `S5.ipynb` file in Jupyter Notebook or any compatible environment.

2. Execute the code cells in the notebook to train the model, evaluate it on the test set, and visualize the results.



## Code Explanation

The provided code trains a neural network model on the MNIST dataset using PyTorch. It performs the following steps:

 1. Imports necessary libraries and modules, including PyTorch, torch.optim, utils, and model.

 2. **Checking CUDA Availability** :
    - The code checks if CUDA is available for GPU acceleration by calling `torch.cuda.is_available()` . The result is stored in the cuda variable.
 
 3. **Data Loading and Transformation**:
    - The code defines the data transformations for the training and test datasets using `train_transforms` and `test_transforms`.
    - The MNIST dataset is loaded using these transformations, and data loaders are created for both the training and test datasets.

 4. **Model Definition**:
    - The code defines the neural network model architecture using the `Net` class from the model module.
    - The model is moved to the available device (GPU if CUDA is available, otherwise CPU).
    - A summary of the model architecture is printed using the `summary` function from the `torchsummary` module.

 5. **Training Setup**:
    - The code sets up the optimizer (`Adam` optimizer), learning rate scheduler (step scheduler), loss criterion (`F.nll_loss`), and the number of epochs for training.
 
 6. **Training Loop**:
    - The code initiates the training loop for the specified number of epochs.
    - In each epoch, the `train` function from the `utils` module is called to train the model on the training data.
    - After training, the `test` function from the `utils` module is called to evaluate the model on the test data.
    - The learning rate scheduler is then adjusted accordingly.
