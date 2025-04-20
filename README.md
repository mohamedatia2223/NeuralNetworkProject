# Neural Network from Scratch - MNIST Classifier

This project is an implementation of a simple feedforward neural network trained on the MNIST dataset, inspired by Michael Nielsen’s *Neural Networks and Deep Learning*.

## Features
- Implements **Stochastic Gradient Descent (SGD)** with backpropagation.
- Uses **ReLU activation and Softmax activation** and supports **L2 regularization**.
- Trains on the **MNIST dataset** (handwritten digits recognition).
- Runs with only **NumPy** for computations and **TensorFlow** for dataset loading.

## Installation
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/NeuralNetwork-MNIST.git
cd NeuralNetwork-MNIST
```

### 2️⃣ Install Dependencies
Ensure you have Python installed, then install the required packages:
```sh
pip install -r requirements.txt
```

## Usage
### 1️⃣ Train the Neural Network
Run the script to train the network on MNIST:
```sh
python train.py
```

### 2️⃣ Evaluate on Test Data
The script prints test accuracy at each epoch to track model performance.

## Acknowledgments
This implementation is inspired by Michael Nielsen’s *Neural Networks and Deep Learning* book.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
