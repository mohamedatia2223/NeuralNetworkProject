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

## Testing in Colab
If you'd like to run a quick test of the model in a Google Colab environment, you can use the provided notebook `DigitRecognizerTest.ipynb`. This notebook allows you to test the trained model on custom images of handwritten digits.

### How to Use:
1. Click the **Open in Colab** button below to open the notebook directly in Colab.
2. Upload an image of a handwritten digit when prompted.
3. The notebook will predict the digit and display the result along with the image.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohamedatia2223/NeuralNetworkProject/blob/main/GoogleColab/DigitRecognizerTest.ipynb)

---

## Acknowledgments
This implementation is inspired by Michael Nielsen’s *Neural Networks and Deep Learning* book.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This updated section explains how users can use the Colab notebook for testing and provides instructions for testing custom images. Let me know if you'd like further changes!
