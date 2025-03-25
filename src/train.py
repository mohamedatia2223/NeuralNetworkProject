import numpy as np
import tensorflow as tf
import random
from network import Network  # Import your Network class

# Load and prepare MNIST data
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape and normalize
train_images = train_images.reshape((60000, 784, 1)) / 255.0
test_images = test_images.reshape((10000, 784, 1)) / 255.0

# One-hot encode labels
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels].reshape(-1, num_classes, 1)

train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)

# Prepare data as tuples
training_data = list(zip(train_images, train_labels))
test_data = list(zip(test_images, test_labels))

# Initialize and train the network
net = Network([784, 30, 10])  # 784 input neurons, 30 hidden, 10 output
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
