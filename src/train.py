import numpy as np
import tensorflow as tf
import random
from network import Network  # Import your Network class

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels].reshape(-1, num_classes, 1)  

train_images = train_images.reshape((60000, 784, 1)) / 255.0
test_images = test_images.reshape((10000, 784, 1)) / 255.0

train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)

training_data = list(zip(train_images, train_labels))
test_data = list(zip(test_images, test_labels))

net = Network([784, 128, 64, 10],0.1)  # Deeper architecture
net.SGD(training_data, epochs=15, mini_batch_size=64, eta=0.1, test_data=test_data)

net.save("model.pk1")
