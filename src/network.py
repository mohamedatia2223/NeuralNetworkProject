import numpy as np
import random
import tensorflow as tf

class Network:
    def __init__(self, sizes, lambda_=0.0):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Initialize biases with Gaussian (0,1)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Initialize weights with Gaussian (0, 1/sqrt(n))
        self.weights = [np.random.randn(y, x)/np.sqrt(x) 
                       for x, y in zip(sizes[:-1], sizes[1:])]
        self.lambda_ = lambda_  # L2 regularization parameter

    def feedforward(self, a):
        """Return the output of the network for input 'a' (all sigmoid)."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train with SGD and evaluate on test_data if provided."""
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for batch in mini_batches:
                self.update_mini_batch(batch, eta)
            
            if test_data:
                accuracy = self.evaluate(test_data)/len(test_data)*100
                print(f"Epoch {j}: {accuracy:.2f}%")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        """Update weights and biases using backpropagation."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Apply updates with L2 regularization
        self.weights = [(1-eta*self.lambda_/len(mini_batch))*w - (eta/len(mini_batch))*nw
                       for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb 
                      for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return gradients for biases and weights."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Forward pass
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # Backward pass
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])  # Output error
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return number of correct predictions."""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) 
                    for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

# Activation functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
