import numpy as np
import pickle

class Network:
    def __init__(self, sizes, lambda_):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)*np.sqrt(2.0/x) 
               for x, y in zip(sizes[:-1], sizes[1:])]
        self.lambda_ = lambda_ 
        self.activation_func  = relu
        self.activation_func_prime = relu_prime
        self.output_activation_func = softmax

    def feedforward(self, a):
        """Return the output of the network for input 'a' (all sigmoid)."""
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = self.activation_func (np.dot(w, a) + b)
        a = self.output_activation_func(np.dot(self.weights[-1],a)+self.biases[-1])
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None,decay_rate= 0.98):
        """Train with SGD and evaluate on test_data if provided."""
        n = len(training_data)
        for j in range(epochs):
            eta *= decay_rate
            np.random.shuffle(training_data)
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

        self.weights = [(1-eta*self.lambda_/len(mini_batch))*w - (eta/len(mini_batch))*nw
                       for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb 
                      for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Forward pass
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights): 
            z = np.dot(w, activation) + b
            zs.append(z)
            if w is self.weights[-1]:
                activation = self.output_activation_func(z)
            else:
                activation = self.activation_func(z)
            activations.append(activation)
        
        delta = (activations[-1] - y) 
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_func_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return number of correct predictions."""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) 
                    for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    import pickle

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'sizes': self.sizes,
                'weights': self.weights,
                'biases': self.biases,
                'lambda_': self.lambda_
            }, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        net = Network(data['sizes'], data['lambda_'])
        net.weights = data['weights']
        net.biases = data['biases']
        return net


# Activation functions
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0,z)

def relu_prime(z):
    return np.where(z>0,1,0)

def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1-(np.tanh(z))**2

def leaky_relu(z,alpha=0.01):
    return np.where(z>0,z,alpha*z)

def leaky_relu_prime(z,alpha = 0.01):
    return np.where(z>0,1,alpha)

def softmax(z):
    shift_z = z - np.max(z, axis=0, keepdims=True)
    exps = np.exp(shift_z)
    return exps / np.sum(exps, axis=0, keepdims=True)
