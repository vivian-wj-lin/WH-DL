import numpy as np


def relu(x):
    return np.maximum(0, x)


def linear(x):
    return x


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def binary_cross_entropy(predictions, targets):
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.sum(
        targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)
    )


def categorical_cross_entropy(predictions, targets):
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.sum(targets * np.log(predictions))


class Network:
    def __init__(self, layer_weights, layer_biases, activations):
        self.layer_weights = [np.array(w) for w in layer_weights]
        self.layer_biases = [np.array(b) for b in layer_biases]
        self.activations = activations
        self.num_layers = len(layer_weights)

    def forward(self, inputs):
        outputs = np.array(inputs)
        for i in range(self.num_layers):
            outputs = np.dot(outputs, self.layer_weights[i]) + self.layer_biases[i]
            outputs = self.activations[i](outputs)
        return outputs


print("----- Model 1 -----")
layer_weights_1 = [[[0.5, 0.6], [0.2, -0.6]], [[0.8, 0.4], [-0.5, 0.5]]]
layer_biases_1 = [[0.3, 0.25], [0.6, -0.25]]
activations_1 = [relu, linear]
nn1 = Network(layer_weights_1, layer_biases_1, activations_1)

outputs1 = nn1.forward([1.5, 0.5])
expects1 = np.array([0.8, 1])
print(f"Outputs: {outputs1}")
print(f"Total Loss: {mse(outputs1, expects1)}")
outputs2 = nn1.forward([0, 1])
expects2 = np.array([0.5, 0.5])
print(f"Outputs: {outputs2}")
print(f"Total Loss: {mse(outputs2, expects2)}")

print("----- Model 2 -----")
layer_weights_2 = [[[0.5, 0.6], [0.2, -0.6]], [[0.8], [0.4]]]
layer_biases_2 = [[0.3, 0.25], [-0.5]]
activations_2 = [relu, sigmoid]
nn2 = Network(layer_weights_2, layer_biases_2, activations_2)

outputs1 = nn2.forward([0.75, 1.25])
expects1 = 1
print(f"Output: {outputs1}")
print(f"Total Loss: {binary_cross_entropy(outputs1, expects1)}")

outputs2 = nn2.forward([-1, 0.5])
expects2 = 0
print(f"Output: {outputs2}")
print(f"Total Loss: {binary_cross_entropy(outputs2, expects2)}")

print("----- Model 3 -----")
layer_weights_3 = [[[0.5, 0.6], [0.2, -0.6]], [[0.8, 0.5, 0.3], [-0.4, 0.4, 0.75]]]
layer_biases_3 = [[0.3, 0.25], [0.6, 0.5, -0.5]]
activations_3 = [relu, sigmoid]
nn3 = Network(layer_weights_3, layer_biases_3, activations_3)

outputs1 = nn3.forward([1.5, 0.5])
expects1 = np.array([1, 0, 1])
print(f"Output: {outputs1}")
print(f"Total Loss: {binary_cross_entropy(outputs1, expects1)}")

outputs2 = nn3.forward([0, 1])
expects2 = np.array([1, 1, 0])
print(f"Output: {outputs2}")
print(f"Total Loss: {binary_cross_entropy(outputs2, expects2)}")

print("----- Model 4 -----")
layer_weights_4 = [[[0.5, 0.6], [0.2, -0.6]], [[0.8, 0.5, 0.3], [-0.4, 0.4, 0.75]]]
layer_biases_4 = [[0.3, 0.25], [0.6, 0.5, -0.5]]
activations_4 = [relu, softmax]
nn4 = Network(layer_weights_4, layer_biases_4, activations_4)

outputs1 = nn4.forward([1.5, 0.5])
expects1 = np.array([1, 0, 0])
print(f"Output: {outputs1}")
print(f"Total Loss: {categorical_cross_entropy(outputs1, expects1)}")

outputs2 = nn4.forward([0, 1])
expects2 = np.array([0, 0, 1])
print(f"Output: {outputs2}")
print(f"Total Loss: {categorical_cross_entropy(outputs2, expects2)}")
