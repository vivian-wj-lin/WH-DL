import numpy as np


def relu(x):
    return np.maximum(0, x)


def linear(x):
    return x


def softmax(x):
    exp_x = np.exp(x - np.max(x))
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


def relu_derivative(z):
    return np.where(z > 0, 1, 0)


def linear_derivative(z):
    return np.ones_like(z)


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


class Network:
    def __init__(self, layer_weights, layer_biases, activations):
        self.layer_weights = [np.array(w) for w in layer_weights]
        self.layer_biases = [np.array(b) for b in layer_biases]
        self.activations = activations
        self.num_layers = len(layer_weights)

        self.weight_gradients = [None] * self.num_layers
        self.bias_gradients = [None] * self.num_layers

        self.layer_inputs = []
        self.layer_outputs = []
        self.activated_outputs = []

    def forward(self, inputs):
        self.layer_inputs = []
        self.layer_outputs = []
        self.activated_outputs = []

        current_input = np.array(inputs)

        for i in range(self.num_layers):
            self.layer_inputs.append(current_input.copy())

            z = np.dot(self.layer_weights[i], current_input) + self.layer_biases[i]
            self.layer_outputs.append(z)

            activated = self.activations[i](z)
            self.activated_outputs.append(activated)

            current_input = activated

        return current_input

    def backward(self, output_gradients):
        #  ∂E_total/∂out（鍊式法則第一部分）
        grad = np.array(output_gradients)

        for i in reversed(range(self.num_layers)):
            z = self.layer_outputs[i]
            activation_name = self.activations[i].__name__

            # activation_grad = ∂out/∂net（鍊式法則第二部分）
            if activation_name == "relu":
                activation_grad = np.where(z > 0, 1, 0)
            elif activation_name == "linear":
                activation_grad = np.ones_like(z)
            elif activation_name == "sigmoid":
                s = sigmoid(z)
                activation_grad = s * (1 - s)

            grad = grad * activation_grad
            layer_input = self.layer_inputs[i]

            # ∂net/∂w（鍊式法則第三部分）
            self.weight_gradients[i] = np.outer(grad, layer_input)

            # ∂net_o1/∂b_2
            self.bias_gradients[i] = grad.copy()

            if i > 0:
                grad = np.dot(self.layer_weights[i].T, grad)

    def zero_grad(self, learning_rate):
        for i in range(self.num_layers):
            self.layer_weights[i] -= learning_rate * self.weight_gradients[i]
            self.layer_biases[i] -= learning_rate * self.bias_gradients[i]

        self.weight_gradients = [None] * self.num_layers
        self.bias_gradients = [None] * self.num_layers


class MSE:
    def get_total_loss(self, predictions, targets):
        return np.mean((predictions - targets) ** 2)

    def get_output_gradients(self, predictions, targets):
        n = len(predictions)
        return (2 / n) * (predictions - targets)


class BinaryCrossEntropy:
    def get_total_loss(self, predictions, targets):
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.sum(
            targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)
        )

    def get_output_gradients(self, predictions, targets):
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -targets / predictions + (1 - targets) / (1 - predictions)


# =======================================
print("----- Model 1 -----")
model1_weights = [[[0.5, 0.2], [0.6, -0.6]], [[0.8, -0.5]], [[0.6], [-0.3]]]
model1_biases = [[0.3, 0.25], [0.6], [0.4, 0.75]]
model1_activations = [relu, linear, linear]
model1_expects = np.array([0.8, 1])
model1_loss_fn = MSE()
model1_learning_rate = 0.01
model1_inputs = [1.5, 0.5]

print("----- Task 1 -----")
nn = Network(model1_weights, model1_biases, model1_activations)

outputs = nn.forward(model1_inputs)
loss = model1_loss_fn.get_total_loss(outputs, model1_expects)
output_gradients = model1_loss_fn.get_output_gradients(outputs, model1_expects)

nn.backward(output_gradients)
nn.zero_grad(model1_learning_rate)

print(f"Layer 0:")
print(nn.layer_weights[0])
print(nn.layer_biases[0])
print(f"Layer 1:")
print(nn.layer_weights[1])
print(nn.layer_biases[1])
print(f"Layer 2:")
print(nn.layer_weights[2])
print(nn.layer_biases[2])

print("----- Task 2 -----")
nn = Network(model1_weights, model1_biases, model1_activations)

for i in range(1000):
    outputs = nn.forward(model1_inputs)
    loss = model1_loss_fn.get_total_loss(outputs, model1_expects)
    output_gradients = model1_loss_fn.get_output_gradients(outputs, model1_expects)
    # print(f"Iteration {i+1}: Loss = {loss}")
    nn.backward(output_gradients)
    nn.zero_grad(model1_learning_rate)

print(f"Final loss: {loss}")

# =======================================
print("\n----- Model 2 -----")
model2_weights = [[[0.5, 0.2], [0.6, -0.6]], [[0.8, 0.4]]]
model2_biases = [[0.3, 0.25], [-0.5]]
model2_activations = [relu, sigmoid]
model2_expects = np.array([1])
model2_loss_fn = BinaryCrossEntropy()
model2_learning_rate = 0.1
model2_inputs = [0.75, 1.25]

print("----- Task 1 -----")
nn = Network(model2_weights, model2_biases, model2_activations)

outputs = nn.forward(model2_inputs)
loss = model2_loss_fn.get_total_loss(outputs, model2_expects)
output_gradients = model2_loss_fn.get_output_gradients(outputs, model2_expects)

nn.backward(output_gradients)
nn.zero_grad(model2_learning_rate)

print(f"Layer 0:")
print(nn.layer_weights[0])
print(nn.layer_biases[0])
print(f"Layer 1:")
print(nn.layer_weights[1])
print(nn.layer_biases[1])

print("----- Task 2 -----")
nn = Network(model2_weights, model2_biases, model2_activations)

for i in range(1000):
    outputs = nn.forward(model2_inputs)
    loss = model2_loss_fn.get_total_loss(outputs, model2_expects)
    output_gradients = model2_loss_fn.get_output_gradients(outputs, model2_expects)
    # print(f"Iteration {i+1}: Loss = {loss}")
    nn.backward(output_gradients)
    nn.zero_grad(model2_learning_rate)

print(f"Final loss: {loss}")
