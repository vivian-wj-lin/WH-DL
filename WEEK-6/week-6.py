import numpy as np
import pandas as pd
import torch


def relu(x):
    return np.maximum(0, x)


def linear(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Network:
    def __init__(self, layer_sizes, activations):
        self.activations = activations
        self.num_layers = len(layer_sizes) - 1

        self.layer_weights = []
        self.layer_biases = []

        for i in range(self.num_layers):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            activation = activations[i]

            if activation == relu:
                # He Initialization
                weights = np.random.randn(output_size, input_size) * np.sqrt(
                    2.0 / input_size
                )
            else:
                # Xavier/Glorot Initialization
                weights = np.random.randn(output_size, input_size) * np.sqrt(
                    1.0 / input_size
                )

            self.layer_weights.append(weights)
            biases = np.zeros(output_size)
            self.layer_biases.append(biases)

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
        grad = np.array(output_gradients)

        for i in reversed(range(self.num_layers)):
            z = self.layer_outputs[i]
            activation_name = self.activations[i].__name__

            if activation_name == "relu":
                activation_grad = np.where(z > 0, 1, 0)
            elif activation_name == "linear":
                activation_grad = np.ones_like(z)
            elif activation_name == "sigmoid":
                s = sigmoid(z)
                activation_grad = s * (1 - s)

            grad = grad * activation_grad
            layer_input = self.layer_inputs[i]

            self.weight_gradients[i] = np.outer(grad, layer_input)
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
print("------ Task 1 ------")
np.random.seed(42)
# architecture：2  → 16 (relu) → 8 (relu) → 1 (linear)

df = pd.read_csv("gender-height-weight.csv")
df["Gender"] = df["Gender"].map({"Male": -1, "Female": 1})

indices = np.random.permutation(len(df))
split_idx = int(len(df) * 0.8)
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

df_train = df.iloc[train_indices].copy()
df_test = df.iloc[test_indices].copy()

height_mean = df_train["Height"].mean()
height_std = df_train["Height"].std()
weight_mean = df_train["Weight"].mean()
weight_std = df_train["Weight"].std()

df_train["Height_std"] = (df_train["Height"] - height_mean) / height_std
df_train["Weight_std"] = (df_train["Weight"] - weight_mean) / weight_std

df_test["Height_std"] = (df_test["Height"] - height_mean) / height_std
df_test["Weight_std"] = (df_test["Weight"] - weight_mean) / weight_std

xs_train = df_train[["Gender", "Height_std"]].values
es_train = df_train["Weight_std"].values.reshape(-1, 1)

xs_test = df_test[["Gender", "Height_std"]].values
es_test = df_test["Weight_std"].values.reshape(-1, 1)

loss_fn = MSE()
learning_rate = 0.01
epochs = 500

nn = Network(layer_sizes=[2, 16, 8, 1], activations=[relu, relu, linear])

print("------ Start Training ------")
for epoch in range(epochs):
    total_loss = 0

    for x, e in zip(xs_train, es_train):
        output = nn.forward(x)
        loss = loss_fn.get_total_loss(output, e)
        total_loss += loss
        output_gradients = loss_fn.get_output_gradients(output, e)

        nn.backward(output_gradients)
        nn.zero_grad(learning_rate)

    if (epoch + 1) % 100 == 0 or epoch + 1 == epochs:
        avg_loss = total_loss / len(xs_train)
        print(f"Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}")

print("------ Start Evaluating ------")
loss_sum = 0
for x, e in zip(xs_train, es_train):
    output = nn.forward(x)
    loss = loss_fn.get_total_loss(output, e)
    loss_sum += loss

avg_loss = loss_sum / len(xs_train)
train_rmse = np.sqrt(avg_loss) * weight_std

loss_sum = 0
for x, e in zip(xs_test, es_test):
    output = nn.forward(x)
    loss = loss_fn.get_total_loss(output, e)
    loss_sum += loss

avg_loss = loss_sum / len(xs_test)
test_rmse = np.sqrt(avg_loss) * weight_std

print(f"Training Set:")
print(f"RMSE: {train_rmse:.2f} lbs")

print(f"Test Set:")
print(f"RMSE: {test_rmse:.2f} lbs")

# # =======================================
print("------ Task 2 ------")
np.random.seed(42)
# Survived: 0 = No, 1 = Yes
# architecture: 6 -> 16 (ReLU) -> 8 (ReLU) -> 1 (Sigmoid)

df = pd.read_csv("titanic.csv")
# print(f"Missing Age count: {df['Age'].isna().sum()}")
df = df.dropna(subset=["Age"])

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

indices = np.random.permutation(len(df))
split_idx = int(len(df) * 0.8)
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

df_train = df.iloc[train_indices].copy()
df_test = df.iloc[test_indices].copy()

age_mean = df_train["Age"].mean()
age_std = df_train["Age"].std()
fare_mean = df_train["Fare"].mean()
fare_std = df_train["Fare"].std()

features_train = df_train[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].copy()
features_train["Age"] = (features_train["Age"] - age_mean) / age_std
features_train["Fare"] = (features_train["Fare"] - fare_mean) / fare_std

xs_train = features_train.values
es_train = df_train["Survived"].values.reshape(-1, 1)

features_test = df_test[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].copy()
features_test["Age"] = (features_test["Age"] - age_mean) / age_std
features_test["Fare"] = (features_test["Fare"] - fare_mean) / fare_std

xs_test = features_test.values
es_test = df_test["Survived"].values.reshape(-1, 1)

loss_fn = BinaryCrossEntropy()
learning_rate = 0.003
epochs = 1200

nn = Network(layer_sizes=[6, 16, 8, 1], activations=[relu, relu, sigmoid])

print("------ Start Training ------")
for epoch in range(epochs):
    total_loss = 0

    for x, e in zip(xs_train, es_train):
        output = nn.forward(x)
        loss = loss_fn.get_total_loss(output, e)
        total_loss += loss
        output_gradients = loss_fn.get_output_gradients(output, e)

        nn.backward(output_gradients)
        nn.zero_grad(learning_rate)

    if (epoch + 1) % 100 == 0 or epoch + 1 == epochs:
        avg_loss = total_loss / len(xs_train)
        print(f"Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}")


print("------ Start Evaluating ------")
threshold = 0.5
train_correct = 0
for x, e in zip(xs_train, es_train):
    output = nn.forward(x)
    survival_status = 1 if output[0] > threshold else 0
    if survival_status == e[0]:
        train_correct += 1

train_accuracy = train_correct / len(xs_train)

test_correct = 0
for x, e in zip(xs_test, es_test):
    output = nn.forward(x)
    survival_status = 1 if output[0] > threshold else 0
    if survival_status == e[0]:
        test_correct += 1

test_accuracy = test_correct / len(xs_test)

print(f"Training Set:")
print(f"Correct: {train_correct}/{len(xs_train)}")
print(f"Accuracy: {train_accuracy * 100:.2f}%")

print(f"Test Set:")
print(f"Correct: {test_correct}/{len(xs_test)}")
print(f"Accuracy: {test_accuracy * 100:.2f}%")

# =======================================
print("----- Task 3-1 -----")
data_list = [[2, 3, 1], [5, -2, 1]]
tensor1 = torch.tensor(data_list)
print(tensor1.shape)
print(tensor1.dtype)

print("----- Task 3-2 -----")
tensor2 = torch.rand(3, 4, 2)
print(tensor2.shape)
print(tensor2)

print("----- Task 3-3 -----")
tensor3 = torch.ones(2, 1, 5)
print(tensor3.shape)
print(tensor3)

print("----- Task 3-4 -----")
matrix_a = torch.tensor([[1, 2, 4], [2, 1, 3]])
matrix_b = torch.tensor([[5], [2], [1]])
result_task3_4 = torch.matmul(matrix_a, matrix_b)
print(result_task3_4)

print("----- Task 3-5 -----")
matrix_c = torch.tensor([[1, 2], [2, 3], [-1, 3]])
matrix_d = torch.tensor([[5, 4], [2, 1], [1, -5]])
result_task3_5 = matrix_c * matrix_d
print(result_task3_5)
