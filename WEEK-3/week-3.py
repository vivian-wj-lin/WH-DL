# class Network:
#     def __init__(self, layer_weights, layer_biases):
#         self.layer_weights = layer_weights
#         self.layer_biases = layer_biases
#         self.num_layers = len(layer_weights)

#     def forward(self, inputs):
#         current_output = inputs

#         for layer_idx in range(self.num_layers):
#             next_output = []

#             weights = self.layer_weights[layer_idx]
#             biases = self.layer_biases[layer_idx]

#             num_neurons = len(biases)

#             for neuron_idx in range(num_neurons):
#                 neuron_value = biases[neuron_idx]
#                 for input_idx in range(len(current_output)):
#                     weight = weights[input_idx][neuron_idx]
#                     neuron_value += current_output[input_idx] * weight

#                 next_output.append(neuron_value)

#             current_output = next_output

#         return current_output


# print("----- Model 1 -----")
# weights_layer1_t1 = [[0.5, 0.6], [0.2, -0.6]]
# bias_layer1_t1 = [0.3, 0.25]

# weights_layer2_t1 = [[0.8], [0.4]]
# bias_layer2_t1 = [-0.5]

# layer_weights_1 = [weights_layer1_t1, weights_layer2_t1]
# layer_biases_1 = [bias_layer1_t1, bias_layer2_t1]

# nn1 = Network(layer_weights_1, layer_biases_1)
# outputs1 = nn1.forward([1.5, 0.5])
# print(f"test 1: {outputs1}")

# outputs2 = nn1.forward([0, 1])
# print(f"test 1: {outputs2}")


# print("----- Model 2 -----")
# weights_layer1_t2 = [[0.5, 0.6], [1.5, -0.8]]
# bias_layer1_t2 = [0.3, 1.25]

# weights_layer2_t2 = [[0.6], [-0.8]]
# bias_layer2_t2 = [0.3]

# weights_layer3_t2 = [[0.5, -0.4]]
# bias_layer3_t2 = [0.2, 0.5]

# layer_weights_2 = [weights_layer1_t2, weights_layer2_t2, weights_layer3_t2]
# layer_biases_2 = [bias_layer1_t2, bias_layer2_t2, bias_layer3_t2]

# nn2 = Network(layer_weights_2, layer_biases_2)

# outputs3 = nn2.forward([0.75, 1.25])
# print(f"test 1: {outputs3}")

# outputs4 = nn2.forward([-1, 0.5])
# print(f"test 2: {outputs4}")

import numpy as np


class Network:
    def __init__(self, layer_weights, layer_biases):
        self.layer_weights = [np.array(w) for w in layer_weights]
        self.layer_biases = [np.array(b) for b in layer_biases]
        self.num_layers = len(layer_weights)

    def forward(self, inputs):
        outputs = np.array(inputs)
        for i in range(self.num_layers):
            outputs = np.dot(outputs, self.layer_weights[i]) + self.layer_biases[i]
        return outputs


print("----- Model 1 -----")
layer_weights_1 = [[[0.5, 0.6], [0.2, -0.6]], [[0.8], [0.4]]]
layer_biases_1 = [[0.3, 0.25], [-0.5]]

nn1 = Network(layer_weights_1, layer_biases_1)

outputs1 = nn1.forward([1.5, 0.5])
print(f"Test 1: {outputs1}")

outputs2 = nn1.forward([0, 1])
print(f"Test 2: {outputs2}")


print("----- Model 2 -----")
layer_weights_2 = [[[0.5, 0.6], [1.5, -0.8]], [[0.6], [-0.8]], [[0.5, -0.4]]]
layer_biases_2 = [[0.3, 1.25], [0.3], [0.2, 0.5]]

nn2 = Network(layer_weights_2, layer_biases_2)

outputs3 = nn2.forward([0.75, 1.25])
print(f"test 1: {outputs3}")

outputs4 = nn2.forward([-1, 0.5])
print(f"test 2: {outputs4}")
