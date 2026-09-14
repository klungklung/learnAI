import random
import numpy as np

class network:
    def __init__(self, neurons, learning_constant, activation_function: str):
        self.weights = np.array([random.random() * 2 - 1 for i in range(neurons + 1)])
        self.learning_constant = learning_constant
        self.activation_function = activation_function

    def activation_function(self, inputs):
        match self.activation_function:
            case "sigmoid":
                return 1 / (1 + np.exp(-inputs))
            case "relu":
                return np.maximum(0, inputs)
            case "tanh":
                return np.tanh(inputs)
            case _:
                return inputs

    def activation(self, inputs):
        return inputs

    def feedforward(self, inputs):
        full_inputs = np.append(inputs, 1)
        # print(full_inputs)
        output = np.sum(np.multiply(full_inputs, self.weights))
        return self.activation(output)

    def train(self, learn, target):
        for learn, target in zip(learn, target):
            guess = self.feedforward(learn)
            error = target - guess
            self.weights = error * learn * self.learning_constant

    def get_weights(self):
        return self.weights

network = network(1, 0.001, activation_function="")
# print(network.weights)
# print(network.feedforward(np.array([1])))

xs = np.array([1, 2, 3, 4, 5, 6])
ys = np.array([2, 4, 6, 8, 10, 12])
network.train(xs, ys)

for _ in range(3000):
    network.train(xs, ys)

print(network.feedforward(np.array([9])))