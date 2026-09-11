import random
import numpy as np

# from main_old import perceptron


class Perceptron:
    def __init__(self, neurons, learning_constant, activation_function: str):
        self.weights = [random.random() * 2 - 1 for _ in range(neurons + 1)]
        self.learningConstant = learning_constant
        self.activationFunction = activation_function

    def activation(self, input_value):
        match self.activationFunction:
            case "sigmoid":
                return 1 / (1 + np.exp(-input_value))
            case "relu":
                return np.maximum(0, input_value)
            case "tanh":
                return np.tanh(input_value)
            case _:
                return input_value

    def feedforward(self, inputs):
        output = 0
        full_inputs = inputs + [1]
        for i in range(len(self.weights)):
            output += full_inputs[i] * self.weights[i]

        return self.activation(output)


    def train(self, x, y, epochs):
        for epoch in range(epochs):
            for x_val, target in zip(x, y):
                self.print_weights()

                x_list = [x_val]
                full_inputs = x_list + [1]

                guess = self.feedforward(x_list)
                error = target - guess

                for j in range(len(self.weights)):
                    self.weights[j] += error * full_inputs[j] * self.learningConstant




    def predict(self, inputs):
        guess = self.feedforward(inputs)
        return guess

    def print_weights(self):
        print(self.weights)


learn = [1, 2, 3, 4, 5]
target = [5, 7, 9, 11, 13]

model = Perceptron(1, 0.01, activation_function="linear")
model.train(learn, target, 10)
# model.print_weights()

print(model.predict([4]))