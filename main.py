import random
import numpy as np

# from main_old import perceptron


class Perceptron:
    def __init__(self, neurons, learning_constant, activation_function: str):
        self.weights = [random.random() * 2 - 1 for _ in range(neurons + 1)]
        self.learningConstant = learning_constant
        self.activationFunction = activation_function


    def softmax(self, input_value):
        return np.exp(input_value) / np.sum(np.exp(input_value), axis=0)

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
        error = 0

        for epoch in range(epochs):

            for learn, target in zip(x, y):
                # print(self.get_weights())

                x_list = [learn] + [1]

                guess = self.feedforward(x_list)
                error = target - guess


                for j in range(len(self.weights)):
                    self.weights[j] += error * x_list[j] * self.learningConstant
            # print(error)

    def get_weights(self):
        return self.weights


learn_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
target_set = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

model = Perceptron(1, 0.01, activation_function="linear")
model.train(learn_set, target_set, 3000)
# model.print_weights()

print(model.feedforward([23]))