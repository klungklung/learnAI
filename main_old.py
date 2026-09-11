import random
from matplotlib import pyplot as plt

perceptron = None
training = []
count = 0

def f(x):
    return 2 * x - 4

def setup():
    global perceptron, x, y
    perceptron = Perceptron(2, 0.0001)

    for i in range(2000):
        x = random.randint(-200, 200)
        y = random.randint(-200, 200)
        training.append([x, y, 1])


class Perceptron:
    def __init__(self, n, learning_constant):
        self.weights = [random.random() * 2 - 1 for _ in range(n + 1)]
        self.learningConstant = learning_constant

    def activation(self, sum):
        if sum > 0:
            return 1
        else:
            return -1

    def feedforward(self, inputs):
        sum = 0
        for i in range(len(self.weights)):
            sum += inputs[i] * self.weights[i]

        return self.activation(sum)

    def train(self, inputs, desired):
        guess = self.feedforward(inputs)
        error = desired - guess
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + error * inputs[i] * self.learningConstant


def draw():
    global count, perceptron


    for _ in range(2000):
        x = training[count][0]
        y = training[count][1]

        desired = -1
        if y > f(x):
            desired = 1

        perceptron.train(training[count], desired)
        count = (count + 1) % len(training)

    x_over, y_over = [], []
    x_under, y_under = [], []

    for datapoint in training:
        guess = perceptron.feedforward(datapoint)
        if guess > 0:
            x_over.append(datapoint[0])
            y_over.append(datapoint[1])
        else:
            x_under.append(datapoint[0])
            y_under.append(datapoint[1])

    plt.scatter(x_over, y_over, color='violet')
    plt.scatter(x_under, y_under, color='red')

setup()

x = [-200, 200]
y = [f(i) for i in x]

plt.plot(x, y)
draw()
plt.show()