# -*- coding: utf-8 -*-

import numpy, matplotlib.pyplot as plt, time
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score

class MLP(object):
    """Class that implements a multilayer perceptron (MLP)"""
    def __init__(self, hidden_layer_size=3, learning_rate=0.2, max_epochs=1000):
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.auc = 0.5

    def fit(self, X, y):
        """Trains the network and returns the trained network"""
        self.input_layer_size = X.shape[1]
        self.output_layer_size = y.shape[1]
        remaining_epochs = self.max_epochs

        # Initialize weights
        self.W1 = numpy.random.rand(1 + self.input_layer_size, self.hidden_layer_size)
        self.W2 = numpy.random.rand(1 + self.hidden_layer_size, self.output_layer_size)

        epsilon = 0.001
        error = 1
        self.J = [] # error

        # Repeats until error is small enough or max epochs is reached
        while error > epsilon and remaining_epochs > 0:
            total_error = numpy.array([])

            # For each input instance
            for self.X, self.y in zip(X, y):
                self.X = numpy.array([self.X])
                self.y = numpy.array([self.y])
                error, gradients = self.single_step(self.X, self.y)
                total_error = numpy.append(total_error, error)
                dJdW1 = gradients[0]
                dJdW2 = gradients[1]

                # Calculates new weights
                self.W1 = self.W1 - self.learning_rate * dJdW1
                self.W2 = self.W2 - self.learning_rate * dJdW2

            # Saves error for plot
            error = total_error.mean()
            self.J.append(error)

            # print 'Epoch: ' + str(remaining_epochs)
            # print 'Error: ' + str(error)

            remaining_epochs -= 1

        
        # After training, we plot error in order to see how it behaves
        #plt.plot(self.J[30:])
        #plt.grid(1)
        #plt.yscale('log')
        #plt.ylabel('Cost')
        #plt.xlabel('Iterations')
        #plt.show()

        return self

    def predict(self, X):
        """Predicts test values"""
        Y = map(lambda x: self.forward(numpy.array([x]))[0], X)
        Y = map(lambda y: 1 if y > self.auc else 0, Y)
        return numpy.array(Y)

    def score(self, X, y_true):
        """Calculates accuracy"""
        y_pred = map(lambda x: self.forward(numpy.array([x]))[0], X)
        auc = roc_auc_score(y_true, y_pred)
        y_pred = map(lambda y: 1 if y > self.auc else 0, y_pred)
        y_pred = numpy.array(y_pred)
        return accuracy_score(y_true.flatten(), y_pred.flatten())

    def single_step(self, X, y):
        """Runs single step training method"""
        self.Y = self.forward(X)
        cost = self.cost(self.Y, y)
        gradients = self.backpropagate(X, y)

        return cost, gradients

    def forward(self, X):
        """Passes input values through network and return output values"""
        self.Zin = numpy.dot(X, self.W1[:-1,:])
        self.Zin += numpy.dot(numpy.ones((1, 1)), self.W1[-1:,:])
        self.Z = self.sigmoid(self.Zin)
        self.Z = numpy.nan_to_num(self.Z)

        self.Yin = numpy.dot(self.Z, self.W2[:-1,])
        self.Yin += numpy.dot(numpy.ones((1, 1)), self.W2[-1:,:])
        Y = self.linear(self.Yin)
        Y = numpy.nan_to_num(Y)
        return Y

    def cost(self, Y, y):
        """Calculates network output error"""
        return mean_squared_error(Y, y)

    def backpropagate(self, X, y):
        """Backpropagates costs through the network"""
        delta3 = numpy.multiply(-(y-self.Y), self.linear_derivative(self.Yin))
        dJdW2 = numpy.dot(self.Z.T, delta3)
        dJdW2 = numpy.append(dJdW2, numpy.dot(numpy.ones((1, 1)), delta3), axis=0)

        delta2 = numpy.dot(delta3, self.W2[:-1,:].T)*self.sigmoid_derivative(self.Zin)
        dJdW1 = numpy.dot(X.T, delta2)
        dJdW1 = numpy.append(dJdW1, numpy.dot(numpy.ones((1, 1)), delta2), axis=0)

        return dJdW1, dJdW2

    def sigmoid(self, z):
        """Apply sigmoid activation function"""
        return 1/(1+numpy.exp(-z))

    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        return numpy.exp(-z)/((1+numpy.exp(-z))**   2)

    def linear(self, z):
        """Apply linear activation function"""
        return z

    def linear_derivative(self, z):
        """Derivarive linear function"""
        return 1
