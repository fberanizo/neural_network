# -*- coding: utf-8 -*-

import numpy, matplotlib.pyplot as plt, time
from sklearn.metrics import mean_squared_error

class ExternalRNN(object):
    """Class that implements a External Recurent Neural Network"""
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, delays):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.delays = delays

        # Initialize weights
        self.W1 = numpy.random.rand(1 + self.input_layer_size, self.hidden_layer_size)
        self.W2 = numpy.random.rand(1 + self.hidden_layer_size, self.output_layer_size)
        self.W3 = numpy.random.rand(self.output_layer_size, self.output_layer_size * self.delays)
        self.Ydelayed = numpy.zeros((1, self.output_layer_size*self.delays))

    def fit(self, X, y):
        """Trains the network and returns the trained network"""
        epsilon = 0.01
        remaining_epochs = 2000
        learning_rate = 0.2
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
                dJdW3 = gradients[2]

                # Calculates new weights
                self.W1 = self.W1 - learning_rate * dJdW1
                self.W2 = self.W2 - learning_rate * dJdW2
                self.W3 = self.W3 - learning_rate * dJdW3

                # Shift Ydelayed values through time
                self.Ydelayed = numpy.roll(self.Ydelayed, 1, 1)
                self.Ydelayed[:,::self.delays] = self.Y

            # Saves error for plot
            error = total_error.mean()
            self.J.append(error)

            print 'Epoch: ' + str(remaining_epochs)
            print 'Error: ' + str(error)

            remaining_epochs -= 1

        # After training, we plot error in order to see how it behaves
        plt.plot(self.J[1:])
        plt.grid(1)
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.show()

        return self

    def predict(self, X):
        """Predicts test values"""
        Y = []
        for x in X:
            Y.append(self.forward(numpy.array([x])))
        return numpy.array(Y)


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
        self.Zin += numpy.dot(self.Ydelayed, self.W3.T)
        self.Z = self.sigmoid(self.Zin)

        self.Yin = numpy.dot(self.Z, self.W2[:-1,])
        self.Yin += numpy.dot(numpy.ones((1, 1)), self.W2[-1:,:])
        Y = self.linear(self.Yin)
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

        dJdW3 = numpy.dot(numpy.repeat(self.Ydelayed, self.hidden_layer_size, 0), \
                          numpy.repeat(numpy.repeat(delta2, self.hidden_layer_size*self.delays, 0), self.delays, 1))

        return dJdW1, dJdW2, dJdW3

    def sigmoid(self, z):
        """Apply sigmoid activation function"""
        return 1/(1+numpy.exp(-z))

    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        return numpy.exp(-z)/((1+numpy.exp(-z))**2)

    def linear(self, z):
        """Apply linear activation function"""
        return z

    def linear_derivative(self, z):
        """Derivarive linear function"""
        return 1
