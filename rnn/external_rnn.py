# -*- coding: utf-8 -*-

import numpy, matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class ExternalRNN(object):
    """Class that implements a External Recurent Neural Network"""
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        # Initialize weights
        self.W1 = numpy.random.rand(self.input_layer_size, self.hidden_layer_size)
        self.W2 = numpy.random.rand(self.hidden_layer_size, self.output_layer_size)

    def fit(self, X, y):
        """Trains the network and returns the trained network"""
        self.X = X  # input
        self.y = y  # output
        self.J = [] # error

        epsilon = 0.03
        remaining_epochs = 2000
        learning_rate = 0.1
        params = numpy.concatenate((self.W1.ravel(), self.W2.ravel()))
        error = 1

        # Max epochs: 2000
        while error > epsilon and remaining_epochs > 0:
            error, gradients = self.run_epoch(params, X, y)
            
            # Saves error for plot
            self.J.append(error)

            # Calculates new weights
            dJdW1, dJdW2 = gradients
            W1 = self.W1 - learning_rate*dJdW1
            W2 = self.W2 - learning_rate*dJdW2
            params = numpy.concatenate((W1.ravel(), W2.ravel()))

            print 'Epoch: ' + str(remaining_epochs)
            print 'Error: ' + str(error)
            #print W1
            #print W2

            remaining_epochs -= 1

        # After training, we plot error in order to see how it behaves
        plt.plot(self.J[1:])
        plt.grid(1)
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.show()
        
        return self

    def predict(self, X):
        return self.forward(X)

    def run_epoch(self, params, X, y):
        """Runs one epoch of training"""
        self.set_params(params)

        self.Y = self.forward(X)
        cost = self.cost(X, y)
        gradients = self.backpropagate(X, y)

        return cost, gradients

    def forward(self, X):
        """Passes input values through network and return output values"""
        self.Zin = numpy.dot(X, self.W1)
        self.Z = self.sigmoid(self.Zin)
        self.Yin = numpy.dot(self.Z, self.W2)
        Y = self.linear(self.Yin)
        return Y

    def cost(self, X, y):
        """Calculates network output error"""
        return mean_squared_error(self.Y, y)

    def backpropagate(self, X, y):
        """Backpropagates costs through the network"""
        delta3 = numpy.multiply(-(y-self.Y), self.linear_derivative(self.Yin))
        dJdW2 = numpy.dot(self.Z.T, delta3)

        delta2 = numpy.dot(delta3, self.W2.T)*self.sigmoid_derivative(self.Zin)
        dJdW1 = numpy.dot(X.T, delta2)

        return dJdW1, dJdW2

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

    def set_params(self, params):
        end1 = self.hidden_layer_size*self.input_layer_size
        self.W1 = numpy.reshape(params[0:end1], (self.input_layer_size, self.hidden_layer_size))
        end2 = end1 + self.hidden_layer_size*self.output_layer_size
        self.W2 = numpy.reshape(params[end1:end2], (self.hidden_layer_size, self.output_layer_size))

    def get_params(self, **params):
        params = numpy.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
