# -*- coding: utf-8 -*-

import numpy
from scipy import optimize

class MLP(object):
    """Class that implements a multilayer perceptron (MLP)"""
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        # Initialize weights
        self.W1 = numpy.random.rand(self.input_layer_size, self.hidden_layer_size)
        self.W2 = numpy.random.rand(self.hidden_layer_size, self.output_layer_size)

    def fit(self, X, y):
        # BFGS algorithm
        self.X = X
        self.y = y
        self.J = []

        initial_guess = self.get_params()
        options = {"maxiter":200, "disp":True}
        self.results = optimize.minimize(self.objective, initial_guess, jac=True, method='BFGS', args=(X,y), options=options, callback=self.callback)
        self.set_params(self.results.x)

        return self

    def predict(self):
        pass

    def callback(self, params):
        self.set_params(params)
        self.J.append(self.cost(self.X, self.y))

    def set_params(self, params):
        end1 = self.hidden_layer_size*self.input_layer_size
        self.W1 = numpy.reshape(params[0:end1], \
                                (self.input_layer_size, self.hidden_layer_size))
        end2 = end1 + self.hidden_layer_size*self.output_layer_size
        self.W2 = numpy.reshape(params[end1:end2], \
                                (self.hidden_layer_size, self.output_layer_size))

    def get_params(self, **params):
        params = numpy.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def forward(self, X):
        """Passes values through network"""
        # TODO: Implementar atrasador
        self.Zin = numpy.dot(X, self.W1)
        self.Z = self.sigmoid(self.Zin)
        self.Yin = numpy.dot(self.Z, self.W2)
        Y = self.linear(self.Yin)
        return Y

    def objective(self, params, X, y):
        """a"""
        self.set_params(params)

        cost = self.cost(X, y)
        gradients = self.gradients(X, y)

        return cost, gradients

    def cost(self, X, y):
        self.Y = self.forward(X)
        return -(y-self.Y)

    def cost_derivative(self, X, y):
        self.Y = self.forward(X)
        delta3 = numpy.multiply(-(y-self.Y), self.linear_derivative(self.Yin))
        dJdW2 = numpy.dot(self.Z.T, delta3)

        delta2 = numpy.dot(delta3, self.W2.T)*self.sigmoid_derivative(self.Zin)
        dJdW1 = numpy.dot(X.T, delta2)

        return dJdW1, dJdW2

    def gradients(self, X, y):
        dJdW1, dJdW2 = self.cost_derivative(X, y)
        return numpy.concatenate((dJdW1.ravel(), dJdW2.ravel()))

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
