# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('../..'))

import unittest, pandas, numpy, mlp
from sklearn import preprocessing, decomposition, cross_validation, svm, ensemble, neighbors

class MLP(unittest.TestCase):
    """Test cases for XOR problem."""
    grid_search = True

    def test_1(self):
        X = pandas.DataFrame(data=[[0,0],[0,1],[1,0],[1,1]], columns=['x1','x2'])
        y = pandas.DataFrame(data=[[0],[1],[1],[0]], columns=['y'])

        hidden_layer_size = 4

        network = mlp.MLP(X.shape[1], hidden_layer_size, 1)
        classifier = network.fit(X, y)

        X = pandas.DataFrame(data=[[0,0],[0,1],[1,0],[1,1]], columns=['x1','x2'])
        y = classifier.predict(X)
        print y


if __name__ == '__main__':
    unittest.main()