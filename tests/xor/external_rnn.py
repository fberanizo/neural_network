# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('../..'))

import unittest, pandas, numpy, itertools, rnn
from sklearn import cross_validation

class ExternalRNN(unittest.TestCase):
    """Test cases for XOR problem."""
    grid_search = True

    def test_1(self):
        """Tests the accuracy of an External RNN using k-folds validation method."""
        X = pandas.DataFrame(data=[[0,0],[0,1],[1,0],[1,1]]*10, columns=['x1','x2']).as_matrix()
        y = pandas.DataFrame(data=[[0],[1],[1],[0]]*10, columns=['y']).as_matrix()

        n_folds = 5
        skf = cross_validation.StratifiedKFold(y.flatten(), n_folds=n_folds)

        accuracies_test = [0]*n_folds
        for fold, (train_index, test_index) in enumerate(skf):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            accuracies = map(lambda x: 0, self.hipergrid())
            for idx, hiperparams in enumerate(self.hipergrid()):
                self.progress(((1.0+idx)+len(self.hipergrid())*fold)/(len(self.hipergrid())*n_folds))
                skf2 = cross_validation.StratifiedKFold(y_train.flatten(), n_folds=n_folds)

                for train_index2, test_index2 in skf2:
                    X_train2, X_test2 = X[train_index2], X[test_index2]
                    y_train2, y_test2 = y[train_index2], y[test_index2]
                    classifier2 = rnn.ExternalRNN(**hiperparams).fit(X_train2, y_train2)
                    accuracies[idx] += classifier2.score(X_test2, y_test2)

            # Finds which hiperparams give maximum accuracy
            best_hiperparams = self.hipergrid()[accuracies.index(numpy.max(accuracies))]

            classifier = rnn.ExternalRNN(**best_hiperparams).fit(X_train, y_train)
            accuracies_test[fold] = classifier.score(X_test, y_test)

        print 'Acurácia média:' + numpy.mean(accuracies_test)

    def hipergrid(self):
        """Hiperparameters for External RNN"""
        hidden_layer_size = [{'hidden_layer_size':3},{'hidden_layer_size':5},{'hidden_layer_size':7}]
        learning_rate = [{'learning_rate':0.1},{'learning_rate':0.3},{'learning_rate':1}]
        delays = [{'delays':1},{'delays':2},{'delays':3}]
        grid = []
        
        for hiperparams in itertools.product(hidden_layer_size, learning_rate, delays):
            d = {}
            for hiperparam in hiperparams:
                d.update(hiperparam)
            grid.append(d)

        return grid

    def progress(self, percent):
        """Prints progress in stdout"""
        bar_length = 20
        hashes = '#' * int(round(percent * bar_length))
        spaces = ' ' * (bar_length - len(hashes))
        sys.stdout.write("\rPerforming 5-folds grid search: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
        sys.stdout.flush()


if __name__ == '__main__':
    unittest.main()