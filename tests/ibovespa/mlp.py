# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('../..'))

import unittest, pandas, numpy, datetime, itertools, mlp
from sklearn import cross_validation

class MLP(unittest.TestCase):
    """Test cases for Ibovespa tendency problem."""
    grid_search = True

    def test_1(self):
        """Tests the accuracy of a MLP using k-folds validation method."""

        # Read data from CSV files
        X_train, X_test, y_train, y_test = self.read_data()

        n_folds = 5

        accuracies = map(lambda x: 0, self.hipergrid())
        for idx, hiperparams in enumerate(self.hipergrid()):
            skf = cross_validation.StratifiedKFold(y_train.flatten(), n_folds=n_folds)

            for fold, (train_index, test_index) in enumerate(skf):
                self.progress(((1.0+fold)+n_folds*idx)/(len(self.hipergrid())*n_folds))
                X_train2, X_test2 = X_train[train_index], X_train[test_index]
                y_train2, y_test2 = y_train[train_index], y_train[test_index]
                classifier = mlp.MLP(**hiperparams).fit(X_train2, y_train2)
                accuracies[idx] += classifier.score(X_test2, y_test2)

        # Finds which hiperparams give maximum accuracy
        best_hiperparams = self.hipergrid()[accuracies.index(numpy.max(accuracies))]

        accuracy = classifier.score(X_test, y_test)
        print 'Acurácia no cj treino:' + str(numpy.max(accuracies)/n_folds)
        print 'Acurácia no cj teste:' + str(accuracy)
        print 'Melhores hiperparâmetros: ' + str(best_hiperparams)

    def read_data(self):
        """Reads and processes financial data from CSV files"""
        ibovespa = "%5EBVSP"
        america = ["%5EGSPC", "%5EDJI", "%5EMERV", "%5EMXX", "%5EIXIC", "%5EIPSA"]
        europe = ["%5EFTSE", "%5EGDAXI", "%5EFCHI", "FTSEMIB.MI", "%5EIBEX"]
        asia = ["%5EN225", "%5EHSI", "%5EBSESN", "%5ESSEC", "%5EJKSE"]

        continents = 3
        stocks_per_continent = 5
        time_window = 7 # 7 days
        prediction_range = 1 # 1 day

        stocks = america + europe + asia

        # Request stock data
        # data = {}
        # url = "http://ichart.finance.yahoo.com/table.csv?s=STOCK_NAME&g=d&a=0&b=1&c=2016&&ignore=.csv"
        # for stock_name in america + europe + asia + [ibovespa]:
        #     print stock_name
        #     s = requests.get(url.replace("STOCK_NAME", stock_name)).content
        #     stock = pandas.read_csv(io.StringIO(s.decode('utf-8'))).set_index("Date")
        #     stock.to_csv('input/' + stock_name  + '.csv')

        ibovespa_data = pandas.read_csv('input/' + ibovespa + '.csv', parse_dates=['Date'])
        stock_data = pandas.DataFrame(data=[], columns=['Date','Open','High','Low','Close','Volume','Adj Close'])
        for stock in stocks:
            stock_data = stock_data.append(pandas.read_csv('input/' + stock + '.csv', parse_dates=['Date']))

        train = pandas.DataFrame(data=[], columns=['Date', 'Trend']).set_index("Date")
        test = pandas.DataFrame(data=[], columns=['Date', 'Trend']).set_index("Date")
        for idx, ibovespa_data in ibovespa_data.iterrows():
            trend = 0 if ibovespa_data["Close"] < ibovespa_data["Open"] else 1

            start_date = ibovespa_data["Date"] + pandas.Timedelta('-1 days')
            end_date = ibovespa_data["Date"] + pandas.Timedelta('-1 days')
            mask = (stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)
            stocks = stock_data.loc[mask]['Close'].tolist()

            columns = ['Date', 'Trend'] + range(len(stocks))
            data = [ibovespa_data["Date"], trend] + stocks
            row = pandas.DataFrame([data], columns=columns).set_index("Date")

            # Data from last 3 months is test, the rest is train
            three_months_ago = pandas.to_datetime('today') + pandas.Timedelta('-90 days')
            if ibovespa_data["Date"] < three_months_ago:
                train = train.append(row)
            else:
                test = test.append(row)

        # Removes rows with NaN columns
        train.dropna(axis=0, how='any', inplace=True)
        test.dropna(axis=0, how='any', inplace=True)

        X_train = train[train.columns.tolist()[:-1]].as_matrix()
        y_train = train[train.columns.tolist()[-1:]].as_matrix()

        X_test = train[train.columns.tolist()[:-1]].as_matrix()
        y_test = train[train.columns.tolist()[-1:]].as_matrix()

        return X_train, X_test, y_train, y_test

    def hipergrid(self):
        """Hiperparameters for MLP"""
        hidden_layer_size = [{'hidden_layer_size':3},{'hidden_layer_size':5},{'hidden_layer_size':7}]
        learning_rate = [{'learning_rate':0.1},{'learning_rate':0.3},{'learning_rate':1}]
        grid = []
        
        for hiperparams in itertools.product(hidden_layer_size, learning_rate):
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
