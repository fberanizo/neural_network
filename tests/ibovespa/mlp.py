# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('../..'))

import unittest, pandas, io, requests, datetime, numpy, mlp
from sklearn import cross_validation

class MLP(unittest.TestCase):
    """Test cases for Ibovespa tendency problem."""
    grid_search = False

    def test_1(self):
        """
        Performs a simple prediction.
        """
        sys.stdout.write("Starting test_1: MLP\n")

        ibovespa = "%5EBVSP"
        america = ["%5EGSPC", "%5EDJI", "%5EMERV", "%5EMXX", "%5EIXIC", "%5EIPSA"]
        europe = ["%5EFTSE", "%5EGDAXI", "%5EFCHI", "FTSEMIB.MI", "%5EIBEX"]
        asia = ["%5EN225", "%5EHSI", "%5EBSESN", "%5ESSEC", "%5EJKSE"]

        continents = 3
        stocks_per_continent = 5
        time_window = 7 # 7 days
        prediction_range = 1 # 1 day

        stocks = america + europe + asia

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

            three_months_ago = pandas.to_datetime('today') + pandas.Timedelta('-90 days')
            if ibovespa_data["Date"] < three_months_ago:
                train = train.append(row)
            else:
                test = test.append(row)

        # Removes rows with NaN columns
        train.dropna(axis=0, how='any', inplace=True)
        test.dropna(axis=0, how='any', inplace=True)

        data = train[train.columns.tolist()[:-1]].as_matrix()
        target = train[train.columns.tolist()[-1:]].as_matrix()

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, target, test_size=0.4, random_state=0)

        # Request stock data
        # data = {}
        # url = "http://ichart.finance.yahoo.com/table.csv?s=STOCK_NAME&g=d&a=0&b=1&c=2016&&ignore=.csv"
        # for stock_name in america + europe + asia + [ibovespa]:
        #     print stock_name
        #     s = requests.get(url.replace("STOCK_NAME", stock_name)).content
        #     stock = pandas.read_csv(io.StringIO(s.decode('utf-8'))).set_index("Date")
        #     stock.to_csv('input/' + stock_name  + '.csv')

        classifier = mlp.MLP(hidden_layer_size=3).fit(X_train, y_train)
        print classifier.score(X_test, y_test)

        #X = pandas.DataFrame(data=[[1,0]], columns=['x1','x2'])
        #y = classifier.predict(X)
        #print y


if __name__ == '__main__':
    unittest.main()
