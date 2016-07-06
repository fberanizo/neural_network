# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath('../..'))

import unittest, enum, pandas, io, requests, datetime, numpy, mlp

class Trend(enum.Enum):
    down = 0
    up = 1
    stable = 2

class MLP(unittest.TestCase):
    """Test cases for CVRP problem."""
    grid_search = False

    def test_1(self):
        """
        Performs a simple prediction.
        """
        sys.stdout.write("Starting test_1: MLP\n")

        url = "http://ichart.finance.yahoo.com/table.csv?s=STOCK_NAME&g=d&a=1&b=1&c=2016&&ignore=.csv"

        ibovespa = "%5EBVSP"
        america = ["%5EDJI"]
        europe = ["%5EFTSE"]
        asia = ["%5EN225"]

        continents = 3
        stocks_per_continent = 1
        time_window = 7 # 7 days
        prediction_range = 1 # 1 day

        columns = [ibovespa]
        columns += america[:stocks_per_continent] # TODO: Test subsets other than 0..n
        columns += europe[:stocks_per_continent]
        columns += asia[:stocks_per_continent]

        # Request stock data
        data = {}
        for stock_name in columns:
            s = requests.get(url.replace("STOCK_NAME", stock_name)).content
            stock = pandas.read_csv(io.StringIO(s.decode('utf-8')))
            for index, row in stock.iterrows():
                date = pandas.Timestamp(datetime.datetime.strptime(row["Date"], '%Y-%m-%d'))
                if not date in data.keys():
                    data[date] = dict.fromkeys(columns+["Date"], None)
                    data[date]["Date"] = date
                
                if stock_name != ibovespa:
                    data[date][stock_name] = row["Close"]
                else:
                    #if abs(row["Close"]-row["Open"]) <= 1e-2:
                    #    data[date][stock_name] = Trend.stable
                    if row["Close"] < row["Open"]:
                        data[date][stock_name] = 0
                    else:
                        data[date][stock_name] = 1

        data = pandas.DataFrame(data=data.values(), columns=columns+["Date"]).set_index("Date")
        data.sort_index(inplace=True)

        # Para cada valor da ibovespa, montar entrada
        X = pandas.DataFrame(data=[])
        y = pandas.DataFrame(data=[], columns=[ibovespa])
        for index, row in data.iterrows():
            yi = pandas.DataFrame([[row[ibovespa]]], columns=[ibovespa])
            y = y.append(yi)
            
            start_date = index-pandas.Timedelta(days=time_window)
            end_date = index-pandas.Timedelta(days=prediction_range)

            input = data.loc[start_date:end_date][columns[1:]].as_matrix().flatten()
            Xi = pandas.DataFrame(input).transpose()
            X = X.append(Xi)
        #print y
        #print X

        X = X.div(X.sum(axis=0), axis=1)
        y = data[columns[0]].as_matrix()
        #print X
        print y

        network = mlp.MLP(X.shape[1], 3, 1)
        print network.fit(X, y)


if __name__ == '__main__':
    unittest.main()