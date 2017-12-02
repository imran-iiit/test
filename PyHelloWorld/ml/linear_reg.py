
'''
https://www.youtube.com/watch?v=r4mwkS2T9aI&index=4&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
'''

import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
print(df.tail())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1 * len(df)) ) 
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
# df.dropna(inplace=True)

# print(df.head())
# print(df.tail())

# TEST: If we remove 'Adj. Close' from X (like 'label'), which is a price material column, 
# the predicted values will be flat and not as expected
# X = np.array(df.drop(['label', 'Adj. Close'], 1))
X = np.array(df.drop(['label'], 1)) 
X = preprocessing.scale(X)
X_lately = X[-forecast_out:] # i.e. dont test on the data we dont have
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1) # What training model to use; For infinite multi threading n_jobs=-1
clf.fit(X_train, y_train) # Takes most time as this is very expensive, as it is training the data
# So, pickle the data
with open('linear_regression.pickle', 'wb') as f:
    pickle.dump(clf, f) # Dump the trained classifier!

pickle_in = open('linear_regression.pickle', 'rb') 
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

print(accuracy)

# Now that we have great accuracy, we can PREDICT the future values! Based on the X values
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

# Now plot this!
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 24 * 60 * 60 
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]



df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
