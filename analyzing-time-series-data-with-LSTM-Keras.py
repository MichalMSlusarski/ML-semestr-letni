import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from tensorflow import keras 
from keras import models
from keras import layers
from keras import Sequential,Model 
from keras import Input
from keras.layers import Activation, Dense
from keras import regularizers
from keras.layers import Dense,Dropout
from tensorflow.keras.layers import BatchNormalization

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from numpy import array

from matplotlib.pyplot import figure
from time import gmtime, strftime

print(strftime("%H:%M:%S start", gmtime()))


# Model
n_lag = 24
n_neurons = 200
n_epochs = 100
n_batch = 1

# Make forecasts
n_forecast = 24
ntimes=3;

#############################################################################

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, 1):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def difference(dataset, interval=1, w=0):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i]- w * dataset[i - interval] 
        diff.append(value)
    return Series(diff)


# transform series into train sets for supervised learning
def prepare_data(series, n_lag):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag)
    supervised_values = supervised.values
    # split into train set
    train = supervised_values
    return scaler, train


def inverse_difference(last_ob, forecast, w=0):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0]+ w * last_ob )
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i]+ w * inverted[i-1] )
    return inverted


# inverse data transform on forecasts
# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted


initializer1 = tf.keras.initializers.GlorotNormal(seed=1934)
initializer2 = tf.keras.initializers.Orthogonal(gain=1.0,seed=9834)
initializer3 = tf.keras.initializers.Zeros()
  
def forecasts_LSTM(train, n_batch=1,n_lag=1, n_forecast=1,n_neurons=4):
    forecasts = list()
    X, y = train[:, 1:n_lag+1], train[:, 0]
    X=X.reshape(X.shape[0], 1, X.shape[1])
    y=y.reshape(y.shape[0], 1)
    _drop=0.1
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]),stateful=True,return_sequences=True,
                   dropout=_drop,recurrent_dropout=0.0,
                   kernel_initializer=initializer1,recurrent_initializer=initializer2,bias_initializer=initializer3))
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]),stateful=True,return_sequences=True,
                   dropout=_drop,recurrent_dropout=0.0,
                   kernel_initializer=initializer1,recurrent_initializer=initializer2,bias_initializer=initializer3))
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]),stateful=True,return_sequences=True,
                   dropout=_drop,recurrent_dropout=0.0,
                   kernel_initializer=initializer1,recurrent_initializer=initializer2,bias_initializer=initializer3))
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]),stateful=True,return_sequences=True,
                   dropout=_drop,recurrent_dropout=0.0,
                   kernel_initializer=initializer1,recurrent_initializer=initializer2,bias_initializer=initializer3))
    model.add(Dense(y.shape[1],
                    kernel_initializer=initializer1,bias_initializer=initializer3))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    for i in range(n_epochs):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    XX=X[len(X)-1].reshape(1, 1, X.shape[2])
    forecast = model.predict(XX, batch_size=n_batch,use_multiprocessing=True,verbose=0)
    for i in range(n_forecast):
        XX=np.concatenate((XX[XX.shape[0]-1],DataFrame(forecast.reshape(-1,1))),axis=1)
        XX=XX[:,1:]
        XX=XX.reshape(XX.shape[0],1, XX.shape[1])
        forecast = model.predict(XX, batch_size=n_batch,use_multiprocessing=True,verbose=0)
        forecasts.append(forecast)
    forecasts=array(forecasts)
    forecasts=forecasts.reshape(1,n_forecast)
    return forecasts

def plot_forecasts(rate,forecasts,low,up,date):
    # plot the entire dataset in blue
    figure(figsize=(4,3),dpi=300)
    plt.rcParams.update({'font.size': 4})

    plt.title('Model LSTM:  '+'  [lag='+str(n_lag)+']', loc='center')
    d=date.iloc[range(0,np.shape(rate)[0])]
    d=d.to_numpy()
    plt.plot(d,rate,linewidth=0.3,label='History')
    # plot the forecasts in red
    nrate=np.shape(rate)[0]
    x=range(nrate-1,nrate+np.shape(forecasts)[0]-1)
    d=date.iloc[x].to_numpy().flatten()
    ##d=d.date
    ##d=d.to_numpy()
    plt.plot(d,forecasts, color='red',linewidth=0.2,label='Forecast')
    plt.fill_between(d, low, up,color='red', alpha=0.1,linewidth=0)
    plt.grid(visible=True, which='major', color='#666666', linestyle='-',linewidth=0.06)
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2,linewidth=0.06)
    plt.legend(loc ="upper right",frameon=False)
    plt.ylabel('Percent of premature babies')
    plt.xlabel("Date")
    # show the plot
    plt.show()

#############################################################################
 

# load dataset
dateparse = lambda x: datetime.strptime(x, '%d.%m.%Y')
series0= read_csv(r"G:\Aktualne\UW\NLP1\Z3\PTB_SB.csv", delimiter=';',parse_dates=['date'], date_parser=dateparse)

series=series0[["pr_s","date"]]
rate=series.pr_s
Date=series["date"]
d=pd.date_range(start = Date.iloc[0], periods = 200,freq='MS')
d=pd.DataFrame(d)
d.columns=["date"]

# prepare data
scaler, train = prepare_data(rate, n_lag)

xfuture=pd.DataFrame([])
for i in range(ntimes):
    # make forecasts
    forecasts=forecasts_LSTM(train, n_batch=n_batch, n_lag=n_lag, n_forecast=n_forecast,n_neurons=n_neurons)
    # inverse transform forecasts 
    forecasts = inverse_transform(rate, forecasts, scaler)
    xfuture=pd.concat((xfuture,pd.DataFrame(forecasts)),axis=0,ignore_index=True)
    
  
mfore=np.array(xfuture.mean(axis=0))
delta3=rate.iloc[-1]-mfore[0]
mfore=mfore+delta3

sfore=np.array(xfuture.std(axis=0))
low=mfore-1.96*sfore
up=mfore+1.96*sfore    
    
    
plot_forecasts(rate.to_numpy(),mfore ,low ,up,d ) 

print(strftime("%H:%M:%S end", gmtime()))
