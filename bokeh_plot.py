# bokeh_plot.py 
# run: bokeh serve bokeh_plot.py --args

# import modules
from collections import defaultdict
import numpy as np
from flask import Flask, render_template, render_template_string
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import talib
from talib import abstract
from fred import Fred
from datetime import datetime
from bokeh.embed import server_document
import requests
from requests_futures.sessions import FuturesSession
import time
from io import BytesIO
from zipfile import ZipFile
import os
import gzip
import json
import pandas as pd
from pandas import Series

from bokeh.io import curdoc
from bokeh.server.server import Server
from bokeh.embed import server_document
from bokeh.models import (BoxSelectTool, ColumnDataSource, CDSView,
                        CategoricalColorMapper, NumeralTickFormatter, 
                        HoverTool, Button, CustomJS, GroupFilter)
from bokeh.models.ranges import Range1d
from bokeh.models.widgets import (Select, Button, Div, RadioGroup,
                                 TableColumn, DataTable)
from bokeh.models.annotations import BoxAnnotation
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.palettes import Spectral5, mpl, Spectral10, Spectral4
from bokeh.themes import Theme
from tornado.ioloop import IOLoop
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.distributions.empirical_distribution import ECDF

import pyramid as pm
from pyramid.arima import auto_arima
from pyramid.arima import arima as pyrima

from sklearn import base, tree
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                            GradientBoostingClassifier, VotingClassifier)
from sklearn.feature_extraction import DictVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
from sklearn.utils import shuffle

AV_API_key = '567TRV8RTL728INO' # Alpha Vantage
ts = TimeSeries(key=AV_API_key)
fx = ForeignExchange(key=AV_API_key)
cc = CryptoCurrencies(key=AV_API_key)

FRED_API_Key = 'a615f729065fa8c55e2014da998b8bd9' #FRED
df_0 = pd.read_csv(r".\tickers.csv")

months = ["January","February", "March", "April", "May", "June", 
          "July", "August", "September", "October", "November", "December"]

session = FuturesSession()

FRED_cache_dir = 'FRED_cache'
# if not os.path.exists(FRED_cache_dir):
#     os.mkdir(FRED_cache_dir)
    
ticker_cache_dir = 'ticker_cache'
# if not os.path.exists(ticker_cache_dir):
#     os.mkdir(ticker_cache_dir)

fit_cache_dir = 'fit_cache'
# if not os.path.exists(fit_cache_dir):
#     os.mkdir(fit_cache_dir)

ticker = 'GOOG'
commodity_type = 'NYSE Equity'

function_groups = talib.get_function_groups()
functions = talib.get_functions()
pattern_list = [function for function in function_groups['Pattern Recognition'] ]
pattern_names = ['2 CROWS', '3 BLACK CROWS', '3 INSIDE', '3 LINE STRIKE', '3 OUTSIDE', '3 STARS IN THE SOUTH',
                    '3 WHITE SOLDIERS', 'ABANDONED BABY', 'ADVANCE BLOCK', 'BELTHOLD', 'BREAKAWAY', 'CLOSING MARUBOZU',
                    'CONCEAL BABY SWALL', 'COUNTERATTACK', 'DARK CLOUD COVER', 'DOJI', 'DOJI STAR', 'DRAGONFLY DOJI', 'ENGULFING',
                    'EVENING DOJISTAR', 'EVENING STAR', 'GAP-SIDE SIDE-WHITE', 'GRAVESTONE DOJI', 'HAMMER', 'HANGINGMAN',
                    'HARAMI', 'HARAMI CROSS', 'HIGH WAVE', 'HIKKAKE', 'HIKKAKE MOD', 'HOMING PIGEON', 'IDENTICAL 3 CROWS',
                    'IN-NECK', 'INVERTED HAMMER', 'KICKING', 'KICKING BY LENGTH', 'LADDER BOTTOM', 'LONG-LEGGED DOJI',
                    'LONG LINE', 'MARUBOZU', 'MATCHING LOW', 'MATHOLD', 'MORNING DOJI STAR', 'MORNING STAR', 'ON NECK',
                    'PIERCING', 'RICKSHAW MAN', 'RISE-FALL 3-METHODS', 'SEPARATING LINES', 'SHOOTING STAR', 'SHORT LINE',
                    'SPINNING TOP', 'STALLED PATTERN', 'STICK SANDWICH', 'TAKURI', 'TASUKI GAP', 'THRUSTING', 'TRI-STAR',
                    'UNIQUE 3 RIVER', 'UP-SIDE GAP 2 CROWS', 'X-SIDE GAP 3 METHODS']
pattern_dict = {k:v for k,v in zip(pattern_list, pattern_names)}

# Set ML parameters
dict_classifiers = {"Logistic Regression": LogisticRegression(C=10.0, penalty='l1'),
                    "Nearest Neighbors": KNeighborsClassifier(metric='euclidean', n_neighbors=3, weights='uniform'),
                    "Decision Tree": tree.DecisionTreeClassifier(max_depth=19, min_samples_leaf=10),
                    "AdaBoost": AdaBoostClassifier(learning_rate=0.01, n_estimators=1000),
                    "Naive Bayes": GaussianNB(),
                    "QDA": QuadraticDiscriminantAnalysis(),
                    "Linear SVM": SVC(C=1, gamma=0.1),
                    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000, learning_rate=0.001, 
                                                                               max_depth=6, criterion='friedman_mse'),
                    "Gaussian Process": GaussianProcessClassifier(),
                    "Random Forest": RandomForestClassifier(n_estimators=362, max_features='auto', min_samples_split=3, max_depth=30,
                                                            min_samples_leaf=2, bootstrap=True),
                    "Neural Net":  MLPClassifier(alpha = 0.01, solver='lbfgs', max_iter=610, hidden_layer_sizes=5, random_state=0),
                    }
Voters = dict_classifiers.values()
dict_classifiers["Voter"] = VotingClassifier([Voters], voting='hard', n_jobs=-1)

GBC_params = {'n_estimators': [100, 500, 1000],
              'learning_rate': [0.1, 0.01, 0.001],
              'criterion': ['friedman_mse', 'mse', 'mae']}

LR_params = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}

kNN_params = {'n_neighbors': [3, 5, 11, 19],
             'weights': ['uniform', 'distance'],
             'metric': ['euclidean', 'manhattan']}

SVM_params = {'C': [0.001, 0.01, 0.1, 1, 10],
              'gamma' : [0.001, 0.01, 0.1, 1]}

DT_params = {'max_depth': np.arange(1, 21, 2),
             'min_samples_leaf': [1, 2, 5, 10, 20, 50]} #[]

RF_params = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 200)],
             'max_features': ['auto', 'sqrt'],
             'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]+[None],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 4],
             'bootstrap': [True, False]}

NN_params = {'solver': ['lbfgs'], 'max_iter': [500, 1000, 1500], 
             'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(5, 12), 
             'random_state':[0,1,2,3,4,5,6,7,8,9]}

AB_params = {'n_estimators':[500,1000,2000],'learning_rate':[.001,0.01,.1]}

dict_classifier_params = {"Logistic Regression": LR_params,
                          "Nearest Neighbors": kNN_params,
                          "Decision Tree": DT_params,
                          "AdaBoost": AB_params,
                          "Naive Bayes": {},
                          "QDA": {},
                          "Linear SVM": SVM_params,
                          "Gradient Boosting Classifier": GBC_params,
                          "Gaussian Process": {},
                          "Random Forest": RF_params,
                          "Neural Net": NN_params}

dense = ['Naive Bayes', 'QDA', 'Gaussian Process', 'Voter']

searchable = ['Neural Net']

# helper functions
def float_or_NaN(s):
    try: 
        r = float(s)
    except:
        r = float('NaN')
    return r

def get_bounds(col):
    mn = df_ticker[col].min()
    mx = df_ticker[col].max()
    rn = mx - mn
    return mn - 0.05*rn , mx + 0.05*rn

def SMA(n, s):
        return [s[0]]*n+[np.mean(s[i-n:i]) for i in range(n,len(s))]

def EMA(n, s):
    k = 2/(n+1)
    ema = np.zeros(len(s))
    ema[0] = s[0]
    for i in range(1, len(s)-1):
        ema[i] = k*s[i] + (1-k)*ema[i-1]
    return ema

def get_FRED_data(FRED_cache_dir, use_cache=False):
    fr = Fred(api_key=FRED_API_Key, response_type='df')
    try:
        if use_cache:
            df_FRED = pd.read_csv(f'.\{FRED_cache_dir}\FRED.csv', index_col='date', parse_dates=True)
            print('FRED data retrieved from cache')
            FRED_series = pd.read_csv(f'.\{FRED_cache_dir}\FRED_series.csv')[['Series ID', 'Series Title']]
            update_time = pd.Timestamp.strftime(pd.date_range(df_FRED.index[-1], periods=2, freq='D')[1], '%Y-%m-%d')
            df_FRED_update = dl_FRED(FRED_series, update_time)
            df_FRED = df_FRED.append(df_FRED_update, verify_integrity=True, sort=False)
        if not use_cache:
            print('Ignoring cache, downloading from API.')
            FRED_series = get_FRED_series()
            df_FRED = dl_FRED(FRED_series, None)
        FRED_series.to_csv(f'.\{FRED_cache_dir}\FRED_series.csv', index=False)
        df_FRED.to_csv(f'.\{FRED_cache_dir}\FRED.csv')
        print('FRED data and series saved to cache')
        return df_FRED
    except:
        raise Exception('Failed FRED Acquisition.')
        print('Failed FRED Acquisition.')
        return None

def get_FRED_series():
    fr = Fred(api_key=FRED_API_Key, response_type='json')
    dfr = pd.read_json(fr.series.search('daily', response_type='json'))
    fids = []
    ftitles = []
    for i in range(len(dfr)):
        if '(DISCONTINUED)' not in dfr.iloc[i]['seriess']['title']:
            fids.append(dfr.iloc[i]['seriess']['id'])
            ftitles.append(dfr.iloc[i]['seriess']['title'])
    return pd.DataFrame(data={'Series ID': fids, 'Series Title': ftitles})

def dl_FRED(FRED_series, start_time):
    successes, failures = 0, 0
    print('Downloading FRED data...')
    for i, row in FRED_series.iterrows():
        time.sleep(0.5)
        fid, ftitle = row['Series ID'], row['Series Title']
        print('\n', i, '\n', fid, '\n', ftitle)
        if start_time:
            url = f'https://api.stlouisfed.org/fred/series/observations?    \
                    series_id={fid}&api_key={FRED_API_Key}&file_type=json&observation_start={start_time}'
        else:
            url = f'https://api.stlouisfed.org/fred/series/observations?    \
                    series_id={fid}&api_key={FRED_API_Key}&file_type=json'
        print("Acquiring:", ftitle)
        future = session.get(url)
        response = future.result()
        print(response)
        if response.status_code == 200:
            print('Acquired.')
            dfr = pd.read_json(response.content)
            dfi = pd.concat([dfr, dfr['observations'].apply(pd.Series)], axis = 1).drop('observations', axis = 1)
            if len(dfi):
                dfi = dfi[['date', 'value', 'units']]
                dfi['value'] = dfi['value'].apply(float_or_NaN)
                dfi.columns = ['date', ftitle, 'units']
                dfi['date'] = pd.to_datetime(dfi['date'], format='%Y-%m-%d')
                dfi.set_index('date', inplace=True)
                print("Updated.")
                successes += 1
                if successes == 1:
                    df_FRED_ = dfi
                else:
                    df_FRED_ = df_FRED_.merge(dfi, left_index=True, right_index=True, copy=False, how='outer')
            else:
                print("No update available.")
        else:
            print('Acquisition failed.')
            failures += 1
    else:
        print()
        print('Loop finished.')
        print(f'Acquired {successes} / {successes + failures}.')
    return df_FRED_ if successes else None

def get_data(ticker_cache_dir, AV_API_key, ticker, commodity_type, use_cache=False):
    try:
        if use_cache == True:
            # load from cache
            df = pd.read_csv(f'.\{ticker_cache_dir}\{ticker}.csv', index_col='date', parse_dates=True)
            print('Ticker data retrieved from cache.')
        else:
            raise Exception('Ignoring cache, downloading from API.')
            print('Ignoring cache, downloading from API.')
    except:
        # Download from AlphaVantage
        print('Acquiring ticker data...')
        if commodity_type not in ["Physical Currency", "Digital Currency"]:
            data, metadata = ts.get_daily(ticker,'full')
            df = pd.DataFrame(data).transpose().sort_index(axis=0 ,ascending=True).astype('float')
            df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        elif commodity_type == "Physical Currency":
            data, metadata = fx.get_currency_exchange_daily(ticker, 'USD', 'full')
            df = pd.DataFrame(data).transpose().sort_index(axis=0 ,ascending=True).astype('float')
            df.columns = ['open', 'high', 'low', 'close']
            df['volume'] = 0.0
               
        else:
            data, metadata = cc.get_digital_currency_daily(ticker, 'USD')
            df = pd.DataFrame(data).transpose().sort_index(axis=0 ,ascending=True).iloc[:,[0,2,4,6,8,9]].astype('float')
            df.columns = ['open', 'high', 'low', 'close', 'volume', 'market_cap']
        print('Ticker data acquired.')
        
        # datetime index
        df.index.name = 'date'
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
        
        # add a 1-period differenced column
        df['diff1'] = df['close'].diff()/df['close']
        
        # add calendrical categoricals
        for i in ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter']:
            df[i] = getattr(df.index, i)
            if np.int64(0) in list(df[i]):
                df[i] = df[i].apply(lambda x: str(int(x)))
            else:
                df[i] = df[i].apply(lambda x: str(int(x-1)))
                
        # compute MACD and candlesticks
        df['inc'] = df.close > df.open
        df['inc'] = df['inc'].apply(lambda bool: str(bool))
        df['ema12'] = EMA(12, df['open'])
        df['ema26'] = EMA(26, df['open'])
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = EMA(9, df['macd'])
        df['zero'] = df['volume'].apply(lambda x: x*0)
        df['hist'] = df.macd - df.signal
        df['histc'] = df.macd > df.signal
        df['histc'] = df['histc'].apply(lambda bool: str(bool))
              
        # compute technical indicators from TA-Lib  
        inputs = df.loc[:, ['open', 'high', 'low', 'close', 'volume']]
        function_groups = talib.get_function_groups()
        functions = talib.get_functions()
        for function in functions:
            f = talib.abstract.Function(function)
            try:
                y = f(inputs)
                if function in function_groups['Pattern Recognition']:
                    d = {200:'++', 100:'+', 0:'0', -100:'-', -200:'--'}
                    df[pattern_dict[function]] = [d[i] for i in y]
                elif function == 'HT_PHASOR':
                    df['HT_PHASOR_I'], df['HT_PHASOR_Q'] = y['inphase'], y['quadrature']
                elif function == 'HT_SINE':
                    df['HT_SINE'], df['HT_LEADSINE'] = y['sine'], y['leadsine']
                elif function == 'MINMAX':
                    df['MIN'], df['MAX'] = y['min'], y['max']
                elif function == 'MINMAXINDEX':
                    df['MINIDX'], df['MAXIDX'] = y['minidx'], y['maxidx']
                elif  function == 'AROON':
                    df['AROON_DOWN'], df['AROON_UP'] = y['aroondown'], y['aroonup']
                elif function == 'MACD':
                    df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = y['macd'], y['macdsignal'], y['macdhist']
                elif function == 'MACDEXT': # TODO allow custom periods
                    df['MACDext'], df['MACD_SIGNALext'], df['MACD_HISText'] = y['macd'], y['macdsignal'], y['macdhist']
                elif function == 'MACDFIX': # TODO allow custom periods
                    df['MACDfix'], df['MACD_SIGNALfix'], df['MACD_HISTfix'] = y['macd'], y['macdsignal'], y['macdhist']
                elif function == 'STOCH':
                    df['STOCH_slowk'], df['STOCH_slowd'] = y['slowk'], y['slowd']
                elif function == 'STOCHF':
                    df['STOCH_fastk'], df['STOCH_fastd'] = y['fastk'], y['fastd']
                elif function == 'STOCHRSI':
                    df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = y['fastk'], y['fastd']
                elif function == 'BBANDS':
                    df['BB_u'], df['BB_m'], df['BB_l'] = y['upperband'], y['middleband'], y['lowerband']
                elif function == 'MAMA':
                    df['MAMA'], df['FAMA'] = y['mama'], y['fama']
                else:
                    df[function] = y
            except:
                pass
        df.dropna(subset=['close'], inplace=True)
        # store df_ticker in cache
        if use_cache:
            df.to_csv(f'.\{ticker_cache_dir}\{ticker}.csv')
            print('Ticker data cached.')
    
    start_time = df.index[0].strftime(format='%Y-%m-%d')
    return df, start_time

def get_df_cat(df_ticker):
    categorical = ['inc', 'dayofweek', 'month', 'quarter','2 CROWS', '3 BLACK CROWS', '3 INSIDE', '3 LINE STRIKE',
               '3 OUTSIDE', '3 STARS IN THE SOUTH', '3 WHITE SOLDIERS', 'ABANDONED BABY', 'ADVANCE BLOCK', 'BELTHOLD', 'BREAKAWAY', 
               'CLOSING MARUBOZU', 'CONCEAL BABY SWALL', 'COUNTERATTACK', 'DARK CLOUD COVER',  'DOJI', 'DOJI STAR', 'DRAGONFLY DOJI', 
               'ENGULFING', 'EVENING DOJISTAR', 'EVENING STAR', 'GAP-SIDE SIDE-WHITE', 'GRAVESTONE DOJI',  'HAMMER', 'HANGINGMAN', 'HARAMI', 
               'HARAMI CROSS', 'HIGH WAVE', 'HIKKAKE', 'HIKKAKE MOD', 'HOMING PIGEON', 'HT_TRENDMODE', 'IDENTICAL 3 CROWS', 'IN-NECK', 
               'INVERTED HAMMER', 'KICKING', 'KICKING BY LENGTH', 'LADDER BOTTOM', 'LONG LINE', 'LONG-LEGGED DOJI', 'MARUBOZU', 
               'MATCHING LOW',  'MATHOLD', 'MORNING DOJI STAR', 'MORNING STAR', 'ON NECK', 'PIERCING', 'RICKSHAW MAN', 'RISE-FALL 3-METHODS',
               'SEPARATING LINES',  'SHOOTING STAR', 'SHORT LINE', 'SPINNING TOP', 'STALLED PATTERN', 'STICK SANDWICH', 'TAKURI', 
               'TASUKI GAP', 'THRUSTING',  'TRI-STAR', 'UNIQUE 3 RIVER', 'UP-SIDE GAP 2 CROWS', 'X-SIDE GAP 3 METHODS']
 
    df_cat = df_ticker.dropna(subset=['close'])[categorical]
    df_cat = df_cat.fillna(method='ffill')
    df_cat = df_cat.fillna(method='bfill')
    df_cat['HT_TRENDMODE'] = df_cat['HT_TRENDMODE'].apply(lambda x : str(x))   
    print()
    print(list(df_cat['HT_TRENDMODE'].unique()))
    print()

    for col in df_cat.columns:
        if (set(df_cat[col])) == {'False', 'True'}:
            df_cat[col] = df_cat[col].apply(lambda x: {'False':'0', 'True':'1'}[x])

    # setup df_pat for plotting P4    
    df_pat = df_cat[[cat for cat in categorical if cat not in ['inc', 'dayofweek', 'month', 'quarter']]].copy()
    df_pat['HT_TRENDMODE'] = df_pat['HT_TRENDMODE'].apply(lambda x: {'0':'--', '1':'++'}[x])
    df_pat = pd.DataFrame(df_pat.stack(), columns=['signal']).reset_index()
    color_dict = {'--':'red', '-':'lightcoral', '0':'white', '+':'lightgreen', '++':'green'}
    df_pat['color']=df_pat['signal'].apply(lambda x: color_dict[x])

    for col in df_cat.columns:
        if col not in ['day','dayofweek', 'dayofyear', 'month', 'quarter', 'year']:
            df_cat[col+'-1'] = df_cat[col].shift(+1)
            df_cat[col+'-2'] = df_cat[col].shift(+2)
            df_cat[col+'-3'] = df_cat[col].shift(+3)

    df_cat['inc+1'] = df_cat['inc'].shift(-1) # target
    df_cat = df_cat.dropna()   
    
    return df_cat, df_pat

def get_ML_ab(df_cat, train_a, train_b, n_projections):
    df = df_cat.copy()
    y = df['inc+1'] # target
    df = df.drop(columns=['inc+1']) # target removed
    
    # OHE
    for col in df.columns:
        if len(set(df[col])) == 1 or '0' not in set(df[col]):
            df.drop(columns=[col], inplace=True)
    le = LabelEncoder()
    y = le.fit_transform(y)
    enc = OneHotEncoder(handle_unknown='error', drop=['0']*len(df.columns), sparse=True)
    X = enc.fit_transform(df)
    
    print('ok1')
    # split
    end_ind = min(len(df), train_b + n_projections)
    X_train, X_test = X[train_a:train_b,:], X[train_b:end_ind,:]
    y_train, y_test = y[train_a:train_b], y[train_b:end_ind]
    test_dates = df_cat.index[train_b:end_ind].strftime('%Y-%m-%d')
    print('ok2')

    # classify
    predictions = {}
    predictions['Date'] = test_dates
    predictions['Target'] = y_test
    for classifier_name, classifier in list(dict_classifiers.items())[:11]:       
        if classifier_name not in dense:
            classifier.fit(X_train, y_train)
            yhat = classifier.predict(X_test)
        else:
            classifier.fit(X_train.toarray(), y_train)
            yhat = classifier.predict(X_test.toarray())
        predictions[classifier_name] = yhat
    print('ok3')
    df_ML = pd.DataFrame(predictions)
    return df_ML

def get_ARIMA_ab(df_ticker, train_a, train_b, forecast_length=5, n_projections=10):
    
    df_ARIMA = df_ticker[['close', 'diff1']].copy()
    full_series_length = len(df_ARIMA)
    seriest = df_ARIMA['close']
    seriesi = seriest.reset_index(drop=True)
    aa_kwargs = {'start_p':1, 'start_q':1, 'max_p':3, 'max_q':3, 
                 'm':7, 'start_P':0, 'start_Q':1, 'seasonal':True,
                 'd':None, 'D':1, 'trace':True, 'error_action':'ignore', 
                 'suppress_warnings':True, 'stepwise':True, 'njobs':6}
    
    model = pm.auto_arima(seriesi.iloc[train_a:train_b], **aa_kwargs)
    print('SARIMAX hyperparameters determined.')
    model_kwargs = model.arima_res_._init_kwds.copy()
    model_params = model.params()
    start_params = model_params

    fc_cols = [e for t in [(f't+{i}_ciL', f't+{i}_ciU', f't+{i}_FC', f't+{i}_dt') 
                       for i in range(1, forecast_length + 1)] for e in t]
    df_ARIMA = pd.concat([df_ARIMA, pd.DataFrame(columns=fc_cols)], sort=False)
    dates = pd.date_range(df_ARIMA.index[-1], periods=forecast_length+1, freq='B')[1:]
    idx = df_ARIMA.index.append(dates)
    df_ARIMA = df_ARIMA.reindex(idx)
    df_ARIMA.index.name = 'date'
    print('Training and forecasting sliding window models...')
    for k in range(0, min(n_projections, len(seriesi)-train_b)):
        mod = sm.tsa.statespace.SARIMAX(endog=seriesi.iloc[train_a+k:train_b+k], **model_kwargs)
        try:
            fit_res = mod.fit(start_params=start_params, disp=0)
        except:
            pass
        # will use previous fit's values if fit fails
        forecast = fit_res.get_forecast(steps=forecast_length)
        ci = forecast.conf_int()
        ci['mean']=forecast.predicted_mean
        ci['trained_w']=seriest.index[train_b+k-1]
        start_params = fit_res.params
        
        for i in range(forecast_length):
            df_ARIMA.at[df_ARIMA.index[train_b+k+i], f't+{i+1}_ciL'] = ci.iloc[i]['lower close']
            df_ARIMA.at[df_ARIMA.index[train_b+k+i], f't+{i+1}_ciU'] = ci.iloc[i]['upper close']
            df_ARIMA.at[df_ARIMA.index[train_b+k+i], f't+{i+1}_FC'] = ci.iloc[i]['mean']
            df_ARIMA.at[df_ARIMA.index[train_b+k+i], f't+{i+1}_dt'] = ci.iloc[i]['trained_w']

    for i in range(forecast_length):
        df_ARIMA[f't+{i+1}_PE'] =     ( df_ARIMA[f't+{i+1}_FC'] - df_ARIMA['close'] ) / df_ARIMA['close']
        df_ARIMA[f't+{i+1}_PE_ciL'] = ( df_ARIMA[f't+{i+1}_ciL'] - df_ARIMA['close'] ) / df_ARIMA['close']
        df_ARIMA[f't+{i+1}_PE_ciU'] = ( df_ARIMA[f't+{i+1}_ciU'] - df_ARIMA['close'] ) / df_ARIMA['close']
       
    # This would be a good time to cache the dataframe .....
    # df_ARIMA.to_csv(f'.\{ticker_cache_dir}\{ticker}_ARIMA_{train_a}-{train_b}.csv')
    print('ARIMA models trained and forecasted')
    return df_ARIMA

def get_df_hist(df_ARIMA):
    
    def get_hist(col):
        hist, edges = np.histogram(df_ARIMA[col].dropna(), 
                                bins=10, density=True)
        top, bottom, left, right = hist, [0]*10, edges[:-1], edges[1:]
        return top, bottom, left, right
    
    ddict = defaultdict(list)
    for col in ['t+5_PE', 't+4_PE', 't+3_PE', 't+2_PE', 't+1_PE', 'diff1']:
        top, bottom, left, right = get_hist(col)
        ddict['col'].append([col]*10)
        ddict['top'].append(list(top))
        ddict['bottom'].append(bottom)
        ddict['left'].append(list(left))
        ddict['right'].append(list(right))
    ddict = {k:[e for l in v for e in l] for k,v in ddict.items()}
    df_hist = pd.DataFrame(ddict)
    return df_hist

def get_df_hist2(df_ARIMA):
    ddict2 = defaultdict(list)
    for col, clr in zip(['t+5_PE', 't+4_PE', 't+3_PE', 't+2_PE', 't+1_PE', 'diff1'],
                        ['orange', 'yellow', 'green', 'blue', 'purple', 'red']):
        ecdf = ECDF(df_ARIMA[col].dropna())
        x, y = ecdf.x[abs(ecdf.x) < 100], ecdf.y[abs(ecdf.x) < 100]
        colname, clrname = [col]*len(x), [clr]*len(x)
        ddict2['x'].append(x)
        ddict2['y'].append(y)
        ddict2['col'].append(colname)
        ddict2['clr'].append(clrname)
    ddict2 = {k:[e for l in v for e in l] for k,v in ddict2.items()}
    df_hist2 = pd.DataFrame(ddict2)
    return df_hist2

def modify_doc(doc):
    
    # Make widgets
    type_picker = Select(title="Type", value="S&P500", 
                         options=["Digital Currency", 
                                  "Physical Currency", 
                                  "S&P500", 
                                  "NYSE Equity", 
                                  "NASDAQ Equity", 
                                  "AMEX Equity"])
    
    stock_picker = Select(title="Select a Stock",  
                          value=df_0['name'][0], 
                          options=df_0[df_0['type']=="S&P500"]['name'].tolist())
    
    year_picker = Select(title="Select a Year",  
                         value="2018", 
                         options=[str(i) for i in range(2008,2019)], 
                         width=100)
    
    month_picker = Select(title="Select a Month",  
                          value="January", 
                          options=months, 
                          width=150)

    reset_button = Button(label='Full History')
    
    c = column(Div(text="", height=8), 
               reset_button, 
               width=100)

    # Setup data
    df_ticker, start_time = get_data(ticker_cache_dir, AV_API_key, 
                                     ticker, commodity_type, use_cache=False)
    source = ColumnDataSource(data=df_ticker)

    df_cat, df_pat = get_df_cat(df_ticker)
    source_pat = ColumnDataSource(df_pat)

    # Setup forecasts
    series_length = len(df_ticker) 
    train_a, train_b = series_length-40, series_length-10
    forecast_length, n_projections = 5, 10

    df_ARIMA = get_ARIMA_ab(df_ticker, train_a, train_b, 
                            forecast_length=forecast_length, 
                            n_projections=n_projections)   
    sourceA = ColumnDataSource(data=df_ARIMA)

    df_hist = get_df_hist(df_ARIMA)
    source_hist = ColumnDataSource(data=df_hist)

    df_hist2 = get_df_hist2(df_ARIMA)
    source_hist2 = ColumnDataSource(data=df_hist2)

    df_ML = get_ML_ab(df_cat, train_a, train_b, n_projections=n_projections)
    source_ML = ColumnDataSource(data=df_ML)
    
    ########################
    ###### Setup Plots #####
    ########################

    width_ms = 12*60*60*1000 # half day in ms
    color_mapper = CategoricalColorMapper(factors=["True", "False"], 
                                          palette=["green", "red"])
    # Make P1 Plot (Candlesticks)
    def makeP1():
        p1 = figure(plot_height=400, x_axis_type="datetime", tools="xwheel_pan,xwheel_zoom,pan,box_zoom", active_scroll='xwheel_zoom')
        bst = p1.add_tools(BoxSelectTool(dimensions="width"))
        # Create price glyphs and volume glyph
        p1O = p1.line(x='date', y='open', source=source, color=Spectral5[0], alpha=0.8, legend="OPEN")
        p1C = p1.line(x='date', y='close', source=source, color=Spectral5[1], alpha=0.8, legend="CLOSE")
        p1L = p1.line(x='date', y='low', source=source, color=Spectral5[4], alpha=0.8, legend="LOW")
        p1H = p1.line(x='date', y='high', source=source, color=Spectral5[3], alpha=0.8, legend="HIGH")   
        p1.line(x='date', y='ema12', source=source, color="magenta", legend="EMA-12")
        p1.line(x='date', y='ema26', source=source, color="black", legend="EMA-26")
        p1.segment(x0='date', y0='high', x1='date', y1='low', color={'field': 'inc', 'transform': color_mapper}, source=source)
        p1.vbar(x='date', width=width_ms, top='open', bottom='close', color={'field': 'inc', 'transform': color_mapper}, source=source)
        # Add axis labels
        p1.yaxis.axis_label = '\n \n \n \n \n \n \n \n Price ($USD/share)'
        # Add legend
        p1.legend.orientation = 'horizontal'
        p1.legend.title = 'Daily Stock Price'
        p1.legend.click_policy="hide"
        p1.legend.location="top_left"
        # Add HoverTools
        p1.add_tools(HoverTool(tooltips=[('Date','@date{%F}'),('Open','@open{($ 0.00)}'),('Close','@close{($ 0.00)}'),
                                        ('Low','@low{($ 0.00)}'),('High','@high{($ 0.00)}'),('Volume','@volume{(0.00 a)}')],
                               formatters={'date': 'datetime'},mode='mouse'))
        p1.toolbar.logo = None
        # Formatting
        p1.yaxis[0].formatter = NumeralTickFormatter(format="$0.00")
        p1.outline_line_width = 1
        return p1
    p1 = makeP1()

    # Make P2 Plot (Volume)
    def makeP2():
        p2 = figure(plot_height=150, x_axis_type="datetime", x_range=p1.x_range, tools="xwheel_pan,xwheel_zoom,pan,box_zoom", active_scroll='xwheel_pan')
        p2V = p2.varea(x='date', y1='volume', y2='zero', source=source, color="black", alpha=0.8)
        p2.add_tools(HoverTool(tooltips=[('Date','@date{%F}'),('Open','@open{($ 0.00)}'),('Close','@close{($ 0.00)}'),
                                        ('Low','@low{($ 0.00)}'),('High','@high{($ 0.00)}'),('Volume','@volume{(0.00 a)}')],
                            formatters={'date': 'datetime'},mode='mouse'))
        p2.toolbar.logo = None
        p2.xaxis.axis_label = 'Date'
        p2.yaxis.axis_label = '\n \n \n \n \n Volume (shares)'
        p2.yaxis[0].formatter = NumeralTickFormatter(format="0.0a")
        p2.outline_line_width = 1
        return p2
    p2 = makeP2()

    # Make P3 Plot (MACD-Signal) 
    def makeP3():
        p3 = figure(plot_height=150, x_axis_type="datetime", x_range=p1.x_range, tools="xwheel_pan,xwheel_zoom,pan,box_zoom", active_scroll='xwheel_pan')   
        p3.line(x='date', y='macd', source=source, color="green", legend="-MACD")
        p3.line(x='date', y='signal', source=source, color="red", legend="-Signal")
        p3.vbar(x='date', top='hist', source=source, width=width_ms, color={'field': 'histc', 'transform': color_mapper}, alpha=0.5)
        # Add HoverTools
        p3.add_tools(HoverTool(tooltips=[('Date','@date{%F}'),('EMA-12','@ema12{($ 0.00)}'),('EMA-26','@ema26{($ 0.00)}'),
                                        ('MACD','@macd{($ 0.00)}'),('Signal','@signal{($ 0.00)}')],
                            formatters={'date': 'datetime'},mode='mouse'))
        p3.toolbar.logo = None
        # Add legend
        p3.legend.orientation = 'horizontal'
        p3.legend.location="top_left"
        p3.legend.orientation = 'horizontal'
        p3.legend.title = 'Moving Average Convergence Divergence'
        p3.legend.location="top_left"
        p3.legend.label_text_font_size = '12pt'
        p3.legend.glyph_height = 1 #some int
        #p3.legend.glyph_width = 12 #some int
        # Add axis labels
        p3.yaxis.axis_label = '\n \n \n \n \n Indicator ($USD)'
        # Add tick formatting
        p3.yaxis[0].formatter = NumeralTickFormatter(format="$0.00")
        p3.outline_line_width = 1
        return p3
    p3 = makeP3()

    # Make P4 Plot (Price-Action Event Indicators)
    def makeP4():          
        df_pat = df_ticker[[col for col in pattern_dict.values()]]
        df_pat.columns.name = 'pattern'
   
        # D, L sort event types so 
        # the most common events are at top of plot
        D = {col:df_pat[col].value_counts().to_dict()['0'] 
                 for col in df_pat.columns}
        
        L = [i[0] for i in sorted(D.items(), 
                                  key=lambda kv:(kv[1], kv[0]), 
                                  reverse=True)]
        
        df_pat['HT_TRENDMODE'] = df_cat['HT_TRENDMODE'].apply(lambda x: {'0':'--', '1':'++'}[x])
        df_pat2 = pd.DataFrame(df_pat.stack(), columns=['signal']).reset_index()
        color_dict = {'--':'red', '-':'lightcoral', '0':'white', '+':'lightgreen', '++':'green'}
        df_pat2['color']=df_pat2['signal'].apply(lambda x: color_dict[x])
        source_pat2 = ColumnDataSource(df_pat2)
              
        p4 = figure(plot_width=800, plot_height=800, y_range=L,
                    x_axis_location="above", x_axis_type="datetime", 
                    x_range=p1.x_range)
        
        p4.rect(x="date", y="pattern", width=24*60*60*1000, height=1, 
                source=source_pat2, line_color=None, line_width=0.2, 
                fill_color='color')
        return p4
    p4 = makeP4()

    ind_end = min(len(df_ARIMA) -1 , train_b + n_projections + forecast_length + 30)    
    # Make P21 Plot (ARIMA Training - Price)
    def make_P21():
        p21 = figure(plot_width=600, plot_height=250, x_axis_type="datetime",
                     tools="xwheel_pan,xwheel_zoom,pan,box_zoom", active_scroll='xwheel_zoom')
        p21.scatter(x='date', y='t+5_FC', source=sourceA, color='orange', alpha=0.1, legend='T+5')
        p21.varea(x='date', y1='t+5_ciU', y2='t+5_ciL', source=sourceA, color='orange', alpha=0.1, legend='T+5')
        p21.scatter(x='date', y='t+4_FC', source=sourceA, color='yellow', alpha=0.1, legend='T+4')
        p21.varea(x='date', y1='t+4_ciU', y2='t+4_ciL', source=sourceA, color='yellow', alpha=0.1, legend='T+4')
        p21.scatter(x='date', y='t+3_FC', source=sourceA, color='green', alpha=0.1, legend='T+3')
        p21.varea(x='date', y1='t+3_ciU', y2='t+3_ciL', source=sourceA, color='green', alpha=0.1, legend='T+3')
        p21.scatter(x='date', y='t+2_FC', source=sourceA, color='blue', alpha=0.1, legend='T+2')
        p21.varea(x='date', y1='t+2_ciU', y2='t+2_ciL', source=sourceA, color='blue', alpha=0., legend='T+2')
        p21.scatter(x='date', y='t+1_FC', source=sourceA, color='purple', alpha=0.1, legend='T+1')
        p21.varea(x='date', y1='t+1_ciU', y2='t+1_ciL', source=sourceA, color='purple', alpha=0.1, legend='T+1')
        p21.line(x='date', y='close', source=sourceA, color='black', legend='ACTUAL')
        p21.yaxis.axis_label = 'Price ($USD/Share)'
        p21.xaxis.axis_label = 'Date'
        p21.legend.title = 'Forecast'
        p21.legend.location = 'top_left'
        p21.x_range = Range1d(df_ARIMA.index[max(0, train_a - 10)], df_ARIMA.index[ind_end])
        cols=['t+5_ciU', 't+5_ciL', 'close']
        ymin=0.9 * df_ARIMA[cols].iloc[max(0, train_a - 10):ind_end].min().min()
        ymax=1.1 * df_ARIMA[cols].iloc[max(0, train_a - 10):ind_end].max().max()
        p21.y_range = Range1d(ymin, ymax)
        box21 = BoxAnnotation(left=df_ARIMA.index[train_a], right=df_ARIMA.index[train_b - 1], fill_color='red', fill_alpha=0.1)
        p21.add_layout(box21)
        return p21, box21
    p21, box21 = make_P21()

    # Make P22 Plot (ARIMA Training - Price Diff)
    def make_P22():
        p22 = figure(plot_width=600, plot_height=250, x_axis_type="datetime", x_range=p21.x_range,
                     tools="xwheel_pan,xwheel_zoom,pan,box_zoom", active_scroll='xwheel_pan')
        
        p22.varea(x='date', y1='t+5_PE_ciU', y2='t+5_PE_ciL', source=sourceA, color='orange', alpha=0.1, legend='T+5')
        p22.line(x='date', y='t+5_PE', source=sourceA, color='orange', alpha=0.1, legend='T+5')
        p22.scatter(x='date', y='t+5_PE', source=sourceA, color='orange', alpha=0.1, legend='T+5')
        p22.varea(x='date', y1='t+4_PE_ciU', y2='t+4_PE_ciL', source=sourceA, color='yellow', alpha=0.1, legend='T+4')
        p22.line(x='date', y='t+4_PE', source=sourceA, color='yellow', alpha=0.1, legend='T+4')
        p22.scatter(x='date', y='t+4_PE', source=sourceA, color='yellow', alpha=0.1, legend='T+4')
        p22.varea(x='date', y1='t+3_PE_ciU', y2='t+3_PE_ciL', source=sourceA, color='green', alpha=0.1, legend='T+3')
        p22.line(x='date', y='t+3_PE', source=sourceA, color='green', alpha=0.1, legend='T+3')
        p22.scatter(x='date', y='t+3_PE', source=sourceA, color='green', alpha=0.1, legend='T+3')
        p22.varea(x='date', y1='t+2_PE_ciU', y2='t+2_PE_ciL', source=sourceA, color='blue', alpha=0.1, legend='T+2')
        p22.line(x='date', y='t+2_PE', source=sourceA, color='blue', alpha=0.1, legend='T+2')
        p22.scatter(x='date', y='t+2_PE', source=sourceA, color='blue', alpha=0.1, legend='T+2')
        p22.varea(x='date', y1='t+1_PE_ciU', y2='t+1_PE_ciL', source=sourceA, color='purple', alpha=0.1, legend='T+1')
        p22.line(x='date', y='t+1_PE', source=sourceA, color='purple', alpha=0.1, legend='T+1')
        p22.scatter(x='date', y='t+1_PE', source=sourceA, color='purple', alpha=0.1, legend='T+1')
        # let's make a comparison: what are the corresponding PE's from the "day-behind" model
        p22.line(x='date', y='diff1', source=sourceA, color='red', alpha=0.4, legend='yest')
        
        p22.yaxis.axis_label = 'Error'
        p22.xaxis.axis_label = 'Date'
        p22.legend.title = 'Forecast'
        p22.legend.location = 'top_left'
        cols=['t+5_PE_ciU', 't+5_PE_ciL', 'diff1']
        p22.y_range = Range1d(1.1 * df_ARIMA[cols].iloc[max(0, train_a - 10):ind_end].min().min(),
                            1.1 * df_ARIMA[cols].iloc[max(0, train_a - 10):ind_end].max().max())
        box22 = BoxAnnotation(left=df_ARIMA.index[train_a], right=df_ARIMA.index[train_b - 1], fill_color='red', fill_alpha=0.1)
        p22.add_layout(box21)
        return p22, box22
    p22, box22 = make_P22()
      
    # Make P23 Plot (ARIMA Error - Dist)
    def make_P23():
        # Make CDSviews for each forecast set
        view5 = CDSView(source=source_hist,
                        filters=[GroupFilter(column_name='col', group='t+5_PE')])
        view4 = CDSView(source=source_hist,
                        filters=[GroupFilter(column_name='col', group='t+4_PE')])
        view3 = CDSView(source=source_hist,
                        filters=[GroupFilter(column_name='col', group='t+3_PE')])
        view2 = CDSView(source=source_hist,
                        filters=[GroupFilter(column_name='col', group='t+2_PE')])
        view1 = CDSView(source=source_hist,
                        filters=[GroupFilter(column_name='col', group='t+1_PE')])
        viewD = CDSView(source=source_hist,
                        filters=[GroupFilter(column_name='col', group='diff1')])
        
        p23 = figure(plot_height=250, plot_width=200)
        
        p23.quad(source=source_hist, view=view5, top='top', bottom='bottom', 
                left='left', right='right', fill_color='orange', 
                line_color="white", alpha=0.1)

        p23.quad(source=source_hist, view=view4, top='top', bottom='bottom', 
                left='left', right='right', fill_color='yellow', 
                line_color="white", alpha=0.1)

        p23.quad(source=source_hist, view=view3, top='top', bottom='bottom', 
                left='left', right='right', fill_color='green', 
                line_color="white", alpha=0.1)
                
        p23.quad(source=source_hist, view=view2, top='top', bottom='bottom', 
                left='left', right='right', fill_color='blue', 
                line_color="white", alpha=0.1)

        p23.quad(source=source_hist, view=view1, top='top', bottom='bottom', 
                left='left', right='right', fill_color='purple', 
                line_color="white", alpha=0.1)

        p23.quad(source=source_hist, view=viewD, top='top', bottom='bottom', 
                left='left', right='right', fill_color="purple", 
                fill_alpha=0, line_color="red", alpha=1)
        p23.xaxis.axis_label = 'Error'
        return p23
    p23 = make_P23()

    #Make P24 Plot (ARIMA Error - CumDist)
    def make_P24():
        # Make CDSviews for each forecast set
        view5_2 = CDSView(source=source_hist2,
                        filters=[GroupFilter(column_name='col', group='t+5_PE')])
        view4_2 = CDSView(source=source_hist2,
                        filters=[GroupFilter(column_name='col', group='t+4_PE')])
        view3_2 = CDSView(source=source_hist2,
                        filters=[GroupFilter(column_name='col', group='t+3_PE')])
        view2_2 = CDSView(source=source_hist2,
                        filters=[GroupFilter(column_name='col', group='t+2_PE')])
        view1_2 = CDSView(source=source_hist2,
                        filters=[GroupFilter(column_name='col', group='t+1_PE')])
        viewD_2 = CDSView(source=source_hist2,
                        filters=[GroupFilter(column_name='col', group='diff1')])
        
        # panel showing empCDF of forecast and diff1
        p24 = figure(plot_height=250, plot_width=200)
        p24.line(source=source_hist2, view=view5_2, x='x', y = 'y', color='orange')
        p24.line(source=source_hist2, view=view4_2, x='x', y = 'y', color='yellow')
        p24.line(source=source_hist2, view=view3_2, x='x', y = 'y', color='green')
        p24.line(source=source_hist2, view=view2_2, x='x', y = 'y', color='blue')
        p24.line(source=source_hist2, view=view1_2, x='x', y = 'y', color='purple')
        p24.line(source=source_hist2, view=viewD_2, x='x', y = 'y', color='red')
        p24.xaxis.axis_label = 'Error'
        return p24
    p24 = make_P24()

    # Make Bokeh DataTable from ML predictions df
    def make_PT():
        columns = [TableColumn(field=col, title=col) 
                          for col in df_ML.columns]                  
        pT = DataTable(source=source_ML, columns=columns, 
                       editable=False, width=800, height=200)
        return pT
    pT = make_PT()
    
    #######################
    ###### Callbacks ######
    #######################

    #Define JavaScript callbacks
    JS = '''
        clearTimeout(window._autoscale_timeout);

        var date = source.data.date,
            low = source.data.low,
            high = source.data.high,
            volume = source.data.volume,
            macd = source.data.macd,
            start = cb_obj.start,
            end = cb_obj.end,
            min1 = Infinity,
            max1 = -Infinity,
            min2 = Infinity,
            max2 = -Infinity,
            min3 = Infinity,
            max3 = -Infinity;

        for (var i=0; i < date.length; ++i) {
            if (start <= date[i] && date[i] <= end) {
                max1 = Math.max(high[i], max1);
                min1 = Math.min(low[i], min1);
                max2 = Math.max(volume[i], max2);
                min2 = Math.min(volume[i], min2);
                max3 = Math.max(macd[i], max3);
                min3 = Math.min(macd[i], min3);
            }
        }
        var pad1 = (max1 - min1) * .05;
        var pad2 = (max2 - min2) * .05;
        var pad3 = (max3 - min3) * .05;

        window._autoscale_timeout = setTimeout(function() {
            y1_range.start = min1 - pad1;
            y1_range.end = max1 + pad1;
            y2_range.start = min2 - pad2;
            y2_range.end = max2 + pad2;
            y3d = Math.max(Math.abs(min3 - pad3), Math.abs(max3 + pad3))
            y3_range.start = -y3d;
            y3_range.end = y3d;
        });
    '''
    callbackJS = CustomJS(args={'y1_range': p1.y_range, 
                                'y2_range': p2.y_range, 
                                'y3_range': p3.y_range, 
                                'source': source}, code=JS)
    p1.x_range.js_on_change('start', callbackJS)
    p1.x_range.js_on_change('end', callbackJS)

    # reset history button
    reset_button.js_on_click(CustomJS(args=dict(p1=p1, p2=p2), code="""
    p1.reset.emit()
    p2.reset.emit()
    """))

    # Define Python callbacks
    def update_bst(attrname, old, new):
        print('updating bst')
        r = source.selected.indices
        train_a, train_b = min(r), max(r)
        
        # retrain and update predictions
        print('retraining ARIMA')
        df_ARIMA = get_ARIMA_ab(df_ticker, train_a, train_b, 
                                forecast_length=5, n_projections=n_projections)
        sourceA.data = ColumnDataSource(df_ARIMA).data
        
        df_hist = get_df_hist(df_ARIMA)
        source_hist.data = ColumnDataSource(data=df_hist).data

        df_hist2 = get_df_hist2(df_ARIMA)
        source_hist2.data = ColumnDataSource(data=df_hist2).data
        
        print('refitting ML')
        df_ML = get_ML_ab(df_cat, train_a, train_b, 
                          n_projections=n_projections)
        source_ML.data = ColumnDataSource(data=df_ML).data
 
        ind_end = min(len(df_ARIMA) - 1, train_b + n_projections + forecast_length + 30) 
        # update p21 plot
        print('updating p21')
        p21.x_range.start = df_ARIMA.index[max(0, train_a - 10)]
        p21.x_range.end = df_ARIMA.index[ind_end]
        cols=['t+5_ciU', 't+5_ciL', 'close']
        ymin=0.9 * df_ARIMA[cols].iloc[max(0, train_a - 10):
                                       ind_end].min().min()
        ymax=1.1 * df_ARIMA[cols].iloc[max(0, train_a - 10):
                                       ind_end].max().max()
        p21.y_range.start = ymin
        p21.y_range.end =  ymax
        box21.left = df_ARIMA.index[train_a]
        box21.right = df_ARIMA.index[train_b - 1]
        
        # update p22 plot
        print('updating p22')
        cols=['t+5_PE_ciU', 't+5_PE_ciL', 'diff1']
        p22.y_range.start = 1.1 * df_ARIMA[cols].iloc[max(0, train_a - 10):
                                                      ind_end].min().min()
        p22.y_range.end =   1.1 * df_ARIMA[cols].iloc[max(0, train_a - 10):
                                                      ind_end].max().max()
        box22.left = df_ARIMA.index[train_a]
        box22.right = df_ARIMA.index[train_b - 1]

    def update_widget(attrname, old, new):
        commodity_type = type_picker.value
        stock_picker.options = df_0[df_0['type']==commodity_type]['name'].tolist()
    
    def update_data(attrname, old, new):
        # Get the current Select value
        commodity_type = type_picker.value
        print('type:', commodity_type)
        ticker = df_0.loc[df_0['name'] == stock_picker.value, 'code'].iloc[0]
        print('ticker:', ticker)
 
        # Get the new data
        df_ticker, start_time = get_data(ticker_cache_dir, AV_API_key, ticker, commodity_type, use_cache=False)
        source.data = ColumnDataSource(data=df_ticker).data
        
        df_cat, df_pat = get_df_cat(df_ticker)
        source_pat.data = ColumnDataSource(df_pat).data
        
        p1.x_range.start = df_ticker.index[0]
        p1.x_range.end = df_ticker.index[-1]
        reset_on_update=True
        if reset_on_update:
            CustomJS(args=dict(p1=p1, p2=p2), code="""p1.reset.emit()
                                                      p2.reset.emit()""")
        
    def update_axis(attrname, old, new):
        # Get the current Select values
        source.data = ColumnDataSource(data=df_ticker).data
        year = year_picker.value
        month = f'{months.index(month_picker.value) + 1:02d}'   
        start = datetime.strptime(f'{year}-{month}-01', "%Y-%m-%d")
        if month == '12':
            end = datetime.strptime(f'{str(int(year)+1)}-01-01', "%Y-%m-%d")
        else:
            end = datetime.strptime(f'{year}-{int(month)+1:02d}-01', "%Y-%m-%d")     
        p1.x_range.start = start
        p1.x_range.end = end
        # dfi = df_ticker.set_index(['date'])
        p1.y_range.start = df_ticker.loc[end:start]['low'].min()*0.95
        p1.y_range.end = df_ticker.loc[end:start]['high'].max()*1.05
        p2.y_range.start = df_ticker.loc[end:start]['volume'].min()*0.95
        p2.y_range.end = df_ticker.loc[end:start]['volume'].max()*1.05
        p3.y_range.start = df_ticker.loc[end:start]['macd'].min()*0.75
        p3.y_range.end = df_ticker.loc[end:start]['macd'].max()*1.25

    # Route Python callbacks
    source.selected.on_change('indices', update_bst)
    type_picker.on_change('value', update_widget)
    stock_picker.on_change('value', update_data)
    year_picker.on_change('value', update_axis)
    month_picker.on_change('value', update_axis)
    
    # Set up layouts and add to document
    row1 = row(type_picker, stock_picker, year_picker, month_picker, c,
               height=65, width=800, sizing_mode='stretch_width')
    row2 = row(p1, width=800, height=400, sizing_mode='stretch_width')
    row3 = row(p2, width=800, height=150, sizing_mode='stretch_width')
    row4 = row(p3, width=800, height=150, sizing_mode='stretch_width')
    row5 = row(p4, width=800, height=800, sizing_mode='stretch_width')
    row6 = row(p21, p23, width=800, height=250, sizing_mode='stretch_width')
    row7 = row(p22, p24, width=800, height=250, sizing_mode='stretch_width')
    row8 = row(pT, width=800, height=800, sizing_mode='stretch_width')
    layout = column(row1, row2, row4, row3, row5, row6, row7, row8,
                    width=800, height=2265, sizing_mode='stretch_both')
    doc.add_root(layout)

modify_doc(curdoc())