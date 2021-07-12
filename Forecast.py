import os
import time

import eikon as ek
import numpy as np
import pandas as pd
import xlwings as xw

ek.set_app_id('604ea22425e048d39af0ef760ec9c64ebbe1fe68')
import pickle
import zipfile

import statsmodels.api as sm
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import functions


@xw.func
@xw.ret(expand='table')
def get_traindata(headers,tickers,FX,start_date='2015-01-01', month_dummies = 0):
    tickers_train = dict(zip(headers,tickers))
    dataset = functions.get_train_data(tickers_train, ek,start_date)
    for i in range(len(FX)):
        if FX[i]=="USD":
            dataset[headers[i]+'_CLOSE'] = dataset[headers[i]+'_CLOSE']/dataset['EUR_USD_CLOSE']
        if FX[i]=="GBP":
            dataset[headers[i]+'_CLOSE'] = dataset[headers[i]+'_CLOSE']/dataset['EUR_GBP_CLOSE']

    try:
        dataset = dataset.drop(['EUR_USD_CLOSE'], axis=1)
    except:
        pass
    try:
        dataset = dataset.drop(['EUR_GBP_CLOSE'], axis=1)
    except:
        pass
    if month_dummies == 1:
        #one-hot encoding of months
        dataset['month'] = dataset['Date'].dt.month  
        y = pd.get_dummies(dataset.month,prefix='Month')
        dataset = pd.concat([dataset,y],axis = 1)
        dataset = dataset.drop(['month'], axis=1)
        dataset.index = np.arange(0,len(dataset))
        dataset = dataset.drop(['Month_12'], axis=1)        
    dataset = dataset.dropna()
    return dataset


#@xw.func
@xw.ret(expand='table')
def linear_regression(FileName,x,y,i=0):
    est = sm.OLS(y, x)
    est = est.fit()      
    if i == 0:
        results_as_html1 = est.summary().tables[0].as_html()
        test = pd.read_html(results_as_html1, header=0, index_col=0)[0]
    else:    
        results_as_html2 = est.summary().tables[1].as_html()
        test = pd.read_html(results_as_html2, header=0, index_col=0)[0]
    cwd = os.getcwd()
    #filename = 'D:\Corporate\SaoPaulo\pmendes\Projects\Python\INSEAD2\TTF_regression\TTF_REGRESSION.sav'
    filename = FileName
    pickle.dump(est, open(filename, 'wb'))
    return test


@xw.func
#@xw.arg('dataset','table')
@xw.ret(expand='table')
def corr_matrix(dataset):
    dataset =  pd.DataFrame(dataset)
    return dataset.corr()

@xw.func
#@xw.arg('dataset','table')
@xw.ret(expand='table')
def forecast_reg(filename,x):
    reg2 = pickle.load(open(filename, 'rb'))
    forecast = reg2.predict(x)
    return (pd.DataFrame(forecast)).values


@xw.func
@xw.ret(expand='table')
def fair_value_gap(pastXrange,pastYrange,currentXrange,standardize=0):
    #try:
    if standardize == 1:
        scaler = MinMaxScaler() 
        pastXrange = scaler.fit_transform(pastXrange)  
        currentXrange = scaler.transform(np.array(currentXrange).reshape(1,-1))
    est = sm.OLS(pastYrange, pastXrange)
    est = est.fit()        
   
    forecast = est.predict(currentXrange)
    #return est.rsquared_adj
    return (pd.DataFrame(forecast)).values
    #except:
    #    pass


@xw.func
#@xw.arg('x','table')
@xw.ret(expand='table')
def z_score(x):    
    try:
        return ((pd.DataFrame(stats.zscore(x))).values)[-1]
    except:
        pass
    
@xw.func
#@xw.arg('x','table')
@xw.ret(expand='table')
def z_score_diff(x,y):    
    try:
        return ((pd.DataFrame(stats.zscore(x)-stats.zscore(y))).values)[-1]
    except:
        pass
