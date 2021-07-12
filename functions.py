import time
import pandas as pd
import eikon as ek
import numpy as np
import datetime 

def get_train_data(tickers_train, ek,startdate):
#     startdate='1000-01-20T15:04:05'
    #startdate='2015-01-01'
    current_date=time.strftime("%Y-%m-%d")
    dataset = pd.DataFrame()
    error = 0
    for key in tickers_train:
            try:
                ticker = tickers_train[key] #brent front month USD/BBL
                DFname = key
                DFname = pd.DataFrame() #dynamically name dataframe
                DFname = ek.get_timeseries(ticker,start_date=startdate,end_date=current_date)   
                DFname = DFname['CLOSE']
                DFname = DFname.reset_index()
                DFname = DFname.rename(columns = {'CLOSE' : key+'_CLOSE'})    
    #             display(DFname)
            except:
                print(ticker + ' FAILED')  
                error = 1
                while error == 1:
                    try:
                        DFname = ek.get_timeseries(ticker,start_date=startdate,end_date=current_date)   
                        DFname = DFname['CLOSE']
                        DFname = DFname.reset_index()
                        DFname = DFname.rename(columns = {'CLOSE' : key+'_CLOSE'})    
                        error = 0
                    except:
                        print(ticker + ' FAILED AGAIN') 
                        

            if len(dataset)==0:
                dataset = DFname
            else:
                dataset = pd.merge_asof(dataset,DFname, on='Date')
    return dataset


def forecast_storage(dataset,on_date, year,month):
    #this function forecasts the expected storage level "on_date" for the "month" in given "year"
    
    #get historic data and calculate daily changes in storage levels
#     tickers_train = {"Storage Levels" : "NGAS-TOTD-GIE"}
#     dataset = get_train_data(tickers_train, ek)
    dataset["Change"] = dataset["Storage Levels_CLOSE"] - dataset["Storage Levels_CLOSE"].shift(1)
    dataset = dataset.fillna(0)
#     display(dataset)
    #for each year in the "years" list, start with the current storage level and apply the daily change for that year
    years = ['2013','2014','2015','2016','2017','2018','2019','2020']
#     date = datetime.datetime.strptime(on_date, '%Y-%m-%d')
    date = on_date
    level = dataset.loc[dataset["Date"]==date]['Storage Levels_CLOSE'].tolist()[0]
    matrix = pd.DataFrame(columns=['Date'] + years)
    matrix.loc[0] = [date,level,level,level,level,level,level,level,level]

    for j in range(1,750):
        date = date + datetime.timedelta(days=1)    
        row = tuple()
        row = row+(date,)
        for i in range(len(years)):
            try:
                lookup_date = date
                lookup_date = lookup_date.replace(year=int(years[i]))
                change = dataset.loc[dataset["Date"]==lookup_date]['Change'].tolist()[0]
            except:
                pass
            row = row + (matrix.loc[j-1][years[i]]+change,)
        matrix.loc[j] = row
    matrix['Average'] = matrix.mean(axis=1)
    matrix['Year'] = matrix['Date'].dt.year
    matrix['Month'] = matrix['Date'].dt.month
    matrix = matrix.groupby(['Year','Month'], as_index=False).mean()
    matrix.index = np.arange(0,len(matrix))
    return int(matrix[matrix['Year']==year][matrix['Month']==month]['Average'])
#look ahead

