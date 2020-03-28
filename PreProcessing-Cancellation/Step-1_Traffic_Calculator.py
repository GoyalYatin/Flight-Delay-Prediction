# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:19:37 2019

@author: bsoni
"""
import datetime
import pandas as pd
import numpy as np
from datetime import datetime as dt

#import merged flights dataset 
data = pd.read_csv('2017flights.csv')

#for the missing values of dep time. Fill the values with crs_dep_time as the flight will be cancelled
#fill any non occuring value to arrival time of cancelled flights
data['DEP_TIME'] = data['DEP_TIME'].fillna(value=data['CRS_DEP_TIME'])
data['ARR_TIME'] = data['ARR_TIME'].fillna(value=-100)

#extract hour from time columns
data['DEP_HOUR'] = data['DEP_TIME'].apply(lambda x: int(x/100))
data['ARR_HOUR'] = data['ARR_TIME'].apply(lambda x: int(x/100))

#replace 0 value with 24
#this will make us easy to see time
data = data.replace({'DEP_HOUR': {24: 0}})
data = data.replace({'ARR_HOUR': {24: 0}})


#select only that origin and destination whose weather data is available to us
data = data[(data.ORIGIN == "BOS") | (data.ORIGIN == "EWR") | (data.ORIGIN == "JFK") | 
        (data.ORIGIN == "LGA") | (data.ORIGIN == "ORD") | (data.ORIGIN == "DEN") |
        (data.ORIGIN == "DFW") | (data.ORIGIN == "IAH") | (data.ORIGIN == "PHL") | (data.ORIGIN == "SFO")]

data = data[(data.DEST == "BOS") | (data.DEST == "EWR") | (data.DEST == "JFK") |
        (data.DEST == "LGA") | (data.DEST == "ORD") | (data.DEST == "DEN") | (data.DEST == "DFW") | 
        (data.DEST == "IAH") | (data.DEST == "PHL") | (data.DEST == "SFO")]

#add a traffic column initialized with 0
data['traffic'] = 0

#make a unique list for month and origin
#this will help us to calculate the traffic
month = data.MONTH.unique()
k=0
orgs = data.ORIGIN.unique()

#the below code is to calculate the traffic 
#It iterates over each hour of a day of every month
#At that particular hour it see how many arriving/departing flights are there
#it appends taht value to the traffic column for that particuar hour at that particular city
for m in month:
    df = data[data['MONTH']==m]
    print('month is',m)
    day = df.DAY_OF_MONTH.unique()
    for i in day:
        df1 = df[df['DAY_OF_MONTH']==i]
        dephr = data.DEP_HOUR.unique()
        for j in dephr:
            for o in orgs:
                df2 = df1[(df1.DEP_HOUR == j) & (df1.ORIGIN == o)]
                df3 = df1[(df1.ARR_HOUR == j) & (df1.DEST == o)]
                dff = [df2,df3]
                dfff = pd.concat(dff)
                count_traffic = dfff.shape[0]
                print('count traffic at ',j,'hour is ',count_traffic)
                data.loc[(data['ORIGIN']==o)&(data['DAY_OF_MONTH']==i)&(data['DEP_HOUR']==j),'traffic']=count_traffic

                    

#retain the data again            
data.loc[data['CANCELLED']==1,'DEP_TIME']=np.nan
data.loc[data['CANCELLED']==1,'DEP_HOUR']=np.nan
data.loc[data['CANCELLED']==1,'ARR_TIME']=np.nan
data.loc[data['CANCELLED']==1,'ARR_HOUR']=np.nan

#finally!!! save to csv
data.to_csv('2017flightwithtraffic.csv')