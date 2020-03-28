# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:17:44 2019

@author: bsoni
"""

import datetime
import pandas as pd
import numpy as np
from datetime import datetime as dt

#import the traffic calculated flights data 
#drop the un-necessary columns
data = pd.read_csv('2017flightwithtraffic.csv')
data = data.drop('WHEELS_ON', axis=1)

#Constantly keep checking datatypes 
data.dtypes

#Separate cancelled and non-cancelled flights data
#Because both have diiferent type of columns that need to be converted
dataNCa = data[data['CANCELLED']==0]
dataCa= data[data['CANCELLED']==1]

#create a list for the columns which need conversion for non-cancelled data
dateList = ['CRS_DEP_TIME','DEP_TIME','WHEELS_OFF','CRS_ARR_TIME','ARR_TIME']

#Below is the code for converting the columns for non-cancelled data to make them suitable for converting
#the columns to UTC format
#Take a close look on dtypes of data at each step
#Temp time is made to Convert FL_date to datetime format
#Make sure that no na values is present in data
for i in dateList:
    dataNCa[i] = dataNCa[i].apply(lambda x: int(x))
    dataNCa = dataNCa.ix[dataNCa[i] > 0]
    dataNCa.dtypes
    dataNCa['tempHr'] = dataNCa[i].apply(lambda x: int(x/100))
    dataNCa = dataNCa.replace({'tempHr': {24:0 }})
    dataNCa['tempHr'] = dataNCa['tempHr'].apply(lambda x: str(x))
    dataNCa['tempMin'] = dataNCa[i].apply(lambda x: int(x%100))
    dataNCa['tempMin'] = dataNCa['tempMin'].apply(lambda x: '{0:0>2}'.format(x))
    dataNCa.dtypes
    #dataNCa[i] = dataNCa[i].map(lambda x: str(x)[0:4])
    dataNCa['temptime'] = dataNCa[['tempHr', 'tempMin']].apply(lambda x: ':'.join(x), axis=1)
    dataNCa[i] = dataNCa[['FL_DATE', 'temptime']].apply(lambda x: ' '.join(x), axis=1)
    dataNCa[i] = dataNCa[i].apply(lambda x: datetime.datetime.strptime(str(x), "%m/%d/%Y %H:%M")) 

#create a list for the columns which need conversion for cancelled data
dateListForCancelled = ['CRS_DEP_TIME','CRS_ARR_TIME']

#Below is the code for converting the columns for cancelled data to make them suitable for converting
#the columns to UTC format
#Take a close look on dtypes of data at each step
#Temp time is made to Convert FL_date to datetime format
#Make sure that no na values is present in data
for i in dateListForCancelled:
    print(i)
    print('total na values',len(dataCa) - dataCa[i].count())
    dataCa[i] = dataCa[i].apply(lambda x: int(x))
    dataCa = dataCa.ix[dataCa[i] > 0]
    dataCa.dtypes
    dataCa['tempHr'] = dataCa[i].apply(lambda x: int(x/100))
    dataCa = dataCa.replace({'tempHr': {24:0 }})
    dataCa['tempHr'] = dataCa['tempHr'].apply(lambda x: str(x))
    dataCa['tempMin'] = dataCa[i].apply(lambda x: int(x%100))
    dataCa['tempMin'] = dataCa['tempMin'].apply(lambda x: '{0:0>2}'.format(x))
    dataCa.dtypes
    #datNCa[i] = dataNCa[i].map(lambda x: str(x)[0:4])
    dataCa['temptime'] = dataCa[['tempHr', 'tempMin']].apply(lambda x: ':'.join(x), axis=1)
    dataCa[i] = dataCa[['FL_DATE', 'temptime']].apply(lambda x: ' '.join(x), axis=1)
    dataCa[i] = dataCa[i].apply(lambda x: datetime.datetime.strptime(str(x), "%m/%d/%Y %H:%M")) 

#to check the data is in correct form or not uncomment the code below to check the excel file
#dataNCa = dataNCa.append(dataCa)
#dataNCa.to_csv('2017DT.csv')

#below is the code for non-cancelled data
#separate the columns according to their origin/destination nature
#Import the time difference excel to check the minute difference between local time and UTC for each airport
#Make a dictionary with key as airport and value as difference in minutes
originEntities = ['CRS_DEP_TIME','DEP_TIME','WHEELS_OFF']
destEntities = ['CRS_ARR_TIME','ARR_TIME']
timediff = pd.read_excel('UTC.xlsx')
mydic = timediff.set_index('Airport').to_dict()['UTC-Diff']

#In the non cancelled data, make a list for unique origin and destination present
O = dataNCa.ORIGIN.unique()
D = dataNCa.DEST.unique()
k=0

#Below is the code for converting the origin only data to UTC format
#timedelta function is used for calculating time after adding/subtracting the time differece
for i in O:
    dataNCa1 = dataNCa[dataNCa['ORIGIN']==i]
    for j in originEntities:
            dataNCa1[j] = dataNCa1[j].apply(lambda x: x-datetime.timedelta(minutes = mydic[i]))
    if k==0:
        dataNCa2 = dataNCa1
        k=k+1
    else:
        dataNCa2 = dataNCa2.append(dataNCa1)
k=0

#Below is the code for converting the destination only data to UTC format
#timedelta function is used for calculating time after adding/subtracting the time differece
for i in D:
    dataNCa3 = dataNCa2[dataNCa2['DEST']==i]
    for j in destEntities:
            dataNCa3[j] = dataNCa3[j].apply(lambda x: x-datetime.timedelta(minutes = mydic[i]))
    if k==0:
        dataNCa4 = dataNCa3
        k=k+1
    else:
        dataNCa4 = dataNCa4.append(dataNCa3)
        k=k+1

#Convert the data to desired format
#As we require only Hour, we will extract hour from Dep and Arr hour
dataNCa4['DEP_HOUR'] = dataNCa4['DEP_TIME'].apply(lambda x: x.strftime('%H'))
dataNCa4['ARR_HOUR'] = dataNCa4['ARR_TIME'].apply(lambda x: x.strftime('%H'))

#below is the code for non-cancelled data
#separate the columns according to their origin/destination nature
originEntities = ['CRS_DEP_TIME']
destEntities = ['CRS_ARR_TIME']

#In the non cancelled data, make a list for unique origin and destination present
O = dataCa.ORIGIN.unique()
D = dataCa.DEST.unique()
k=0

#Below is the code for converting the origin only data to UTC format
#timedelta function is used for calculating time after adding/subtracting the time differece
for i in O:
    dataCa1 = dataCa[dataCa['ORIGIN']==i]
    for j in originEntities:
            dataCa1[j] = dataCa1[j].apply(lambda x: x-datetime.timedelta(minutes = mydic[i]))
            
    if k==0:
        dataCa2 = dataCa1
        k=k+1
    else:
        dataCa2 = dataCa2.append(dataCa1)
k=0
#Below is the code for converting the destination only data to UTC format
#timedelta function is used for calculating time after adding/subtracting the time differece
for i in D:
    dataCa3 = dataCa2[dataCa2['DEST']==i]
    for j in destEntities:
            dataCa3[j] = dataCa3[j].apply(lambda x: x-datetime.timedelta(minutes = mydic[i]))
    if k==0:
        dataCa4 = dataCa3
        k=k+1
    else:
        dataCa4 = dataCa4.append(dataCa3)
        k=k+1

#Convert the data to desired format
#As we require only Hour, we will extract hour from Dep and Arr hour
dataCa4['DEP_HOUR'] = dataNCa4['CRS_DEP_TIME'].apply(lambda x: x.strftime('%H'))
dataCa4['ARR_HOUR'] = dataNCa4['CRS_ARR_TIME'].apply(lambda x: x.strftime('%H'))

#Finally append the cancelled and non cancelled data
dataUTC = dataNCa4.append(dataCa4)        

#Save dataframe to csv
dataUTC.to_csv('2017UTC.csv')
