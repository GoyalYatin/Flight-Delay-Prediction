# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:26:09 2019

@author: bsoni
"""

import datetime
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime as dt

#import the downloaded weather data
#take only required data that will contribute to delay
#replace missing value codes with nan 
data=pd.read_excel('SFO.xlsx')
data=data[['station','valid','tmpf','p01i','skyc1','ice_accretion_1hr','sknt']]
data = data.replace('M', np.nan)
data = data.replace('T', np.nan)

#in some cases sknt will have M codes. In that case uncomment the code below
#data = data.replace({'sknt': {'M': 0}})

#convert each value to numeric
#if any error is coming, there is still some data present as string. COnvert that accordingly.
data[["sknt"]] = data[["sknt"]].apply(pd.to_numeric)
data[["tmpf"]] = data[["tmpf"]].apply(pd.to_numeric)
data[["p01i"]] = data[["p01i"]].apply(pd.to_numeric)

#Convert knots to kmph
data['sknt'] = data['sknt'].apply(lambda x: x*1.15078)

#rename the data according to standard you're following
#Following a standard will help in merging later on.
data = data.rename(columns={'station': 'CITY', 'valid': 'FL_DATE', 'tmpf': 'SurfaceTemperatureFarenheit', 'p01i': 'PrecipitationPreviousHourInches',
                            'skyc1': 'CloudCoveragePercent', 'ice_accretion_1hr': 'SnowfallInches', 'sknt': 'WindSpeedKnots'})

#replace the cloud coverage values from their standard numeric values
#fill na values with 0
data = data.replace({'CloudCoveragePercent': {'OVC': 80, 'BKN': 60, 'SCT':40, 'FEW':20,'CLR':0}})
data['CloudCoveragePercent'].fillna(0,inplace=True)

#extract date and hour values from FLY_date column
#If any error is there, the data is not in proper date-time format.
data['FLY_date'] = data['FL_DATE'].apply(lambda x: x.date())
data['FLY_hour'] = data['FL_DATE'].apply(lambda x: x.strftime('%H'))

#Make a list of unique values of date and hour
#this will help in averaging data successfully
k = data.FLY_date.unique()
m = data.FLY_hour.unique()
count = 0

#below is the code for replacing na with averge of the hourwise values of weather for each date
#the if else condition is just for initiation of the loop
for i in k:
    for j in m:
        if(count==0):
            data1 = data[data['FLY_date']==i]
            data1 = data1[data1['FLY_hour']==j]
            data1['SurfaceTemperatureFarenheit'].fillna(data1['SurfaceTemperatureFarenheit'].mean(), inplace=True)
            data1['PrecipitationPreviousHourInches'].fillna(data1['PrecipitationPreviousHourInches'].mean(), inplace=True)
            data1['WindSpeedKnots'].fillna(data1['WindSpeedKnots'].mean(), inplace=True)
            count=count+1
        else:
            print(count)
            data2 = data[data['FLY_date']==i]
            data2 = data2[data2['FLY_hour']==j]
            data2['SurfaceTemperatureFarenheit'].fillna(data2['SurfaceTemperatureFarenheit'].mean(), inplace=True)
            data2['PrecipitationPreviousHourInches'].fillna(data2['PrecipitationPreviousHourInches'].mean(), inplace=True)
            data2['WindSpeedKnots'].fillna(data2['WindSpeedKnots'].mean(), inplace=True)
            data1 = data1.append(data2)
            count=count+1

#fill na values with 0, we dont want any data to be lost
data1.fillna(0,inplace=True)

#finally convert to csv
data1.to_csv('SFO_weather-2018.csv')