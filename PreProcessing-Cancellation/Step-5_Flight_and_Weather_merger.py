# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:22:36 2019

@author: bsoni
"""

import datetime
import pandas as pd
import numpy as np
from datetime import datetime as dt

#importing the cleaned weather data and flights data that was converted to UTC
flightdata = pd.read_csv('2018UTC.csv')
finaldf = pd.read_csv('2018weathercleaned.csv')
list(flightdata)

#taking only useful data
flightdata = flightdata[['DAY_OF_WEEK','OP_UNIQUE_CARRIER','OP_CARRIER_FL_NUM',
 'ORIGIN', 'DEST', 'CRS_DEP_TIME','CANCELLED','CANCELLATION_CODE','DISTANCE','traffic']]

#Convert CRS_DEP_TIME to datetime format for proper visualization
flightdata['CRS_DEP_TIME'] = flightdata['CRS_DEP_TIME'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S")) 

#Extract hour and date from CRS_DEP_TIME
#rename columns according to need and convert FL_DATE to such a format so that it could be converted to int easily
#date in int would help us to merge data easily
flightdata['Hour'] = flightdata['CRS_DEP_TIME'].apply(lambda x: x.strftime('%H'))
flightdata['FLY_DATE'] = flightdata['CRS_DEP_TIME'].apply(lambda x: x.strftime('%Y%m%d'))
flightdata['Hour'] = flightdata.Hour.astype(int)
flightdata.rename(columns={'FLY_DATE':'FLY_date',
                                   }, inplace=True)

#rename the airport column as origin and each weather as O_Weather_feature
#Than merge the UTC data and weather data on origin, date and hour
#This will give us hourly weather values for origin
finaldf.rename(columns={'Airport':'ORIGIN','SurfaceTemperatureFarenheit':'O_SurfaceTemperatureFarenheit',
                                   'CloudCoveragePercent':'O_CloudCoveragePercent','WindSpeedMph':'O_WindSpeedMph',
                                    'PrecipitationPreviousHourInches':'O_PrecipitationPreviousHourInches','SnowfallInches':'O_SnowfallInches',
                                   }, inplace=True)

data = pd.merge(flightdata,finaldf,on=['FLY_date','Hour','ORIGIN'])

#rename the origin columns as destination and each weather as O_Weather_feature
#Than merge the UTC data and weather data on destination, date and hour
#This will give us hourly weather values for both origin and destination
finaldf.rename(columns={'ORIGIN':'DEST','O_SurfaceTemperatureFarenheit':'D_SurfaceTemperatureFarenheit',
                                   'O_CloudCoveragePercent':'D_CloudCoveragePercent','O_WindSpeedMph':'D_WindSpeedMph',
                                    'O_PrecipitationPreviousHourInches':'D_PrecipitationPreviousHourInches','O_SnowfallInches':'D_SnowfallInches',
                                   }, inplace=True)
mydf = pd.merge(data,finaldf,on=['FLY_date','Hour','DEST'])


#retain the old columns names again
mydf.rename(columns={'OP_UNIQUE_CARRIER':'UNIQUE_CARRIER','OP_CARRIER_FL_NUM':'FL_NUM','FLY_date':'FL_DATE',
                     'O_SurfaceTemperatureFarenheit':'O_SurfaceTemperatureFahrenheit',
                     'D_SurfaceTemperatureFarenheit':'D_SurfaceTemperatureFahrenheit'
                                   }, inplace=True)

#take only required data for prediction or visualization purpose
mydf=mydf[['DAY_OF_WEEK','FL_DATE',	'UNIQUE_CARRIER',	'FL_NUM',	'ORIGIN',	'DEST','CRS_DEP_TIME',	
           'Hour'	,'CANCELLED',	'CANCELLATION_CODE',	'DISTANCE',	'traffic',	'O_SurfaceTemperatureFahrenheit',	'O_CloudCoveragePercent',	
           'O_WindSpeedMph',	'O_PrecipitationPreviousHourInches',	'O_SnowfallInches',	'D_SurfaceTemperatureFahrenheit',	'D_CloudCoveragePercent','D_WindSpeedMph',	
           'D_PrecipitationPreviousHourInches',	'D_SnowfallInches']]

#remove the duplicate rows from the database
#also convert unwanted values of column to their numeric/allocated value
mydf = mydf.drop_duplicates()
mydf = mydf.replace({'O_CloudCoveragePercent': 'VV ', 'D_CloudCoveragePercent': 'VV '}, 100)

#finally!! save data to csv
mydf.to_csv('weathertrafficandflights-2018.csv')

















