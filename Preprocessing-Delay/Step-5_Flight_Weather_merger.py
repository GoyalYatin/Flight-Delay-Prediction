# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:00:48 2019

@author: bsoni
"""

import datetime
import pandas as pd
import numpy as np
from datetime import datetime as dt

#importing the cleaned weather data and flights data that was converted to UTC
#rename columns according to need and convert FL_DATE to such a format so that it could be converted to int easily
#date in int would help us to merge data easily

finaldf = pd.read_csv('2016weathercleaned.csv')
flightdata = pd.read_csv('2016UTC.csv')

finaldf.rename(columns={'FLY_date':'FL_DATE','SurfaceTemperatureFahrenheit':'SurfaceTemperatureFarenheit'},inplace=True)
flightdata['FL_DATE'] = flightdata['FL_DATE'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d")) 
flightdata['FL_DATE'] = flightdata['FL_DATE'].dt.strftime('%Y%m%d')
flightdata['FL_DATE'] = flightdata['FL_DATE'].astype(int)

#we only need non cancelled data for delay
flightdata = flightdata[flightdata['CANCELLED']==0]
flightdata = flightdata.drop(columns=['Hour'],axis=1)

#rename the airport column as origin and each weather as O_Weather_feature
#Than merge the UTC data and weather data on origin, date and hour
#This will give us hourly weather values for origin
finaldf.rename(columns={'Airport':'ORIGIN','SurfaceTemperatureFarenheit':'O_SurfaceTemperatureFarenheit',
                                   'CloudCoveragePercent':'O_CloudCoveragePercent','WindSpeedMph':'O_WindSpeedMph',
                                    'PrecipitationPreviousHourInches':'O_PrecipitationPreviousHourInches','SnowfallInches':'O_SnowfallInches',
                                   }, inplace=True)

flightdata.rename(columns={'DEP_HOUR':'Hour'}, inplace = True)
data = pd.merge(flightdata,finaldf,on=['FL_DATE','Hour','ORIGIN'])

#rename the origin columns as destination and each weather as O_Weather_feature
#Than merge the UTC data and weather data on destination, date and hour
#This will give us hourly weather values for both origin and destination
finaldf.rename(columns={'ORIGIN':'DEST','O_SurfaceTemperatureFarenheit':'D_SurfaceTemperatureFarenheit',
                                   'O_CloudCoveragePercent':'D_CloudCoveragePercent','O_WindSpeedMph':'D_WindSpeedMph',
                                    'O_PrecipitationPreviousHourInches':'D_PrecipitationPreviousHourInches','O_SnowfallInches':'D_SnowfallInches',
                                   }, inplace=True)
data.rename(columns={'Hour':'DEP_HOUR'}, inplace = True)
data.rename(columns={'ARR_HOUR':'Hour'}, inplace = True)
mydf = pd.merge(data,finaldf,on=['FL_DATE','Hour','DEST'])
mydf.rename(columns={'OP_UNIQUE_CARRIER':'UNIQUE_CARRIER','OP_CARRIER_FL_NUM':'FL_NUM','FLY_date':'FL_DATE',
                     'O_SurfaceTemperatureFarenheit':'O_SurfaceTemperatureFahrenheit',
                     'D_SurfaceTemperatureFarenheit':'D_SurfaceTemperatureFahrenheit'
                                   }, inplace=True)

#retain the old columns names again
mydf.rename(columns={'Hour':'ARR_HOUR'}, inplace = True)

#take only required data for prediction or visualization purpose
mydf=mydf[['YEAR','MONTH','DAY_OF_MONTH','DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','FL_NUM','ORIGIN','DEST','ARR_DELAY',
 'CRS_ELAPSED_TIME','DISTANCE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY',
 'DEP_HOUR','ARR_HOUR','traffic','O_SurfaceTemperatureFahrenheit','O_PrecipitationPreviousHourInches','O_CloudCoveragePercent',
 'O_SnowfallInches','O_WindSpeedMph','D_SurfaceTemperatureFahrenheit','D_PrecipitationPreviousHourInches','D_CloudCoveragePercent',
 'D_SnowfallInches','D_WindSpeedMph']]

#remove the duplicate rows from the database
#also convert unwanted values of column to their numeric/allocated value
mydf = mydf.drop_duplicates()
mydf = mydf.replace({'O_CloudCoveragePercent': 'VV ', 'D_CloudCoveragePercent': 'VV '}, 100)

#finally!! save data to csv
mydf.to_csv('weathertrafficandflights-2016.csv')