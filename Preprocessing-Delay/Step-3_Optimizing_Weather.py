import datetime
import datetime
import pandas as pd
import numpy as np
from datetime import datetime as dt

#import the semi-cleaned weather data
df_bos = pd.read_csv('BOS_weather-2018.csv') 
df_den = pd.read_csv('DEN_weather-2018.csv') 
df_dfw = pd.read_csv('DFW_weather-2018.csv') 
df_ewr = pd.read_csv('EWR_weather-2018.csv') 
df_iah = pd.read_csv('IAH_weather-2018.csv') 
df_jfk_lga  = pd.read_csv('JFK-LGA.csv')
df_ord = pd.read_csv('ORD_weather-2018.csv') 
df_phl = pd.read_csv('PHL_weather-2018.csv') 
df_sfo = pd.read_csv('SFO_weather-2018.csv') 

#join each dataframe
#make sure that column names must be same for each dataframe
aggr_dataset = [df_bos,df_den,df_dfw,df_ewr,df_iah,df_jfk_lga,df_ord,df_phl,df_sfo]
airport_weather_df = pd.concat(aggr_dataset)

#find unique cities in the data
#this will help us to clean data later
M = airport_weather_df.CITY.unique()

#constantly checking the dtypes is a good practice
#it also helps us to find whether data is in proper format or not
airport_weather_df.dtypes


airport_weather_df['FL_DATE'] = airport_weather_df['FL_DATE'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S")) 
airport_weather_df['FLY_min'] = airport_weather_df['FL_DATE'].apply(lambda x: x.strftime('%M'))

#make list for unique date and hours
#it will help us to clean data later
D = airport_weather_df.FLY_date.unique()
H = airport_weather_df.FLY_hour.unique()
k=0

#below code takes the only first value of each hour at each date
#it ignores the rest of the values for a particular date and for a particular hour
#tis is necessary because when we merge the data to flights data and if code finds multiple values against a single value, it will throw an error
for m in M:
    airport_weather_df1 = airport_weather_df[airport_weather_df['CITY']==m]
    D = airport_weather_df1.FLY_date.unique()
    for i in D:
        tempdf1 = airport_weather_df1[airport_weather_df1['FLY_date']==i]
        H = tempdf1.FLY_hour.unique()
        for j in H:
            tempdf2 = tempdf1[tempdf1['FLY_hour']==j]
            tempdf2 = tempdf2.reset_index()
            if k==0:
                finaldf = tempdf2.loc[[0]]
                k=k+1
            else:
                finaldf = finaldf.append(tempdf2.loc[[0]])
                k=k+1
           
#rename the columns as per your standards
finaldf.rename(columns={'CITY':'Airport','WindSpeedKnots':'WindSpeedMph','FLY_hour':'Hour',
                                   }, inplace=True)
list(finaldf)

#convert Date to YMD format without any hyphen or slash
#it will help in merging data later
finaldf['FLY_date'] = finaldf['FLY_date'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d")) 
finaldf['FLY_date'] = finaldf['FLY_date'].dt.strftime('%Y%m%d')

#take only needed columns
finaldf = finaldf[['Airport','SurfaceTemperatureFarenheit','PrecipitationPreviousHourInches','CloudCoveragePercent','SnowfallInches','WindSpeedMph','FLY_date','Hour']]
#save data to csv
finaldf.to_csv('2017weathercleaned.csv')