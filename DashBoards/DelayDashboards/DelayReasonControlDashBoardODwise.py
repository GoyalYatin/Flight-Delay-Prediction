# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:19:16 2019

@author: bsoni
"""

import pymongo
import pandas as pd
import pickle
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from datetime import datetime as dt
from dash.dependencies import Input, Output
import datetime
import plotly.plotly as py

#import the flights and weather merged data for the 3 years
#we have to train on 2016-17 while test on 2018
traindata1 = pd.read_csv('DelayData2016.csv')
traindata2 = pd.read_csv('DelayData2017.csv')
testdata = pd.read_csv('DelayData2018.csv')

#import birdstrike data as well
birdStrike = pd.read_csv('BirdStrikes.csv') 

#take only those columns which are useful for predictions
#sometimes unwanted columns are added while merging/separating the data
#to counter that we should mention columns we are going to take
traindata1 = traindata1[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]

#take only those columns which are useful for predictions
#sometimes unwanted columns are added while merging/separating the data
#to counter that we should mention columns we are going to take
traindata2 = traindata2[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]

#take only those columns which are useful for predictions
#sometimes unwanted columns are added while merging/separating the data
#to counter that we should mention columns we are going to take
testdata = testdata[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]

#drop na values from testdata
testdata = testdata.dropna()

#aggregate all the data
aggr_dataset = [traindata1 , traindata2 , testdata]
data = pd.concat(aggr_dataset)

#copy the data to df for further use
df = data

#take only required columns from data
df = df[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','CRS_ELAPSED_TIME',
'DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches','ARR_DELAY','FL_DATE']]

#merge the data of bir strike probability on origin
df = pd.merge(df,birdStrike,on=['ORIGIN'])

#Take only useful columns from data
df = df[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','CRS_ELAPSED_TIME',
'DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches','Bird_Strike',
'ARR_DELAY','FL_DATE']]

#drop duplicates frm data
df = df.drop_duplicates()

#convert the negative delays to zero
df['ARR_DELAY'] = df['ARR_DELAY'].clip_lower(0)

#Make a list of categorical columns for one-hot-encoding
cols_to_transform = ['UNIQUE_CARRIER','ORIGIN','DEST']

#take only that carriers which are available in 2016-17 data
carrierlist = traindata1.UNIQUE_CARRIER.unique()
df = df[df.UNIQUE_CARRIER.isin(carrierlist)]

#perform one-hot-encoding
df = pd.get_dummies(df, columns = cols_to_transform )
#drop na values
df = df.dropna()

#separate the data according to year
#since we need FL_DATE column from test data for visualization purpose
#we are not going to remove it
df1 = df[df['YEAR']==2016]
df1 = df1.drop(['YEAR','FL_DATE'], axis=1)
df2 = df[df['YEAR']==2017]
df2 = df2.drop(['YEAR','FL_DATE'], axis=1)
df3 = df[df['YEAR']==2018]
df3 = df3.drop(columns=['YEAR'])

#aggregate the data to form train data
aggr_dataset = [df1,df2]
traindata = pd.concat(aggr_dataset)

#remove na values from traindata
traindata = traindata.dropna()

#the code below remove the outliers from train data
z = np.abs(stats.zscore(traindata))
z= pd.DataFrame(z)
z=z.fillna(0)
z=z.values
traindata = traindata[(z < 3.5).all(axis=1)]

#the code below it to balance the data
#we found out that the value greater than 80 were very less in number
#so we created two classes for delay greater tahn  and less than 80
#and than we applied smote with class as delay_class
#this helps the model to properly evaluate the data
traindata['DELAY_CLASS'] = 3
traindata['DELAY_CLASS'] = np.where((traindata.ARR_DELAY>=0)&(traindata.ARR_DELAY <80)  ,0,3)
data1 = traindata[traindata['DELAY_CLASS']==0]
traindata['DELAY_CLASS'] = np.where((traindata.ARR_DELAY>=80)&(traindata.ARR_DELAY >0)  ,1,3)
data2 = traindata[traindata['DELAY_CLASS']==1]
aggr_dataset = [data1,data2]
dataModel = pd.concat(aggr_dataset)
train_labels = dataModel['DELAY_CLASS']
traindata = dataModel.drop(columns=['DELAY_CLASS'])
sm = SMOTE(random_state=1)
X_train_res, y_train_res = sm.fit_sample(traindata, train_labels)

#convert the smoted data to dataframe
X_train_res = pd.DataFrame(X_train_res)
#retain the column names again to data
X_train_res = X_train_res.rename(columns = {0:'MONTH',1:'DAY_OF_MONTH',2:'FL_NUM',3:'DAY_OF_WEEK',4:'Dep_Hour',
5:'Arr_Hour', 6:'CRS_ELAPSED_TIME', 7:'DISTANCE',
8:'traffic',9:'O_SurfaceTemperatureFahrenheit',10:'O_CloudCoveragePercent',
11:'O_WindSpeedMph',12:'O_PrecipitationPreviousHourInches',13:'O_SnowfallInches',
14:'D_SurfaceTemperatureFahrenheit',15:'D_CloudCoveragePercent',16:'D_WindSpeedMph',
17:'D_PrecipitationPreviousHourInches',18:'D_SnowfallInches',19:'Bird_Strike',20:'ARR_DELAY',
21:'UNIQUE_CARRIER_AA',
22:'UNIQUE_CARRIER_B6',23:'UNIQUE_CARRIER_DL',24:'UNIQUE_CARRIER_EV',
25:'UNIQUE_CARRIER_F9',26:'UNIQUE_CARRIER_NK',
27:'UNIQUE_CARRIER_OO',28:'UNIQUE_CARRIER_UA',
29:'UNIQUE_CARRIER_VX',30:'UNIQUE_CARRIER_WN',31:'ORIGIN_BOS',32:'ORIGIN_DEN',
33:'ORIGIN_DFW',34:'ORIGIN_EWR',35:'ORIGIN_IAH',36:'ORIGIN_JFK',
37:'ORIGIN_LGA',38:'ORIGIN_ORD',39:'ORIGIN_PHL',40:'ORIGIN_SFO',41:'DEST_BOS',
42:'DEST_DEN',43:'DEST_DFW',44:'DEST_EWR',45:'DEST_IAH',46:'DEST_JFK',
47:'DEST_LGA',48:'DEST_ORD',49:'DEST_PHL',50:'DEST_SFO'})

#separate the arr_delay(target value) from data
y_train_res = X_train_res['ARR_DELAY']
#remove the target value from features
X_train_res = X_train_res.drop(columns=['ARR_DELAY'],axis=1)

#df3 is 2018 test data
# remove na values
df3 = df3.dropna()
# create feature and test split for test purposes
X_test = df3
X_test_for_delay = X_test.copy()
y_test = X_test['ARR_DELAY']
X_test = X_test.drop(columns=['ARR_DELAY'],axis=1)

#convert training features to series
y_train_res = pd.Series(y_train_res)



#copy the data to datadash so that we can use it further
#!Always use copy function to  copy the data
#!Simply using datadash=data can be harmful, as if you change something in datadash it will do same changes in data 
datadash1 = X_test.copy()

#make the datadash1 suitable for displaying purposes
#this will help us in visualization later
datadash1['FL_DATE'] = datadash1['FL_DATE'].astype(str) 
datadash1['DEP_HOUR'] = datadash1['DEP_HOUR'].astype(float) 
datadash1['DEP_HOUR'] = datadash1['DEP_HOUR'].astype(int) 
datadash1['DEP_HOUR'] = datadash1['DEP_HOUR'].astype(str) 
datadash1['FL_DATE'] = datadash1[['FL_DATE', 'DEP_HOUR']].apply(lambda x: ' '.join(x), axis=1)
datadash1['FL_DATE'] = datadash1['FL_DATE'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d %H")) 

#make a list of columns that are needed to be scaled
coll = ['CRS_ELAPSED_TIME','DISTANCE','O_SurfaceTemperatureFahrenheit',
 'O_CloudCoveragePercent','O_WindSpeedMph','D_SurfaceTemperatureFahrenheit',
 'D_CloudCoveragePercent','D_WindSpeedMph']

#perform scaling on columns
scaler = StandardScaler()
scaler.fit(X_train_res[coll])
X_train_res[coll] = scaler.transform(X_train_res[coll])

#perform polynomial tranformation on the data
#we have found that it decreases the error by significant amount
#this "poly" variable will help us to tranform the test data at time of prediction
poly = PolynomialFeatures(2)
poly.fit(X_train_res)
X_train_res = poly.transform(X_train_res)

#!!!this is the prediction phase area
#!!!you should write predictions code here
################################################################################################
#                                                                                              #
#                                                                                              #
#                                                                                              #
#                Prediction phase area                                                         #
#                                                                                              #
#                                                                                              #
#                                                                                              #
#                                                                                              #
################################################################################################


#Now we are going for delay class prediction
#below code is for prediction of delay classes
#again perform the similar steps as delay predictions
#import data from csv
delaytraindata1 = pd.read_csv('DelayData2016.csv')
delaytraindata2 = pd.read_csv('DelayData2017.csv')
delaytestdata = pd.read_csv('DelayData2018.csv')
#also import bird strike data
delaybirdStrike = pd.read_csv('BirdStrikes.csv') 
#take only those columns which are useful for predictions
#sometimes unwanted columns are added while merging/separating the data
#to counter that we should mention columns we are going to take
delaytraindata1 = delaytraindata1[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]
#take only those columns which are useful for predictions
#sometimes unwanted columns are added while merging/separating the data
#to counter that we should mention columns we are going to take
delaytraindata2 = delaytraindata2[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]

#take only those columns which are useful for predictions
#sometimes unwanted columns are added while merging/separating the data
#to counter that we should mention columns we are going to take
delaytestdata = delaytestdata[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]

#The data will be havng separate columns for each class
#convert all the na values of reason  as 0 in train data
delaytraindata1['CARRIER_DELAY'] = delaytraindata1['CARRIER_DELAY'].fillna(0)
delaytraindata1['WEATHER_DELAY'] = delaytraindata1['WEATHER_DELAY'].fillna(0)
delaytraindata1['NAS_DELAY'] = delaytraindata1['NAS_DELAY'].fillna(0)
delaytraindata1['SECURITY_DELAY'] = delaytraindata1['SECURITY_DELAY'].fillna(0)
delaytraindata1['LATE_AIRCRAFT_DELAY'] = delaytraindata1['LATE_AIRCRAFT_DELAY'].fillna(0)

#The data will be havng separate columns for each class
#convert all the na values of reason  as 0 in train data
delaytraindata2['CARRIER_DELAY'] = delaytraindata2['CARRIER_DELAY'].fillna(0)
delaytraindata2['WEATHER_DELAY'] = delaytraindata2['WEATHER_DELAY'].fillna(0)
delaytraindata2['NAS_DELAY'] = delaytraindata2['NAS_DELAY'].fillna(0)
delaytraindata2['SECURITY_DELAY'] = delaytraindata2['SECURITY_DELAY'].fillna(0)
delaytraindata2['LATE_AIRCRAFT_DELAY'] = delaytraindata2['LATE_AIRCRAFT_DELAY'].fillna(0)

#The data will be havng separate columns for each class
#convert all the na values of reason  as 0 in test data
delaytestdata['CARRIER_DELAY'] = delaytestdata['CARRIER_DELAY'].fillna(0)
delaytestdata['WEATHER_DELAY'] = delaytestdata['WEATHER_DELAY'].fillna(0)
delaytestdata['NAS_DELAY'] = delaytestdata['NAS_DELAY'].fillna(0)
delaytestdata['SECURITY_DELAY'] = delaytestdata['SECURITY_DELAY'].fillna(0)
delaytestdata['LATE_AIRCRAFT_DELAY'] = delaytestdata['LATE_AIRCRAFT_DELAY'].fillna(0)

#aggregate all the data
aggr_dataset = [delaytraindata1 , delaytraindata2 , delaytestdata]
delaydata = pd.concat(aggr_dataset)

#merge the bird strike data on origin
delaydata = pd.merge(delaydata,delaybirdStrike,on=['ORIGIN'])
delaydata = delaydata[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches','Bird_Strike']]

#take only those carriers which are presnet in 2016-17 data
delaycarrierlist = delaytraindata1.UNIQUE_CARRIER.unique()
delaydata = delaydata[delaydata.UNIQUE_CARRIER.isin(delaycarrierlist)]
#drop the na values
delaydata = delaydata.dropna()

#convert the negative values of delay as 0
delaydata['ARR_DELAY'] = delaydata['ARR_DELAY'].clip_lower(0)

#below code separates the data of flights getting delayed and flights arriving on time
delaydata1 = delaydata[delaydata['ARR_DELAY']==0]
delaydata2 = delaydata[delaydata['ARR_DELAY']>0]
delaydata1['DELAY_CLASS'] = 'NO_DELAY'
#the below code  returns the major reason of delay against all the delayed data rows
delaydata2['DELAY_CLASS'] = delaydata2[['CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']].idxmax(axis=1)

#aggregate the delayed and on-time data again
aggr_dataset = [delaydata1 ,delaydata2]
delaydata = pd.concat(aggr_dataset)

#Select only that columns which are needed for classification
delaydata_delay_class = delaydata[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM','DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR',
 'ARR_HOUR','CRS_ELAPSED_TIME','DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
 'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches',
 'Bird_Strike','DELAY_CLASS']]

#take only that carriers which are available in 2016-17 data
cols_to_transform = ['UNIQUE_CARRIER','ORIGIN','DEST']
#perform one-hot-encoding
delaydf = pd.get_dummies(delaydata_delay_class, columns = cols_to_transform )

#separate the data according to year
delaydf1 = delaydf[delaydf['YEAR']==2016]
delaydf1 = delaydf1.drop(['YEAR','FL_DATE'], axis=1)
delaydf2 = delaydf[delaydf['YEAR']==2017]
delaydf2 = delaydf2.drop(['YEAR','FL_DATE'], axis=1)
delaydf3 = delaydf[delaydf['YEAR']==2018]
delaydf3 = delaydf3.drop(columns=['YEAR','FL_DATE'])

#aggregate the training data
aggr_dataset = [delaydf1,delaydf2]
delaytraindata = pd.concat(aggr_dataset)

#only training on delayed data
#so selecting the delayed data from dataframe
delay_category = ['CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']
delaytraindata = delaytraindata[delaytraindata.DELAY_CLASS.isin(delay_category)]

#drop na values
delaytraindata = delaytraindata.dropna()
#delay class will be our train labe; 
delaytrain_labels = delaytraindata['DELAY_CLASS']
#drop train label from train features
delaytraindata = delaytraindata.drop(columns=['DELAY_CLASS'])

#select only required data for training
X_test_for_delay = X_test_for_delay[['MONTH','DAY_OF_MONTH','FL_NUM','DAY_OF_WEEK','DEP_HOUR','ARR_HOUR','CRS_ELAPSED_TIME','DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent',
'O_WindSpeedMph','O_PrecipitationPreviousHourInches','O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches',
'D_SnowfallInches','Bird_Strike','UNIQUE_CARRIER_AA','UNIQUE_CARRIER_B6','UNIQUE_CARRIER_DL','UNIQUE_CARRIER_EV','UNIQUE_CARRIER_F9','UNIQUE_CARRIER_NK',
'UNIQUE_CARRIER_OO','UNIQUE_CARRIER_UA','UNIQUE_CARRIER_VX','UNIQUE_CARRIER_WN','ORIGIN_BOS','ORIGIN_DEN','ORIGIN_DFW','ORIGIN_EWR','ORIGIN_IAH','ORIGIN_JFK',
'ORIGIN_LGA','ORIGIN_ORD','ORIGIN_PHL','ORIGIN_SFO','DEST_BOS','DEST_DEN','DEST_DFW','DEST_EWR','DEST_IAH','DEST_JFK','DEST_LGA','DEST_ORD',
'DEST_PHL','DEST_SFO']]

#perform smote to counter the unbalanced data
sm = SMOTE(random_state=1)
delayX_train_res, delayy_train_res = sm.fit_sample(delaytraindata, delaytrain_labels)
delayX_train_res = pd.DataFrame(delayX_train_res)

#reatain the column names after smoting
delayX_train_res = delayX_train_res.rename(columns = {0:'MONTH',1:'DAY_OF_MONTH',2:'FL_NUM',3:'DAY_OF_WEEK',4:'Dep_Hour',
5:'Arr_Hour', 6:'CRS_ELAPSED_TIME', 7:'DISTANCE',
8:'traffic',9:'O_SurfaceTemperatureFahrenheit',10:'O_CloudCoveragePercent',
11:'O_WindSpeedMph',12:'O_PrecipitationPreviousHourInches',13:'O_SnowfallInches',
14:'D_SurfaceTemperatureFahrenheit',15:'D_CloudCoveragePercent',16:'D_WindSpeedMph',
17:'D_PrecipitationPreviousHourInches',18:'D_SnowfallInches',19:'Bird_Strike',
20:'UNIQUE_CARRIER_AA',
21:'UNIQUE_CARRIER_B6',22:'UNIQUE_CARRIER_DL',23:'UNIQUE_CARRIER_EV',
24:'UNIQUE_CARRIER_F9',25:'UNIQUE_CARRIER_NK',
26:'UNIQUE_CARRIER_OO',27:'UNIQUE_CARRIER_UA',
28:'UNIQUE_CARRIER_VX',29:'UNIQUE_CARRIER_WN',30:'ORIGIN_BOS',31:'ORIGIN_DEN',
32:'ORIGIN_DFW',33:'ORIGIN_EWR',34:'ORIGIN_IAH',35:'ORIGIN_JFK',
36:'ORIGIN_LGA',37:'ORIGIN_ORD',38:'ORIGIN_PHL',39:'ORIGIN_SFO',40:'DEST_BOS',
41:'DEST_DEN',42:'DEST_DFW',43:'DEST_EWR',44:'DEST_IAH',45:'DEST_JFK',
46:'DEST_LGA',47:'DEST_ORD',48:'DEST_PHL',49:'DEST_SFO'})

#make a list for columns that need scaling
delaycoll = ['CRS_ELAPSED_TIME','DISTANCE','O_SurfaceTemperatureFahrenheit',
 'O_CloudCoveragePercent','O_WindSpeedMph','D_SurfaceTemperatureFahrenheit',
 'D_CloudCoveragePercent','D_WindSpeedMph']

#perform scaling on the data
#this "delaycoll" will help us in future to scale testdata
delayscaler = StandardScaler()
delayscaler.fit(delayX_train_res[delaycoll])
delayX_train_res[delaycoll] = delayscaler.transform(delayX_train_res[delaycoll])
X_test_for_delay[delaycoll] = delayscaler.transform(X_test_for_delay[delaycoll])
#!!!this is the prediction phase area
#!!!you should write predictions code here
################################################################################################
#                                                                                              #
#                                                                                              #
#                                                                                              #
#                Prediction phase area                                                         #
#                                                                                              #
#                                                                                              #
#                                                                                              #
#                                                                                              #
################################################################################################

#to save the models into db perform refer to readme
#to load the saved models from mongodb use the function below
def load_saved_model_from_db(model_name, client, db, dbconnection):
    json_data = {}
    
    #saving model to mongoDB
    # creating connection
    myclient = pymongo.MongoClient(client)
    
    #creating database in mongodb
    mydb = myclient[db]
    
    #creating collection
    mycon = mydb[dbconnection]
    data = mycon.find({'name': model_name})
    
    
    for i in data:
        json_data = i
    #fetching model from db
    pickled_model = json_data[model_name]
    return pickle.loads(pickled_model)

#you can use the functions to load saved models
#!!!replace delay_1 with  the model name you have saved
#!!!replace models with database name
#!!!replace regression with your collection name
clf  = load_saved_model_from_db(model_name = 'delay_1', client = 'mongodb://localhost:27017/', 
                         db = 'models', dbconnection = 'regression')

clf1  = load_saved_model_from_db(model_name = 'delay_reason', client = 'mongodb://localhost:27017/', 
                         db = 'models', dbconnection = 'regression')

#these are exteernal stylesheets we are using
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#!!!if you have any local css file you have to give __name__ as an argument in dash.dash()
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#strore testdata in datadash for further use
datadash = testdata

#make a unique list of carriers
#this will help us in selections in future
available_indicator_airline = datadash['UNIQUE_CARRIER'].unique()

#below is app layout for our app
#it is designed in python with HTML on top
#make sure that every id you give to each element should not be repeated.
app.layout = html.Div([
                
				html.Div( 
                 [html.H3([html.Span('DELAY DASHBOARD',className="main-heading-primary")]
                 ,className='main-heading'),
				]),
    
    
				dcc.Markdown(''' --- '''),
                
				 html.Div([html.H3('Enter a carrier code:', style={'margin-left': '149px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-03',
						  options=[{'label': i, 'value': i} for i in available_indicator_airline],
                          value='',
                          
						  style={'fontSize': 15, 'width': 300,'margin-left': '75px',},
				)], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-left': '220px'}),
    
    
                html.Div([html.H3('Enter origin city:', style={'margin-left': '149px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-02',
                          
						  style={'fontSize': 15, 'width': 300,'margin-left': '75px'},
                           
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
                
                
                
                html.Div([html.H3('Enter destination city:', style={'margin-left': '149px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-01',
						  style={'fontSize': 15, 'width': 300,'margin-left': '75px'},
                          
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-left': '220px'}),
            
                
                
                
    

				html.Div([html.H3('Enter start / end date:'),
					dcc.DatePickerRange(id='my_date_picker-00',
										min_date_allowed = dt(2018,1,1),
										max_date_allowed = dt(2018,12,12),
										start_date = dt(2018, 1,2),
										end_date = dt(2018, 4,28),
                                         display_format='MMM Do, YY',
                                         
					)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '28%','margin-left': '150px'}), 
                    
				html.Div([
					html.Button(id='submit-button-00',
								n_clicks = 0,
								children = 'Submit',
								style = {'fontSize': 24, 'marginLeft': '30px','margin-left': '75px'}

					)

				], style={'display': 'None'}),
				dcc.Markdown(''' --- '''), 
				
				html.Div([dcc.Graph(id='my_graph-00',
							figure={'layout':go.Layout(title='daily Cancellations will be shown here', height= 600,
                width= 600,
                               
                                         )}
				), ],id='graph',style={'width': '85%', 'float':'left' ,'margin-left': '220px'}),
    
    html.Div([html.H3('Enter a carrier number:', style={'margin-left': '375px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-00',
						  multi=True,
                          value='',
                          
						  style={'fontSize': 15, 'width': 300,'margin-left': '190px'},
				)], style={'float': 'left', 'margin-left':'570px','margin-top':'-350px'}),
    
     html.Div([dcc.RadioItems(
            id='my_radio-00',
    options=[
            {'label': 'None', 'value': 'S'},
        {'label': 'Airlines/Carrier', 'value': 'CARRIER_DELAY'},
        {'label': 'Weather', 'value': 'WEATHER_DELAY'},
        {'label': 'National Air System', 'value': 'NAS_DELAY'},
        {'label': 'Security', 'value': 'SECURITY_DELAY'},
        {'label': 'Late Aircraft', 'value': 'LATE_AIRCRAFT_DELAY'}
    ],
    value='S'
)], style={'float':'left', 'margin-top':'-500px','margin-left':'1000px'}),
    
    html.Div(id = 'textonly-ce1',children=' ', style={
        'textAlign': 'center',
      'display':'inline-block'
    }),
               
], id='particles-js')

#Now we are starting app callbacks
# each callback calls a function
#the function name should be unique within the app.
#the called function takes the input in same order as written in callback
#below callback returns the origin options for selected airline
@app.callback(
                Output('my_ticker_symbol-02', 'options'),
                [Input('my_ticker_symbol-03', 'value')]
)
def set_destination_options(selected_origin):
                new_data = datadash[datadash['UNIQUE_CARRIER'] == selected_origin]
                return [{'label' : i, 'value' : i} for i in new_data['ORIGIN'].unique()]

#This callback is to return options for destination airport for selected airline and origin airport
@app.callback(
                Output('my_ticker_symbol-01', 'options'),
                [Input('my_ticker_symbol-03', 'value'),
                Input('my_ticker_symbol-02', 'value')]
)
def set_carrier_options(selected_carrier, selected_origin):
                new_data = datadash[datadash['UNIQUE_CARRIER'] == selected_carrier]
                new_data = new_data[new_data['ORIGIN'] == selected_origin]
                return [{'label' : i, 'value' : i} for i in new_data['DEST'].unique()]

#This callback updates the graph on selected data
#for updating the graph we always need two lists i.e. one for each x and y axis
#The graph takes every selected entity as input
@app.callback(Output('my_graph-00', 'figure'),
				[Input('submit-button-00', 'n_clicks'),
                 Input('my_ticker_symbol-03', 'value'),
                 Input('my_ticker_symbol-02', 'value'),
                 Input('my_ticker_symbol-01', 'value'),
				Input('my_date_picker-00', 'start_date'),
                Input('my_date_picker-00', 'end_date'),
                 Input('my_radio-00','value')])
def update_graph(n_clicks,airline,origin,dest,startdate,enddate,radio):
    #The date picker sends the date time format in Y/M/D H:M:S format
    #But we need only date so we take first 10 characters of string
    startdate = startdate[:10]
    enddate = enddate[:10]
    #We convert the string to date time format
    start_date = datetime.datetime.strptime(startdate,'%Y-%m-%d')
    end_date = datetime.datetime.strptime(enddate,'%Y-%m-%d')
    #We select only that dataframe that falls between the selected date range 
    filtered_df = datadash1[datadash1.FL_DATE.between(
        datetime.datetime.strftime(start_date, "%Y-%m-%d"),
        datetime.datetime.strftime(end_date, "%Y-%m-%d")
    )]
    #Now we scrutiny the data according to selected variables
    str1 = 'UNIQUE_CARRIER_'+airline
    str2 = 'ORIGIN_'+origin
    str3 = 'DEST_'+dest
    filtered_df1 = filtered_df[filtered_df[str1]==1]
    filtered_df1 = filtered_df1[filtered_df1[str2]==1]
    filtered_df1 = filtered_df1[filtered_df1[str3]==1]
    #copying the filtered_df to df for further use 
    df = filtered_df1.copy()
    #taking only required data
    dfdelaydash = df[['MONTH','DAY_OF_MONTH','FL_NUM','DAY_OF_WEEK','DEP_HOUR','ARR_HOUR','CRS_ELAPSED_TIME','DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent',
'O_WindSpeedMph','O_PrecipitationPreviousHourInches','O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches',
'D_SnowfallInches','Bird_Strike','UNIQUE_CARRIER_AA','UNIQUE_CARRIER_B6','UNIQUE_CARRIER_DL','UNIQUE_CARRIER_EV','UNIQUE_CARRIER_F9','UNIQUE_CARRIER_NK',
'UNIQUE_CARRIER_OO','UNIQUE_CARRIER_UA','UNIQUE_CARRIER_VX','UNIQUE_CARRIER_WN','ORIGIN_BOS','ORIGIN_DEN','ORIGIN_DFW','ORIGIN_EWR','ORIGIN_IAH','ORIGIN_JFK',
'ORIGIN_LGA','ORIGIN_ORD','ORIGIN_PHL','ORIGIN_SFO','DEST_BOS','DEST_DEN','DEST_DFW','DEST_EWR','DEST_IAH','DEST_JFK','DEST_LGA','DEST_ORD',
'DEST_PHL','DEST_SFO']]
    #make sure taht the number of trained features and number of testing features should be same
        #using delaycoll for reason classification data
    dfdelaydash[delaycoll] = delayscaler.transform(dfdelaydash[delaycoll])
    
    datadash3 = df['FL_DATE']
    #making data ready for predictions
    datadash2 = df.drop(columns=['FL_DATE'],axis=1)
    #using scalar and coll for regression
    datadash2[coll] = scaler.transform(datadash2[coll])
    #using poly for regression(polynomial featuring)
    datadash2 = poly.transform(datadash2)
    #regression predictions
    L = clf.predict(datadash2)
    #reason probability predictions
    delayL = clf1.predict_proba(dfdelaydash)
    #converting the probabilty output to dataframe
    delayL = pd.DataFrame(delayL)
    #separate each reason to lists
    L_carrier = list(round(delayL[0],2))
    L_late_aircraft = list(round(delayL[1],2))
    L_nas = list(round(delayL[2],2))
    L_weather = list(round(delayL[4],2))
    L_security = list(round(delayL[3],2))
    #the code below takes radio option in account
    #It than update the y value(L) by sutracting the contribution of the selected feature in delay
    #It than checks how many flights are going before 15 mins after subtraction
    #it calculates the count of delayed flights
    #it than passes that value of count of on-time and delayed flights to  yaxis
    count1 = 0
    count2 = 0
    if(radio=='S'):
        for i in range(len(L)):
            if(L[i]>15):
                count1 = count1+1
            elif(L[i]<=15):
                count2 = count2+1
    elif(radio=='CARRIER_DELAY'):
        for i in range(len(L)):
            L[i] = L[i]-L_carrier[i]*L[i]
        for i in range(len(L)):
            if(L[i]>15):
                count1 = count1+1
            elif(L[i]<=15):
                count2 = count2+1
    elif(radio=='WEATHER_DELAY'):
        for i in range(len(L)):
            L[i] = L[i]-L_weather[i]*L[i]
        for i in range(len(L)):
            if(L[i]>15):
                count1 = count1+1
            elif(L[i]<=15):
                count2 = count2+1
    elif(radio=='NAS_DELAY'):
        for i in range(len(L)):
            L[i] = L[i]-L_nas[i]*L[i]
        for i in range(len(L)):
            if(L[i]>15):
                count1 = count1+1
            elif(L[i]<=15):
                count2 = count2+1
    elif(radio=='SECURITY_DELAY'):
        for i in range(len(L)):
            L[i] = L[i]-L_security[i]*L[i]
        for i in range(len(L)):
            if(L[i]>15):
                count1 = count1+1
            elif(L[i]<=15):
                count2 = count2+1
    elif(radio=='LATE_AIRCRAFT_DELAY'):
        for i in range(len(L)):
            L[i] = L[i]-L_late_aircraft[i]*L[i]
        for i in range(len(L)):
            if(L[i]>15):
                count1 = count1+1
            elif(L[i]<=15):
                count2 = count2+1

    fig = {
		'data':  [
                {'x': ['On Time','Delayed'], 'y': [count2,count1], 'type': 'bar'},
                
            ],
      'layout': {
                'height': 600,
                'width' : 600
            }
	}
    return fig


#This callback updates the flight list
#it functions as the above callback but outputs the flight getting delayed
#The callback takes every selected entity as input
@app.callback(Output('my_ticker_symbol-00', 'options'),
				[Input('submit-button-00', 'n_clicks'),
                 Input('my_ticker_symbol-03', 'value'),
                 Input('my_ticker_symbol-02', 'value'),
                 Input('my_ticker_symbol-01', 'value'),
				Input('my_date_picker-00', 'start_date'),
                Input('my_date_picker-00', 'end_date'),
                 Input('my_radio-00','value')])
def update_carrier_00(n_clicks,airline,origin,dest,startdate,enddate,radio):
   #The date picker sends the date time format in Y/M/D H:M:S format
    #But we need only date so we take first 10 characters of string
    startdate = startdate[:10]
    enddate = enddate[:10]
    #We convert the string to date time format
    start_date = datetime.datetime.strptime(startdate,'%Y-%m-%d')
    end_date = datetime.datetime.strptime(enddate,'%Y-%m-%d')
    #We select only that dataframe that falls between the selected date range 
    filtered_df = datadash1[datadash1.FL_DATE.between(
        datetime.datetime.strftime(start_date, "%Y-%m-%d"),
        datetime.datetime.strftime(end_date, "%Y-%m-%d")
    )]
    #Now we scrutiny the data according to selected variables
    str1 = 'UNIQUE_CARRIER_'+airline
    str2 = 'ORIGIN_'+origin
    str3 = 'DEST_'+dest
    filtered_df1 = filtered_df[filtered_df[str1]==1]
    filtered_df1 = filtered_df1[filtered_df1[str2]==1]
    filtered_df1 = filtered_df1[filtered_df1[str3]==1]
    #copying the filtered_df to df for further use 
    df = filtered_df1.copy()
    #making another copy of filtered_df as df22
    df22 = df.copy()
    
    #taking only required data
    dfdelaydash = df[['MONTH','DAY_OF_MONTH','FL_NUM','DAY_OF_WEEK','DEP_HOUR','ARR_HOUR','CRS_ELAPSED_TIME','DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent',
'O_WindSpeedMph','O_PrecipitationPreviousHourInches','O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches',
'D_SnowfallInches','Bird_Strike','UNIQUE_CARRIER_AA','UNIQUE_CARRIER_B6','UNIQUE_CARRIER_DL','UNIQUE_CARRIER_EV','UNIQUE_CARRIER_F9','UNIQUE_CARRIER_NK',
'UNIQUE_CARRIER_OO','UNIQUE_CARRIER_UA','UNIQUE_CARRIER_VX','UNIQUE_CARRIER_WN','ORIGIN_BOS','ORIGIN_DEN','ORIGIN_DFW','ORIGIN_EWR','ORIGIN_IAH','ORIGIN_JFK',
'ORIGIN_LGA','ORIGIN_ORD','ORIGIN_PHL','ORIGIN_SFO','DEST_BOS','DEST_DEN','DEST_DFW','DEST_EWR','DEST_IAH','DEST_JFK','DEST_LGA','DEST_ORD',
'DEST_PHL','DEST_SFO']]
    
    #make sure taht the number of trained features and number of testing features should be same
    #using delaycoll for reason classification data
    dfdelaydash[delaycoll] = delayscaler.transform(dfdelaydash[delaycoll])
    datadash3 = df['FL_DATE']
    
    #making data ready for predictions
    datadash2 = df.drop(columns=['FL_DATE'],axis=1)
    #copying datadash2 to datadash use for further use
    datadashuse = datadash2.copy()
    
    #using scalar and coll for regression
    datadash2[coll] = scaler.transform(datadash2[coll])
    
    #using poly for regression(polynomial featuring)
    datadash2 = poly.transform(datadash2)
    
    #regression predictions
    L = clf.predict(datadash2)
    
    #reason probability predictions
    delayL = clf1.predict_proba(dfdelaydash)
    #converting the probabilty output to dataframe
    delayL = pd.DataFrame(delayL)
    
    #separate each reason to lists
    L_carrier = list(round(delayL[0],2))
    L_late_aircraft = list(round(delayL[1],2))
    L_nas = list(round(delayL[2],2))
    L_weather = list(round(delayL[4],2))
    L_security = list(round(delayL[3],2))
    
    count1 = 0
    count2 = 0
    
    #the code below takes radio option in account
    #It than update the y value(L) by sutracting the contribution of the selected feature in delay
    #It than checks how many flights are going before 15 mins after subtraction
    #It than appends arrival delay after subtracting
    #and than iot selects only that data which is greater than 15
    #and than it returns the flightnumber from that dataframe
    K = []
    if(radio=='S'):
        for i in range(len(L)):
            if(L[i]>15):
                count1 = count1+1
                K.append(L[i])
            elif(L[i]<=15):
                count2 = count2+1
                K.append(L[i])
        df22['ARR_DELAY'] = np.array(K)
        result = df22.copy()
        result = result.rename(columns={0:'ARR_DELAY'})
        result = result[result['ARR_DELAY']>15]
        
    elif(radio=='CARRIER_DELAY'):
        for i in range(len(L)):
            L[i] = L[i]-L_carrier[i]*L[i]
        for i in range(len(L)):
            if(L[i]>15):
                count1 = count1+1
                K.append(L[i])
            elif(L[i]<=15):
                count2 = count2+1
                K.append(L[i])
        df22['ARR_DELAY'] = np.array(K)
        result = df22.copy()
        result = result.rename(columns={0:'ARR_DELAY'})
        result = result[result['ARR_DELAY']>15]
    elif(radio=='WEATHER_DELAY'):
        for i in range(len(L)):
            L[i] = L[i]-L_weather[i]*L[i]
        for i in range(len(L)):
            if(L[i]>15):
                count1 = count1+1
                K.append(L[i])
            elif(L[i]<=15):
                count2 = count2+1
                K.append(L[i])
        df22['ARR_DELAY'] = np.array(K)
        result = df22.copy()
        result = result.rename(columns={0:'ARR_DELAY'})
        result = result[result['ARR_DELAY']>15]
    elif(radio=='NAS_DELAY'):
        for i in range(len(L)):
            L[i] = L[i]-L_nas[i]*L[i]
        for i in range(len(L)):
            if(L[i]>15):
                count1 = count1+1
                K.append(L[i])
            elif(L[i]<=15):
                count2 = count2+1
                K.append(L[i])
        df22['ARR_DELAY'] = np.array(K)
        result = df22.copy()
        result = result.rename(columns={0:'ARR_DELAY'})
        result = result[result['ARR_DELAY']>15]
    elif(radio=='SECURITY_DELAY'):
        for i in range(len(L)):
            L[i] = L[i]-L_security[i]*L[i]
        for i in range(len(L)):
            if(L[i]>15):
                count1 = count1+1
                K.append(L[i])
            elif(L[i]<=15):
                count2 = count2+1
                K.append(L[i])
        df22['ARR_DELAY'] = np.array(K)
        result = df22.copy()
        result = result.rename(columns={0:'ARR_DELAY'})
        result = result[result['ARR_DELAY']>15]
    elif(radio=='LATE_AIRCRAFT_DELAY'):
        for i in range(len(L)):
            L[i] = L[i]-L_late_aircraft[i]*L[i]
        for i in range(len(L)):
            if(L[i]>15):
                count1 = count1+1
                K.append(L[i])
            elif(L[i]<=15):
                count2 = count2+1
                K.append(L[i])
        df22['ARR_DELAY'] = np.array(K)
        result = df22.copy()
        result = result.rename(columns={0:'ARR_DELAY'})
        result = result[result['ARR_DELAY']>15]
    
    #This returns the unique list of delayed flights
    L = result.FL_NUM.unique()
    return [{'label' : j, 'value' : j} for j in L]

#This callback updates the text for selected flight
#it functions as the above callback but outputs the reason distribution of selected delayed flight
#The callback takes every selected entity as input
@app.callback(Output('textonly-ce1', 'children'),
				[Input('submit-button-00', 'n_clicks'),
                 Input('my_ticker_symbol-03', 'value'),
                 Input('my_ticker_symbol-02', 'value'),
                 Input('my_ticker_symbol-01', 'value'),
                 Input('my_ticker_symbol-00','value'),
				Input('my_date_picker-00', 'start_date'),
                Input('my_date_picker-00', 'end_date'),
                 Input('my_radio-00','value')])
def update_carrier_1_de6(n_clicks,airline,origin,dest,flnum,startdate,enddate,radio): 
    #The date picker sends the date time format in Y/M/D H:M:S format
    #But we need only date so we take first 10 characters of string
    startdate = startdate[:10]
    enddate = enddate[:10]
    #We convert the string to date time format
    start_date = datetime.datetime.strptime(startdate,'%Y-%m-%d')
    end_date = datetime.datetime.strptime(enddate,'%Y-%m-%d')
    #We select only that dataframe that falls between the selected date range 
    filtered_df = datadash1[datadash1.FL_DATE.between(
        datetime.datetime.strftime(start_date, "%Y-%m-%d"),
        datetime.datetime.strftime(end_date, "%Y-%m-%d")
    )]
    #Now we scrutiny the data according to selected variables
    str1 = 'UNIQUE_CARRIER_'+airline
    str2 = 'ORIGIN_'+origin
    str3 = 'DEST_'+dest
    filtered_df1 = filtered_df[filtered_df[str1]==1]
    filtered_df1 = filtered_df1[filtered_df1[str2]==1]
    filtered_df1 = filtered_df1[filtered_df1[str3]==1]
    traces = []
    #this code updates the text
    #It takes into account the selected flight number
    for i in flnum:
        df = filtered_df1[filtered_df1['FL_NUM']==i]
        #taking only required data
        dfdelaydash = df[['MONTH','DAY_OF_MONTH','FL_NUM','DAY_OF_WEEK','DEP_HOUR','ARR_HOUR','CRS_ELAPSED_TIME','DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent',
'O_WindSpeedMph','O_PrecipitationPreviousHourInches','O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches',
'D_SnowfallInches','Bird_Strike','UNIQUE_CARRIER_AA','UNIQUE_CARRIER_B6','UNIQUE_CARRIER_DL','UNIQUE_CARRIER_EV','UNIQUE_CARRIER_F9','UNIQUE_CARRIER_NK',
'UNIQUE_CARRIER_OO','UNIQUE_CARRIER_UA','UNIQUE_CARRIER_VX','UNIQUE_CARRIER_WN','ORIGIN_BOS','ORIGIN_DEN','ORIGIN_DFW','ORIGIN_EWR','ORIGIN_IAH','ORIGIN_JFK',
'ORIGIN_LGA','ORIGIN_ORD','ORIGIN_PHL','ORIGIN_SFO','DEST_BOS','DEST_DEN','DEST_DFW','DEST_EWR','DEST_IAH','DEST_JFK','DEST_LGA','DEST_ORD',
'DEST_PHL','DEST_SFO']]
        
        #make sure taht the number of trained features and number of testing features should be same
        #using delaycoll for reason classification data
        dfdelaydash[delaycoll] = delayscaler.transform(dfdelaydash[delaycoll])
        #making data ready for predictions
        datadash2 = df.drop(columns=['FL_DATE'],axis=1)
        
        #using scalar and coll for regression
        datadash2[coll] = scaler.transform(datadash2[coll])
        #using poly for regression(polynomial featuring)
        datadash2 = poly.transform(datadash2)
        
        #the code below takes radio option in account
        #it predicts the delay regression 
        #it predicts the delay reason class
        #it than calculates the new delay after subtracting the contribution of selected feature
        #it than only takes the delay>15 into account
        #all the delay reason lists are updated with delay>15 only by using the indices
        #it than converts the selected feature as 0 in the respective list
        #it than creates the dataframe of all the lists and predicts new delay in the case if there is any delay
        #it helps in prediction of new class of delay other tahn the selected delay
        #than it counts total classes and computes delay percentage due to each class
        #it appends the same info as string to the heading division
        if(radio=='S'):
            L = clf.predict(datadash2)
            delayL = clf1.predict(dfdelaydash)
            J = pd.DataFrame(L)
            J = J[J[0]>15]
            delayL = pd.DataFrame(delayL)
            
            O = delayL.shape[0]
            d1 = round((((delayL[delayL[0]=='CARRIER_DELAY']).shape[0])/O),2)
            d2 = round((((delayL[delayL[0]=='WEATHER_DELAY']).shape[0])/O),2)
            d3 = round((((delayL[delayL[0]=='NAS_DELAY']).shape[0])/O),2)
            d4 = round((((delayL[delayL[0]=='LATE_AIRCRAFT_DELAY']).shape[0])/O),2)
            d5 = round((((delayL[delayL[0]=='SECURITY_DELAY']).shape[0])/O),2)
            mystr = "Flight "+str(i)+": Total Delay Count->"+str(J.shape[0])+"Carrier Delay Percent: "+str(round((d1*100),2))+"%"+" Weather Delay Percent: "+str(round((d2*100),2))+"%"+" NAS Delay Percent: "+str(round((d3*100),2))+"%"+" Late Aircraft Delay Percent: "+str(round((d4*100),2))+"%"+" Security Delay Count: "+str(round((d5*100),2))+"%"
            traces.append(html.Div(html.H1(mystr)))
        elif(radio=='CARRIER_DELAY'):
            L = clf.predict(datadash2)
            delayL = clf1.predict_proba(dfdelaydash)
            delayL = pd.DataFrame(delayL)
            L_carrier = list(round(pd.Series(delayL[0]),2))
            L_late_aircraft = list(round(pd.Series(delayL[1]),2))
            L_nas = list(round(pd.Series(delayL[2]),2))
            L_weather = list(round(pd.Series(delayL[4]),2))
            L_security = list(round(pd.Series(delayL[3]),2))
            
            indices = []
            for j in range(len(L)):
                L[j] = L[j]-L_carrier[j]*L[j]
                if(L[j]<=15):
                    indices.append(j)
                    
            J = pd.DataFrame(L)
            J = J[J[0]>15]
            
            L_carrier = [k for j, k in enumerate(L_carrier) if j not in indices]
            L_late_aircraft = [k for j, k in enumerate(L_late_aircraft) if j not in indices]
            L_nas = [k for j, k in enumerate(L_nas) if j not in indices]
            L_weather = [k for j, k in enumerate(L_weather) if j not in indices]
            L_security = [k for j, k in enumerate(L_security) if j not in indices]
            
            
            delayL = pd.DataFrame({'CARRIER_DELAY':L_carrier,'LATE_AIRCRAFT_DELAY':L_late_aircraft,
                                   'WEATHER_DELAY':L_weather,'NAS_DELAY':L_nas,
                                   'SECURITY_DELAY':L_security})
            delayL['CARRIER_DELAY']=0
            delayL['DELAY_CLASS'] = delayL[['CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']].idxmax(axis=1)
            delayL = delayL[['DELAY_CLASS']]
            O = delayL.shape[0]
            
            d1 = round((((delayL[delayL['DELAY_CLASS']=='CARRIER_DELAY']).shape[0])/O),2)
            d2 = round((((delayL[delayL['DELAY_CLASS']=='WEATHER_DELAY']).shape[0])/O),2)
            d3 = round((((delayL[delayL['DELAY_CLASS']=='NAS_DELAY']).shape[0])/O),2)
            d4 = round((((delayL[delayL['DELAY_CLASS']=='LATE_AIRCRAFT_DELAY']).shape[0])/O),2)
            d5 = round((((delayL[delayL['DELAY_CLASS']=='SECURITY_DELAY']).shape[0])/O),2)
            mystr = "Flight "+str(i)+": Total Delay Count->"+str(J.shape[0])+"Carrier Delay Percent: "+str(round((d1*100),2))+"%"+" Weather Delay Percent: "+str(round((d2*100),2))+"%"+" NAS Delay Percent: "+str(round((d3*100),2))+"%"+" Late Aircraft Delay Percent: "+str(round((d4*100),2))+"%"+" Security Delay Count: "+str(round((d5*100),2))+"%"

            traces.append(html.Div(html.H1(mystr)))
        elif(radio=='WEATHER_DELAY'):
            L = clf.predict(datadash2)
            delayL = clf1.predict_proba(dfdelaydash)
            delayL = pd.DataFrame(delayL)
            L_carrier = list(round(pd.Series(delayL[0]),2))
            L_late_aircraft = list(round(pd.Series(delayL[1]),2))
            L_nas = list(round(pd.Series(delayL[2]),2))
            L_weather = list(round(pd.Series(delayL[4]),2))
            L_security = list(round(pd.Series(delayL[3]),2))
            
            indices = []
            for j in range(len(L)):
                L[j] = L[j]-L_weather[j]*L[j]
                if(L[j]<=15):
                    indices.append(j)
            J = pd.DataFrame(L)
            J = J[J[0]>15]
            L_carrier = [k for j, k in enumerate(L_carrier) if j not in indices]
            L_late_aircraft = [k for j, k in enumerate(L_late_aircraft) if j not in indices]
            L_nas = [k for j, k in enumerate(L_nas) if j not in indices]
            L_weather = [k for j, k in enumerate(L_weather) if j not in indices]
            L_security = [k for j, k in enumerate(L_security) if j not in indices]
            
            
            delayL = pd.DataFrame({'CARRIER_DELAY':L_carrier,'LATE_AIRCRAFT_DELAY':L_late_aircraft,
                                   'WEATHER_DELAY':L_weather,'NAS_DELAY':L_nas,
                                   'SECURITY_DELAY':L_security})
            delayL['WEATHER_DELAY']=0
            delayL['DELAY_CLASS'] = delayL[['CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']].idxmax(axis=1)
            delayL = delayL[['DELAY_CLASS']]
            O = delayL.shape[0]
            
            d1 = round((((delayL[delayL['DELAY_CLASS']=='CARRIER_DELAY']).shape[0])/O),2)
            d2 = round((((delayL[delayL['DELAY_CLASS']=='WEATHER_DELAY']).shape[0])/O),2)
            d3 = round((((delayL[delayL['DELAY_CLASS']=='NAS_DELAY']).shape[0])/O),2)
            d4 = round((((delayL[delayL['DELAY_CLASS']=='LATE_AIRCRAFT_DELAY']).shape[0])/O),2)
            d5 = round((((delayL[delayL['DELAY_CLASS']=='SECURITY_DELAY']).shape[0])/O),2)
            mystr = "Flight "+str(i)+": Total Delay Count->"+str(J.shape[0])+"Carrier Delay Percent: "+str(round((d1*100),2))+"%"+" Weather Delay Percent: "+str(round((d2*100),2))+"%"+" NAS Delay Percent: "+str(round((d3*100),2))+"%"+" Late Aircraft Delay Percent: "+str(round((d4*100),2))+"%"+" Security Delay Count: "+str(round((d5*100),2))+"%"

            traces.append(html.Div(html.H1(mystr)))
        elif(radio=='NAS_DELAY'):
            L = clf.predict(datadash2)
            delayL = clf1.predict_proba(dfdelaydash)
            delayL = pd.DataFrame(delayL)
            L_carrier = list(round(pd.Series(delayL[0]),2))
            L_late_aircraft = list(round(pd.Series(delayL[1]),2))
            L_nas = list(round(pd.Series(delayL[2]),2))
            L_weather = list(round(pd.Series(delayL[4]),2))
            L_security = list(round(pd.Series(delayL[3]),2))
            
            indices = []
            for j in range(len(L)):
                L[j] = L[j]-L_nas[j]*L[j]
                if(L[j]<=15):
                    indices.append(j)
            J = pd.DataFrame(L)
            J = J[J[0]>15]
            L_carrier = [k for j, k in enumerate(L_carrier) if j not in indices]
            L_late_aircraft = [k for j, k in enumerate(L_late_aircraft) if j not in indices]
            L_nas = [k for j, k in enumerate(L_nas) if j not in indices]
            L_weather = [k for j, k in enumerate(L_weather) if j not in indices]
            L_security = [k for j, k in enumerate(L_security) if j not in indices]
            
            
            delayL = pd.DataFrame({'CARRIER_DELAY':L_carrier,'LATE_AIRCRAFT_DELAY':L_late_aircraft,
                                   'WEATHER_DELAY':L_weather,'NAS_DELAY':L_nas,
                                   'SECURITY_DELAY':L_security})
            delayL['NAS_DELAY']=0
            delayL['DELAY_CLASS'] = delayL[['CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']].idxmax(axis=1)
            delayL = delayL[['DELAY_CLASS']]
            O = delayL.shape[0]
            
            d1 = round((((delayL[delayL['DELAY_CLASS']=='CARRIER_DELAY']).shape[0])/O),2)
            d2 = round((((delayL[delayL['DELAY_CLASS']=='WEATHER_DELAY']).shape[0])/O),2)
            d3 = round((((delayL[delayL['DELAY_CLASS']=='NAS_DELAY']).shape[0])/O),2)
            d4 = round((((delayL[delayL['DELAY_CLASS']=='LATE_AIRCRAFT_DELAY']).shape[0])/O),2)
            d5 = round((((delayL[delayL['DELAY_CLASS']=='SECURITY_DELAY']).shape[0])/O),2)
            mystr = "Flight "+str(i)+": Total Delay Count->"+str(J.shape[0])+"Carrier Delay Percent: "+str(round((d1*100),2))+"%"+" Weather Delay Percent: "+str(round((d2*100),2))+"%"+" NAS Delay Percent: "+str(round((d3*100),2))+"%"+" Late Aircraft Delay Percent: "+str(round((d4*100),2))+"%"+" Security Delay Count: "+str(round((d5*100),2))+"%"

            traces.append(html.Div(html.H1(mystr)))
        elif(radio=='LATE_AIRCRAFT_DELAY'):
            L = clf.predict(datadash2)
            delayL = clf1.predict_proba(dfdelaydash)
            delayL = pd.DataFrame(delayL)
            L_carrier = list(round(pd.Series(delayL[0]),2))
            L_late_aircraft = list(round(pd.Series(delayL[1]),2))
            L_nas = list(round(pd.Series(delayL[2]),2))
            L_weather = list(round(pd.Series(delayL[4]),2))
            L_security = list(round(pd.Series(delayL[3]),2))
            
            indices = []
            for j in range(len(L)):
                L[j] = L[j]-L_late_aircraft[j]*L[j]
                if(L[j]<=15):
                    indices.append(j)
            J = pd.DataFrame(L)
            J = J[J[0]>15]
            L_carrier = [k for j, k in enumerate(L_carrier) if j not in indices]
            L_late_aircraft = [k for j, k in enumerate(L_late_aircraft) if j not in indices]
            L_nas = [k for j, k in enumerate(L_nas) if j not in indices]
            L_weather = [k for j, k in enumerate(L_weather) if j not in indices]
            L_security = [k for j, k in enumerate(L_security) if j not in indices]
            
            
            delayL = pd.DataFrame({'CARRIER_DELAY':L_carrier,'LATE_AIRCRAFT_DELAY':L_late_aircraft,
                                   'WEATHER_DELAY':L_weather,'NAS_DELAY':L_nas,
                                   'SECURITY_DELAY':L_security})
            delayL['LATE_AIRCRAFT_DELAY']=0
            delayL['DELAY_CLASS'] = delayL[['CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']].idxmax(axis=1)
            delayL = delayL[['DELAY_CLASS']]
            O = delayL.shape[0]
            
            d1 = round((((delayL[delayL['DELAY_CLASS']=='CARRIER_DELAY']).shape[0])/O),2)
            d2 = round((((delayL[delayL['DELAY_CLASS']=='WEATHER_DELAY']).shape[0])/O),2)
            d3 = round((((delayL[delayL['DELAY_CLASS']=='NAS_DELAY']).shape[0])/O),2)
            d4 = round((((delayL[delayL['DELAY_CLASS']=='LATE_AIRCRAFT_DELAY']).shape[0])/O),2)
            d5 = round((((delayL[delayL['DELAY_CLASS']=='SECURITY_DELAY']).shape[0])/O),2)
            mystr = "Flight "+str(i)+": Total Delay Count->"+str(J.shape[0])+"Carrier Delay Percent: "+str(round((d1*100),2))+"%"+" Weather Delay Percent: "+str(round((d2*100),2))+"%"+" NAS Delay Percent: "+str(round((d3*100),2))+"%"+" Late Aircraft Delay Percent: "+str(round((d4*100),2))+"%"+" Security Delay Count: "+str(round((d5*100),2))+"%"

            traces.append(html.Div(html.H1(mystr)))
        elif(radio=='SECURITY_DELAY'):
            L = clf.predict(datadash2)
            delayL = clf1.predict_proba(dfdelaydash)
            delayL = pd.DataFrame(delayL)
            L_carrier = list(round(pd.Series(delayL[0]),2))
            L_late_aircraft = list(round(pd.Series(delayL[1]),2))
            L_nas = list(round(pd.Series(delayL[2]),2))
            L_weather = list(round(pd.Series(delayL[4]),2))
            L_security = list(round(pd.Series(delayL[3]),2))
            
            indices = []
            for j in range(len(L)):
                L[j] = L[j]-L_security[j]*L[j]
                if(L[j]<=15):
                    indices.append(j)
            J = pd.DataFrame(L)
            J = J[J[0]>15]
            L_carrier = [k for j, k in enumerate(L_carrier) if j not in indices]
            L_late_aircraft = [k for j, k in enumerate(L_late_aircraft) if j not in indices]
            L_nas = [k for j, k in enumerate(L_nas) if j not in indices]
            L_weather = [k for j, k in enumerate(L_weather) if j not in indices]
            L_security = [k for j, k in enumerate(L_security) if j not in indices]
            
            
            delayL = pd.DataFrame({'CARRIER_DELAY':L_carrier,'LATE_AIRCRAFT_DELAY':L_late_aircraft,
                                   'WEATHER_DELAY':L_weather,'NAS_DELAY':L_nas,
                                   'SECURITY_DELAY':L_security})
            delayL['SECURITY_DELAY']=0
            delayL['DELAY_CLASS'] = delayL[['CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']].idxmax(axis=1)
            delayL = delayL[['DELAY_CLASS']]
            O = delayL.shape[0]
            
            d1 = round((((delayL[delayL['DELAY_CLASS']=='CARRIER_DELAY']).shape[0])/O),2)
            d2 = round((((delayL[delayL['DELAY_CLASS']=='WEATHER_DELAY']).shape[0])/O),2)
            d3 = round((((delayL[delayL['DELAY_CLASS']=='NAS_DELAY']).shape[0])/O),2)
            d4 = round((((delayL[delayL['DELAY_CLASS']=='LATE_AIRCRAFT_DELAY']).shape[0])/O),2)
            d5 = round((((delayL[delayL['DELAY_CLASS']=='SECURITY_DELAY']).shape[0])/O),2)
            mystr = "Flight "+str(i)+": Total Delay Count->"+str(J.shape[0])+"Carrier Delay Percent: "+str(round((d1*100),2))+"%"+" Weather Delay Percent: "+str(round((d2*100),2))+"%"+" NAS Delay Percent: "+str(round((d3*100),2))+"%"+" Late Aircraft Delay Percent: "+str(round((d4*100),2))+"%"+" Security Delay Count: "+str(round((d5*100),2))+"%"

            traces.append(html.Div(html.H1(mystr)))

    return html.Div(traces)

if __name__ == '__main__':
    app.run_server()