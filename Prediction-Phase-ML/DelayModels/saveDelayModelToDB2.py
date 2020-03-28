# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:18:26 2019

@author: bsoni
"""

import pymongo
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


traindata1 = pd.read_csv('DelayData2016.csv')
traindata2 = pd.read_csv('DelayData2017.csv')
testdata = pd.read_csv('DelayData2018.csv')
birdStrike = pd.read_csv('BirdStrikes.csv') 
traindata2.UNIQUE_CARRIER.unique()
list(traindata1)
traindata1 = traindata1[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]

traindata2 = traindata2[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]


testdata = testdata[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]

traindata1['CARRIER_DELAY'] = traindata1['CARRIER_DELAY'].fillna(0)
traindata1['WEATHER_DELAY'] = traindata1['WEATHER_DELAY'].fillna(0)
traindata1['NAS_DELAY'] = traindata1['NAS_DELAY'].fillna(0)
traindata1['SECURITY_DELAY'] = traindata1['SECURITY_DELAY'].fillna(0)
traindata1['LATE_AIRCRAFT_DELAY'] = traindata1['LATE_AIRCRAFT_DELAY'].fillna(0)

traindata2['CARRIER_DELAY'] = traindata2['CARRIER_DELAY'].fillna(0)
traindata2['WEATHER_DELAY'] = traindata2['WEATHER_DELAY'].fillna(0)
traindata2['NAS_DELAY'] = traindata2['NAS_DELAY'].fillna(0)
traindata2['SECURITY_DELAY'] = traindata2['SECURITY_DELAY'].fillna(0)
traindata2['LATE_AIRCRAFT_DELAY'] = traindata2['LATE_AIRCRAFT_DELAY'].fillna(0)

testdata['CARRIER_DELAY'] = testdata['CARRIER_DELAY'].fillna(0)
testdata['WEATHER_DELAY'] = testdata['WEATHER_DELAY'].fillna(0)
testdata['NAS_DELAY'] = testdata['NAS_DELAY'].fillna(0)
testdata['SECURITY_DELAY'] = testdata['SECURITY_DELAY'].fillna(0)
testdata['LATE_AIRCRAFT_DELAY'] = testdata['LATE_AIRCRAFT_DELAY'].fillna(0)

aggr_dataset = [traindata1 , traindata2 , testdata]
data = pd.concat(aggr_dataset)

data = pd.merge(data,birdStrike,on=['ORIGIN'])
data = data[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches','Bird_Strike']]


carrierlist = traindata1.UNIQUE_CARRIER.unique()
data = data[data.UNIQUE_CARRIER.isin(carrierlist)]
data = data.dropna()

data['ARR_DELAY'] = data['ARR_DELAY'].clip_lower(0)
data1 = data[data['ARR_DELAY']==0]
data2 = data[data['ARR_DELAY']>0]
data1['DELAY_CLASS'] = 'NO_DELAY'
data2['DELAY_CLASS'] = data2[['CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']].idxmax(axis=1)

aggr_dataset = [data1 ,data2]

data = data2

list(data)
data_delay_class = data[['YEAR',
 'MONTH',
 'DAY_OF_MONTH',
 'FL_NUM',
 'DAY_OF_WEEK',
 'FL_DATE',
 'UNIQUE_CARRIER',
 'ORIGIN',
 'DEST',
 'DEP_HOUR',
 'ARR_HOUR',
 'CRS_ELAPSED_TIME',
 'DISTANCE',
 'traffic',
 'O_SurfaceTemperatureFahrenheit',
 'O_CloudCoveragePercent',
 'O_WindSpeedMph',
 'O_PrecipitationPreviousHourInches',
 'O_SnowfallInches',
 'D_SurfaceTemperatureFahrenheit',
 'D_CloudCoveragePercent',
 'D_WindSpeedMph',
 'D_PrecipitationPreviousHourInches',
 'D_SnowfallInches',
 'Bird_Strike',
 'DELAY_CLASS']]

cols_to_transform = ['UNIQUE_CARRIER','ORIGIN','DEST']
df = pd.get_dummies(data_delay_class, columns = cols_to_transform )

df1 = df[df['YEAR']==2016]
df1 = df1.drop(['YEAR','FL_DATE'], axis=1)
df2 = df[df['YEAR']==2017]
df2 = df2.drop(['YEAR','FL_DATE'], axis=1)
df3 = df[df['YEAR']==2018]
df3 = df3.drop(columns=['YEAR','FL_DATE'])

X_test = df3
y_test = X_test['DELAY_CLASS']
X_test = X_test.drop(columns=['DELAY_CLASS'],axis=1)

aggr_dataset = [df1,df2]
traindata = pd.concat(aggr_dataset)
traindata = traindata.dropna()
train_labels = traindata['DELAY_CLASS']
traindata = traindata.drop(columns=['DELAY_CLASS'])

sm = SMOTE(random_state=1)
X_train_res, y_train_res = sm.fit_sample(traindata, train_labels)
X_train_res = pd.DataFrame(X_train_res)
"""
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
"""
X_train_res = X_train_res.rename(columns = {0:'MONTH',1:'DAY_OF_MONTH',2:'FL_NUM',3:'DAY_OF_WEEK',4:'Dep_Hour',
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


X_train_res = traindata
y_train_res = train_labels
X_test = df3
y_test = X_test['DELAY_CLASS']
X_test = X_test.drop(columns=['DELAY_CLASS'],axis=1)


coll = ['CRS_ELAPSED_TIME','DISTANCE','O_SurfaceTemperatureFahrenheit',
 'O_CloudCoveragePercent','O_WindSpeedMph','D_SurfaceTemperatureFahrenheit',
 'D_CloudCoveragePercent','D_WindSpeedMph']

scaler = StandardScaler()
scaler.fit(X_train_res[coll])
X_train_res[coll] = scaler.transform(X_train_res[coll])
X_test[coll] = scaler.transform(X_test[coll])




from sklearn.neural_network import MLPClassifier
clf1 = MLPClassifier(activation='relu',hidden_layer_sizes=(100,50,25), max_iter=3,verbose=2,random_state=1)
clf1.fit(X_train_res,y_train_res)
print( 'Accuracy: ', clf1.score(X_test,y_test))
y_p = clf1.predict_proba(X_test)
y_p = pd.DataFrame(y_p)
y_k = clf1.predict(X_test)

y_k = pd.Series(y_k)
y_k.value_counts()
y_test.value_counts()


import time 

def save_model_to_db(model, client, db, dbconnection, model_name):
    #pickling the model
    pickled_model = pickle.dumps(model)
    
    #saving model to mongoDB
    # creating connection
    myclient = pymongo.MongoClient(client)
    
    #creating database in mongodb
    mydb = myclient[db]
    
    #creating collection
    mycon = mydb[dbconnection]
    info = mycon.insert_one({model_name: pickled_model, 'name': model_name, 'created_time':time.time()})
    print(info.inserted_id, ' saved with this id successfully!')
    
    details = {
        'inserted_id':info.inserted_id,
        'model_name':model_name,
        'created_time':time.time()
    }
    
    return details

details = save_model_to_db(model = clf1, client ='mongodb://localhost:27017/', db = 'models', 
                 dbconnection = 'regression', model_name = 'delay_reason')