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
from app import app

traindata1 = pd.read_csv('Data/DelayData2016.csv')
traindata2 = pd.read_csv('Data/DelayData2017.csv')
testdata = pd.read_csv('Data/DelayData2018.csv')
birdStrike = pd.read_csv('Data/BirdStrikes.csv') 
traindata2.UNIQUE_CARRIER.unique()
list(traindata1)
traindata1 = traindata1[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]

traindata2 = traindata2[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]


testdata = testdata[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]
testdata = testdata.dropna()
aggr_dataset = [traindata1 , traindata2 , testdata]
data = pd.concat(aggr_dataset)
df = data
df = df[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','CRS_ELAPSED_TIME',
'DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches','ARR_DELAY','FL_DATE']]

df = pd.merge(df,birdStrike,on=['ORIGIN'])
df = df[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','CRS_ELAPSED_TIME',
'DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches','Bird_Strike',
'ARR_DELAY','FL_DATE']]

df.dtypes
df = df.drop_duplicates()
df['ARR_DELAY'] = df['ARR_DELAY'].clip_lower(0)
cols_to_transform = ['UNIQUE_CARRIER','ORIGIN','DEST']
carrierlist = traindata1.UNIQUE_CARRIER.unique()
df = df[df.UNIQUE_CARRIER.isin(carrierlist)]


df = pd.get_dummies(df, columns = cols_to_transform )
#df_binary = df[cols_to_transform]
#encoder = ce.OneHotEncoder(cols=cols_to_transform)
#data_encoder = encoder.fit_transform(df_binary)
#df_e = df.drop(columns=cols_to_transform)
#data_final = pd.concat([df_e, data_encoder], axis=1)
#data_final = data_final.drop(columns=['intercept'])
#df = data_final 
df = df.dropna()



df1 = df[df['YEAR']==2016]
df1 = df1.drop(['YEAR','FL_DATE'], axis=1)
df2 = df[df['YEAR']==2017]
df2 = df2.drop(['YEAR','FL_DATE'], axis=1)
df3 = df[df['YEAR']==2018]
df3 = df3.drop(columns=['YEAR'])




aggr_dataset = [df1,df2]
traindata = pd.concat(aggr_dataset)
traindata = traindata.dropna()


z = np.abs(stats.zscore(traindata))
z= pd.DataFrame(z)
z=z.fillna(0)
z=z.values
traindata = traindata[(z < 3.5).all(axis=1)]


traindata['DELAY_CLASS'] = 3
traindata['DELAY_CLASS'] = np.where((traindata.ARR_DELAY>=0)&(traindata.ARR_DELAY <80)  ,0,3)
data1 = traindata[traindata['DELAY_CLASS']==0]
traindata['DELAY_CLASS'] = np.where((traindata.ARR_DELAY>=80)&(traindata.ARR_DELAY >0)  ,1,3)
data2 = traindata[traindata['DELAY_CLASS']==1]

aggr_dataset = [data1,data2]
dataModel = pd.concat(aggr_dataset)
train_labels = dataModel['DELAY_CLASS']
traindata = dataModel.drop(columns=['DELAY_CLASS'])
list(traindata)
sm = SMOTE(random_state=1)
X_train_res, y_train_res = sm.fit_sample(traindata, train_labels)

X_train_res = pd.DataFrame(X_train_res)
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

y_train_res = X_train_res['ARR_DELAY']
X_train_res = X_train_res.drop(columns=['ARR_DELAY'],axis=1)
df3 = df3.dropna()
X_test = df3
X_test_for_delay = X_test.copy()
y_test = X_test['ARR_DELAY']
X_test = X_test.drop(columns=['ARR_DELAY'],axis=1)

y_train_res = pd.Series(y_train_res)
list(X_train_res)




datadash1 = X_test.copy()
datadash1['FL_DATE'] = datadash1['FL_DATE'].astype(str) 
datadash1['DEP_HOUR'] = datadash1['DEP_HOUR'].astype(float) 
datadash1['DEP_HOUR'] = datadash1['DEP_HOUR'].astype(int) 
datadash1['DEP_HOUR'] = datadash1['DEP_HOUR'].astype(str) 
datadash1.dtypes
datadash1['FL_DATE'] = datadash1[['FL_DATE', 'DEP_HOUR']].apply(lambda x: ' '.join(x), axis=1)
datadash1['FL_DATE'] = datadash1['FL_DATE'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d %H")) 

coll = ['CRS_ELAPSED_TIME','DISTANCE','O_SurfaceTemperatureFahrenheit',
 'O_CloudCoveragePercent','O_WindSpeedMph','D_SurfaceTemperatureFahrenheit',
 'D_CloudCoveragePercent','D_WindSpeedMph']

scaler = StandardScaler()
scaler.fit(X_train_res[coll])
X_train_res[coll] = scaler.transform(X_train_res[coll])



poly = PolynomialFeatures(2)
poly.fit(X_train_res)
X_train_res = poly.transform(X_train_res)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++data for delay class++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
delaytraindata1 = pd.read_csv('Data/DelayData2016.csv')
delaytraindata2 = pd.read_csv('Data/DelayData2017.csv')
delaytestdata = pd.read_csv('Data/DelayData2018.csv')
delaybirdStrike = pd.read_csv('Data/BirdStrikes.csv') 
delaytraindata1 = delaytraindata1[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]

delaytraindata2 = delaytraindata2[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]


delaytestdata = delaytestdata[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]

delaytraindata1['CARRIER_DELAY'] = delaytraindata1['CARRIER_DELAY'].fillna(0)
delaytraindata1['WEATHER_DELAY'] = delaytraindata1['WEATHER_DELAY'].fillna(0)
delaytraindata1['NAS_DELAY'] = delaytraindata1['NAS_DELAY'].fillna(0)
delaytraindata1['SECURITY_DELAY'] = delaytraindata1['SECURITY_DELAY'].fillna(0)
delaytraindata1['LATE_AIRCRAFT_DELAY'] = delaytraindata1['LATE_AIRCRAFT_DELAY'].fillna(0)

delaytraindata2['CARRIER_DELAY'] = delaytraindata2['CARRIER_DELAY'].fillna(0)
delaytraindata2['WEATHER_DELAY'] = delaytraindata2['WEATHER_DELAY'].fillna(0)
delaytraindata2['NAS_DELAY'] = delaytraindata2['NAS_DELAY'].fillna(0)
delaytraindata2['SECURITY_DELAY'] = delaytraindata2['SECURITY_DELAY'].fillna(0)
delaytraindata2['LATE_AIRCRAFT_DELAY'] = delaytraindata2['LATE_AIRCRAFT_DELAY'].fillna(0)

delaytestdata['CARRIER_DELAY'] = delaytestdata['CARRIER_DELAY'].fillna(0)
delaytestdata['WEATHER_DELAY'] = delaytestdata['WEATHER_DELAY'].fillna(0)
delaytestdata['NAS_DELAY'] = delaytestdata['NAS_DELAY'].fillna(0)
delaytestdata['SECURITY_DELAY'] = delaytestdata['SECURITY_DELAY'].fillna(0)
delaytestdata['LATE_AIRCRAFT_DELAY'] = delaytestdata['LATE_AIRCRAFT_DELAY'].fillna(0)

aggr_dataset = [delaytraindata1 , delaytraindata2 , delaytestdata]
delaydata = pd.concat(aggr_dataset)

delaydata = pd.merge(delaydata,delaybirdStrike,on=['ORIGIN'])
delaydata = delaydata[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches','Bird_Strike']]


delaycarrierlist = delaytraindata1.UNIQUE_CARRIER.unique()
delaydata = delaydata[delaydata.UNIQUE_CARRIER.isin(delaycarrierlist)]
delaydata = delaydata.dropna()

delaydata['ARR_DELAY'] = delaydata['ARR_DELAY'].clip_lower(0)
delaydata1 = delaydata[delaydata['ARR_DELAY']==0]
delaydata2 = delaydata[delaydata['ARR_DELAY']>0]
delaydata1['DELAY_CLASS'] = 'NO_DELAY'
delaydata2['DELAY_CLASS'] = delaydata2[['CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']].idxmax(axis=1)

aggr_dataset = [delaydata1 ,delaydata2]

delaydata = pd.concat(aggr_dataset)


delaydata_delay_class = delaydata[['YEAR',
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
delaydf = pd.get_dummies(delaydata_delay_class, columns = cols_to_transform )

delaydf1 = delaydf[delaydf['YEAR']==2016]
delaydf1 = delaydf1.drop(['YEAR','FL_DATE'], axis=1)
delaydf2 = delaydf[delaydf['YEAR']==2017]
delaydf2 = delaydf2.drop(['YEAR','FL_DATE'], axis=1)
delaydf3 = delaydf[delaydf['YEAR']==2018]
delaydf3 = delaydf3.drop(columns=['YEAR','FL_DATE'])



aggr_dataset = [delaydf1,delaydf2]
delaytraindata = pd.concat(aggr_dataset)
delay_category = ['CARRIER_DELAY','WEATHER_DELAY','NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']
delaytraindata = delaytraindata[delaytraindata.DELAY_CLASS.isin(delay_category)]
delaytraindata = delaytraindata.dropna()
delaytrain_labels = delaytraindata['DELAY_CLASS']
delaytraindata = delaytraindata.drop(columns=['DELAY_CLASS'])
list(delaytraindata)
X_test_for_delay = X_test_for_delay[['MONTH','DAY_OF_MONTH','FL_NUM','DAY_OF_WEEK','DEP_HOUR','ARR_HOUR','CRS_ELAPSED_TIME','DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent',
'O_WindSpeedMph','O_PrecipitationPreviousHourInches','O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches',
'D_SnowfallInches','Bird_Strike','UNIQUE_CARRIER_AA','UNIQUE_CARRIER_B6','UNIQUE_CARRIER_DL','UNIQUE_CARRIER_EV','UNIQUE_CARRIER_F9','UNIQUE_CARRIER_NK',
'UNIQUE_CARRIER_OO','UNIQUE_CARRIER_UA','UNIQUE_CARRIER_VX','UNIQUE_CARRIER_WN','ORIGIN_BOS','ORIGIN_DEN','ORIGIN_DFW','ORIGIN_EWR','ORIGIN_IAH','ORIGIN_JFK',
'ORIGIN_LGA','ORIGIN_ORD','ORIGIN_PHL','ORIGIN_SFO','DEST_BOS','DEST_DEN','DEST_DFW','DEST_EWR','DEST_IAH','DEST_JFK','DEST_LGA','DEST_ORD',
'DEST_PHL','DEST_SFO']]

sm = SMOTE(random_state=1)
delayX_train_res, delayy_train_res = sm.fit_sample(delaytraindata, delaytrain_labels)
delayX_train_res = pd.DataFrame(delayX_train_res)
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

delaycoll = ['CRS_ELAPSED_TIME','DISTANCE','O_SurfaceTemperatureFahrenheit',
 'O_CloudCoveragePercent','O_WindSpeedMph','D_SurfaceTemperatureFahrenheit',
 'D_CloudCoveragePercent','D_WindSpeedMph']

delayscaler = StandardScaler()
delayscaler.fit(delayX_train_res[delaycoll])
delayX_train_res[delaycoll] = delayscaler.transform(delayX_train_res[delaycoll])
X_test_for_delay[delaycoll] = delayscaler.transform(X_test_for_delay[delaycoll])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++models+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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

clf  = load_saved_model_from_db(model_name = 'delay_1', client = 'mongodb://localhost:27017/', 
                         db = 'models', dbconnection = 'regression')

clf1  = load_saved_model_from_db(model_name = 'delay_reason', client = 'mongodb://localhost:27017/', 
                         db = 'models', dbconnection = 'regression')

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++app starts+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

datadash = testdata
available_indicator_airline = traindata1['UNIQUE_CARRIER'].unique()

layout =  html.Div([
                html.Div( 
                 [html.H3([html.Span('DELAY PREDICTIONS',className="main-heading-primary-in"),
                           html.Span('DELAY PREDICTIONS WITH REASON DISTRIBUTION',className="main-heading-secondary")]
                 ,className='main-heading'),
				],style={'margin-top':'-20px'}),
html.Div([
                html.A([
                        html.Img(
                            src=app.get_asset_url('home.png'),
                            style={
                                'height' : '50px',
                                'width' : '50px',
                                'float' : 'right',
                                'position' : 'absolute',
                                'padding-top' : 0,
                                'padding-right' : 0,
                                'margin-left':'45',
                                'margin-top':'-40px',
                            })
                ],className='hello',href='/')
            ],style={'float':'top','margin-right':'1500px','height':'0px','width':'0px','background-color':'#ffffff'}) ,

html.Div([
                html.A([
                        html.Img(
                            src=app.get_asset_url('back.png'),
                            style={
                                'height' : '50px',
                                'width' : '50px',
                                'float' : 'right',
                                'position' : 'absolute',
                                'padding-top' : 0,
                                'padding-right' : 0,
                                'margin-right':'10px',
                                'margin-top':'-40px',
                            })
                ],className='hello',href='/Delays/delayindex')
            ],style={'float':'top','margin-left':'95%','height':'0px','width':'0px','background-color':'#ffffff'}) ,
    


html.Div([
    
                
				 html.Div([html.H3('Carrier code:', style={'margin-left': '149px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-DD13',
						  options=[{'label': i, 'value': i} for i in available_indicator_airline],
						  # value = ['SPY'], 
                          value='',
                          
						  style={'fontSize': 15, 'width': 300,'margin-left': '75px',},
				)], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-left': '220px','margin-top': '-39px'}),
    
    
                html.Div([html.H3('Origin Airport:', style={'margin-left': '149px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-DD12',
                          
						  style={'fontSize': 15, 'width': 300,'margin-left': '75px'},
                           
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-top': '-39px'}),
                
                
                
                html.Div([html.H3('Destination Airport:', style={'margin-left': '149px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-DD11',
						  style={'fontSize': 15, 'width': 300,'margin-left': '75px'},
                          
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-left': '220px','margin-top': '-12px'}),
            
                
                
                html.Div([html.H3('Flight number:', style={'margin-left': '149px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-DD10',
						  multi=True,
						  # value = ['SPY'], 
                          value='',
                          
						  style={'fontSize': 15, 'width': 300,'margin-left': '75px'},
				)], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-top': '-12px'}),
    

				html.Div([html.H3('Date Range:'),
					dcc.DatePickerRange(id='my_date_picker-DD10',
										min_date_allowed = dt(2018,1,1),
										max_date_allowed = dt(2018,12,12),
										start_date = dt(2018, 1,2),
										end_date = dt(2018, 4,28),
                                         display_format='MMM Do, YY',
                                         
					)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '28%','margin-left': '370px','margin-top': '-10px'}), 
                    
				html.Div([
					html.Button(id='submit-button-DD10',
								n_clicks = 0,
								children = 'Submit',
								style = {'fontSize': 24, 'marginLeft': '30px','margin-left': '75px'}

					)

				], style={'display': 'None'}),
				# dcc.Markdown(''' --- '''),

				# Total Return Charts section
				# html.H1('Total Return Charts'),
					
				dcc.Markdown(''' --- '''), 
				
				html.Div([dcc.Graph(id='my_graph-DD10',
							figure={'layout':go.Layout(title='daily delays will be shown here', 
                               
                                         )}
				), ],id='graph',style={'width': '85%', 'display': 'inline-block', 'padding': '0 20','margin-left': '220px','margin-top': '-45px'}),

],style={'margin-top':'250px'}),
    
], id='particles-js')
    
@app.callback(
                Output('my_ticker_symbol-DD12', 'options'),
                [Input('my_ticker_symbol-DD13', 'value')]
)
def set_destination_options_DD1(selected_origin):
                new_data = datadash[datadash['UNIQUE_CARRIER'] == selected_origin]
                return [{'label' : i, 'value' : i} for i in new_data['ORIGIN'].unique()]

#This callback is to return selected airline and origin airport for available options for destination airport
@app.callback(
                Output('my_ticker_symbol-DD11', 'options'),
                [Input('my_ticker_symbol-DD13', 'value'),
                Input('my_ticker_symbol-DD12', 'value')]
)
def set_carrier_options_DD1(selected_carrier, selected_origin):
                new_data = datadash[datadash['UNIQUE_CARRIER'] == selected_carrier]
                new_data = new_data[new_data['ORIGIN'] == selected_origin]
                return [{'label' : i, 'value' : i} for i in new_data['DEST'].unique()]
#This callback is to return selected airline, origin airport and destination airport for available options for flight number
@app.callback(
                Output('my_ticker_symbol-DD10', 'options'),
                [Input('my_ticker_symbol-DD13', 'value'),
                Input('my_ticker_symbol-DD12', 'value'),
                Input('my_ticker_symbol-DD11', 'value')]
)
def set_flight_options_DD1(selected_carrier, selected_origin, selected_dest):
                new_data = datadash[datadash['UNIQUE_CARRIER'] == selected_carrier]
                new_data = new_data[new_data['ORIGIN'] == selected_origin]
                new_data = new_data[new_data['DEST'] == selected_dest]
                return [{'label' : i, 'value' : i} for i in new_data['FL_NUM'].unique()]


@app.callback(Output('my_graph-DD10', 'figure'),
				[Input('submit-button-DD10', 'n_clicks'),
                 Input('my_ticker_symbol-DD13', 'value'),
                 Input('my_ticker_symbol-DD12', 'value'),
                 Input('my_ticker_symbol-DD11', 'value'),
                 Input('my_ticker_symbol-DD10', 'value'),
				Input('my_date_picker-DD10', 'start_date'),
                Input('my_date_picker-DD10', 'end_date')])
def update_graph_DD1(n_clicks,airline,origin,dest,flnum,startdate,enddate):
    
    print("++++++++++++++",app)
    print(type(startdate))
    startdate = startdate[:10]
    enddate = enddate[:10]
    print(startdate)
    #datadash['FLL_DATE']=datadash['FLL_DATE'].apply(lambda x: x.date())
    start_date = datetime.datetime.strptime(startdate,'%Y-%m-%d')
    end_date = datetime.datetime.strptime(enddate,'%Y-%m-%d')
    filtered_df = datadash1[datadash1.FL_DATE.between(
        datetime.datetime.strftime(start_date, "%Y-%m-%d"),
        datetime.datetime.strftime(end_date, "%Y-%m-%d")
    )]
    str1 = 'UNIQUE_CARRIER_'+airline
    str2 = 'ORIGIN_'+origin
    str3 = 'DEST_'+dest
    filtered_df1 = filtered_df[filtered_df[str1]==1]
    filtered_df1 = filtered_df1[filtered_df1[str2]==1]
    filtered_df1 = filtered_df1[filtered_df1[str3]==1]
    traces = []
    print("-----------------===========================",flnum)
    for i in flnum:
        df = filtered_df1[filtered_df1['FL_NUM']==i]
        dfdelaydash = df[['MONTH','DAY_OF_MONTH','FL_NUM','DAY_OF_WEEK','DEP_HOUR','ARR_HOUR','CRS_ELAPSED_TIME','DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent',
'O_WindSpeedMph','O_PrecipitationPreviousHourInches','O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches',
'D_SnowfallInches','Bird_Strike','UNIQUE_CARRIER_AA','UNIQUE_CARRIER_B6','UNIQUE_CARRIER_DL','UNIQUE_CARRIER_EV','UNIQUE_CARRIER_F9','UNIQUE_CARRIER_NK',
'UNIQUE_CARRIER_OO','UNIQUE_CARRIER_UA','UNIQUE_CARRIER_VX','UNIQUE_CARRIER_WN','ORIGIN_BOS','ORIGIN_DEN','ORIGIN_DFW','ORIGIN_EWR','ORIGIN_IAH','ORIGIN_JFK',
'ORIGIN_LGA','ORIGIN_ORD','ORIGIN_PHL','ORIGIN_SFO','DEST_BOS','DEST_DEN','DEST_DFW','DEST_EWR','DEST_IAH','DEST_JFK','DEST_LGA','DEST_ORD',
'DEST_PHL','DEST_SFO']]
        dfdelaydash[delaycoll] = delayscaler.transform(dfdelaydash[delaycoll])
        print("done1")
        datadash3 = df['FL_DATE']
        datadash2 = df.drop(columns=['FL_DATE'],axis=1)
        datadash2[coll] = scaler.transform(datadash2[coll])
        datadash2 = poly.transform(datadash2)
        print("done2")
        print(datadash2.shape)
        L = clf.predict(datadash2)
        delayL = clf1.predict_proba(dfdelaydash)
        delayL = pd.DataFrame(delayL)
        L_carrier = list(round(delayL[0]*100,2))
        L_late_aircraft = list(round(delayL[1]*100,2))
        L_nas = list(round(delayL[2]*100,2))
        L_weather = list(round(delayL[3]*100,2))
        L_security = list(round(delayL[4]*100,2))
        hoverList = []
        for j in range(len(L)):
            if(L[j]>0):
                a = str(round(L[j],2))+" minutes"+"<br>"+"Carrier Delay "+str(L_carrier[j])+"%"+"<br>"+"Late Aircraft Delay "+str(L_late_aircraft[j])+"%"+"<br>"+"NAS Delay "+str(L_nas[j])+"%"+"<br>"+"Weather Delay "+str(L_weather[j])+"%"+"<br>"+"Security Delay "+str(L_security[j])+"%"
                hoverList.append(str(a))
            elif(L[j]<=0):
                a = "0"
                hoverList.append(str(a))
        print("done3")
        depD = L.tolist()
        print("done4")
        depHour = datadash3.tolist()
        lists = sorted(zip( depHour,depD,hoverList))
        new_x, new_y, hoverData = list(zip(*lists))
        traces.append({'x':new_x, 'y':new_y,'name':i, 'text' : hoverData,'hoverinfo' : 'text'})
        print("done5")
        print(flnum)
    fig = {
		'data': traces,
		'layout': {'title':flnum,
             'xaxis' : {'title':'Date'},
                'yaxis' : {'title':'Arrival Delay'},},
        
        'transition': {
                'duration': 5000,
            },
        'frame': {'duration': 5000, 'redraw': False},
      
	}
    return fig


    
if __name__ == '__main__':
    app.run_server()