# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:55:21 2019

@author: bsoni
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import datetime
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime as dt
from sklearn import datasets, metrics
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle
from app import app

traindata = pd.read_csv('Data/weathertrafficandflights-2016.csv')
testdata = pd.read_csv('Data/weathertrafficandflights-2017.csv')
testdata1 = pd.read_csv('Data/weathertrafficandflights-2018.csv')

traindata['YEAR']=2016
testdata['YEAR']=2017
testdata1['YEAR']=2018
list(traindata)

traindata['CRS_DEP_TIME'] = 0
traindata = traindata[['DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','FL_NUM','ORIGIN','DEST','Hour',
'CANCELLED','CANCELLATION_CODE','DISTANCE','traffic','O_SurfaceTemperatureFahrenheit',
'O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent',
'D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches','YEAR','CRS_DEP_TIME']]

testdata['CRS_DEP_TIME'] = 0
testdata= testdata[['DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','FL_NUM','ORIGIN','DEST','Hour',
'CANCELLED','CANCELLATION_CODE','DISTANCE','traffic','O_SurfaceTemperatureFahrenheit',
'O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent',
'D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches','YEAR','CRS_DEP_TIME']]

testdata1= testdata1[['DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','FL_NUM','ORIGIN','DEST','Hour',
'CANCELLED','CANCELLATION_CODE','DISTANCE','traffic','O_SurfaceTemperatureFahrenheit',
'O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent',
'D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches','YEAR','CRS_DEP_TIME']]

aggr_dataset = [traindata,testdata,testdata1]
data = pd.concat(aggr_dataset)
data = data.sample(frac=1).reset_index(drop=True)
data.dtypes
data['FLL_DATE'] = data['FL_DATE'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d")) 
data['DATE_ONLY']=data['FLL_DATE'].apply(lambda x: x.strftime('%d'))
data['MONTH_ONLY']=data['FLL_DATE'].apply(lambda x: x.strftime('%m'))
datadash = data.copy()
data = data[['DAY_OF_WEEK',
           	'UNIQUE_CARRIER','CANCELLED',	'FL_NUM',	'ORIGIN',	'DEST',	
           'Hour',	'DISTANCE',	
           'traffic',	'O_SurfaceTemperatureFahrenheit',	'O_CloudCoveragePercent',	
           'O_WindSpeedMph',	'O_PrecipitationPreviousHourInches',	'O_SnowfallInches',	
           'D_SurfaceTemperatureFahrenheit',	'D_CloudCoveragePercent','D_WindSpeedMph',	
           'D_PrecipitationPreviousHourInches',	'D_SnowfallInches','DATE_ONLY','MONTH_ONLY','YEAR','CRS_DEP_TIME']]


"""
cols_to_transform = [ 'UNIQUE_CARRIER', 'ORIGIN', 'DEST']
df_with_dummies = pd.get_dummies(data, columns = cols_to_transform )
print("=====================================",df_with_dummies.shape)

y = df_with_dummies['CANCELLED']
x = df_with_dummies.drop(columns=['CANCELLED'])

x_test = df_with_dummies[df_with_dummies['YEAR']==2018]
y_test = x_test['CANCELLED']
x_test = x_test.drop(columns=['CANCELLED'])

training_features, test_features, training_target, test_target, = train_test_split(x, y, test_size=0.15, random_state=12)
print (training_features.shape, test_features.shape)
print (training_target.shape, test_target.shape)
print(list(df_with_dummies))
x_train, x_val, y_train, y_val = train_test_split(training_features, training_target, test_size = .15, random_state=12)

clf_rf = RandomForestClassifier(n_estimators=100, random_state=12)
clf_rf.fit(x_train, y_train)

x_test = df_with_dummies[df_with_dummies['YEAR']==2018]
x_test = x_test[x_test['CANCELLED']==0]
y_test = x_test['CANCELLED']
x_test = x_test.drop(columns=['CANCELLED'])
print( 'Accuracy: ', clf_rf.score(x_test, y_test))
"""

filename = 'cancellation_model.sav'

"""
pickle.dump(clf_rf, open(filename, 'wb'))
"""
clf_rf = pickle.load(open(filename, 'rb'))


#===================================================for cancellation class===================================================================
datadash_class = datadash.copy()
datadash_class = datadash_class[datadash_class['CANCELLED']==1]
list(datadash_class)
datadash_class = datadash_class[['DAY_OF_WEEK',
           	'UNIQUE_CARRIER','CANCELLATION_CODE',	'FL_NUM',	'ORIGIN',	'DEST',	
           'Hour',	'DISTANCE',	
           'traffic',	'O_SurfaceTemperatureFahrenheit',	'O_CloudCoveragePercent',	
           'O_WindSpeedMph',	'O_PrecipitationPreviousHourInches',	'O_SnowfallInches',	
           'D_SurfaceTemperatureFahrenheit',	'D_CloudCoveragePercent','D_WindSpeedMph',	
           'D_PrecipitationPreviousHourInches',	'D_SnowfallInches','DATE_ONLY','MONTH_ONLY','YEAR']]

"""
cols_to_transform = [ 'UNIQUE_CARRIER', 'ORIGIN', 'DEST']
df_with_dummies_class = pd.get_dummies(datadash_class, columns = cols_to_transform )
print("=====================================",df_with_dummies_class.shape)

y_class = df_with_dummies_class['CANCELLATION_CODE']
x_class = df_with_dummies_class.drop(columns=['CANCELLATION_CODE'])

x_test_class = df_with_dummies_class[df_with_dummies_class['YEAR']==2018]
y_test_class = x_test_class['CANCELLATION_CODE']
x_test_class = x_test_class.drop(columns=['CANCELLATION_CODE'])

training_features_class, test_features_class, training_target_class, test_target_class, = train_test_split(x_class, y_class, test_size=0.15, random_state=12)
print (training_features_class.shape, test_features_class.shape)
print (training_target_class.shape, test_target_class.shape)
print(list(df_with_dummies_class))
x_train_class, x_val_class, y_train_class, y_val_class = train_test_split(training_features_class, training_target_class, test_size = .15, random_state=12)

clf_rf_class = RandomForestClassifier(n_estimators=100, random_state=12)
clf_rf_class.fit(x_train_class, y_train_class)

print( 'Accuracy: ', clf_rf_class.score(x_test_class, y_test_class))
"""
filename = 'cancellation_reason_model.sav'

"""
pickle.dump(clf_rf_class, open(filename, 'wb'))
"""

clf_rf_class = pickle.load(open(filename, 'rb'))


#=====================================================datadash operations====================================================================
datadash1 = datadash.drop(columns=['CANCELLED','CRS_DEP_TIME'])
datadash1 = datadash1[datadash1['YEAR']==2018]
datadashdash = datadash1.copy()
available_indicator_airline = datadash1['UNIQUE_CARRIER'].unique()
cols_to_transform = [ 'UNIQUE_CARRIER', 'ORIGIN', 'DEST']
datadash1 = pd.get_dummies(datadash1, columns = cols_to_transform )
mycrsdata = datadash.copy()
mycrsdata = mycrsdata[mycrsdata['YEAR']==2018]
crstestdata= mycrsdata[['DAY_OF_WEEK','CRS_DEP_TIME','FL_DATE','UNIQUE_CARRIER','FL_NUM','ORIGIN','DEST','Hour',
'CANCELLED','CANCELLATION_CODE','DISTANCE','traffic','O_SurfaceTemperatureFahrenheit',
'O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent',
'D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches','YEAR','FLL_DATE']]


layout =    html.Div([
        html.Div( 
                 [html.H3([html.Span('CANCELLATION PREDICTION',className="main-heading-primary-in"),
                           html.Span('CANCELLATION PREDICTION PROBABILITIES WITH FEATURE CONTROL ANALYSIS',className="main-heading-secondary")]
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
                ],className='hello',href='/Cancellations/cancellationindex')
            ],style={'float':'top','margin-left':'95%','height':'0px','width':'0px','background-color':'#ffffff'}) ,
    
html.Div([
				                html.Div([html.H3('Carrier code:', style={'margin-left': '149px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-C23',
						  options=[{'label': i, 'value': i} for i in available_indicator_airline],
						  # value = ['SPY'], 
                          value='',
                          
						  style={'fontSize': 15, 'width': 300,'margin-left': '75px'},
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-left': '220px','margin-top':'-40px'}),
                html.Div([html.H3('Origin Airport:', style={'margin-left': '149px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-C22',
						   style={'fontSize': 15, 'width': 300,'margin-left': '75px'},
                           
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-top':'-40px'}),
                html.Div([html.H3('Destination Airport:', style={'margin-left': '149px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-C21',
						  style={'fontSize': 15, 'width': 300,'margin-left': '75px'},
                          
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-left': '220px'}),
                
				html.Div([html.H3('Flight number:', style={'margin-left': '149px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-C20',
						   # value = ['SPY'], 
						  multi = True,
						  style={'fontSize': 15, 'width': 300,'margin-left': '75px'},
                          
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
                
				html.Div([html.H3('Date Range:'),
					dcc.DatePickerRange(id='my_date_picker-C20',
										min_date_allowed = dt(2018,1,1),
										max_date_allowed = dt(2018,12,12),
										start_date = dt(2018, 1,2),
										end_date = dt(2018, 4,28),
                                         display_format='MMM Do, YY',
                                         
					)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '28%','margin-left': '370px'}), 
                    
				html.Div([
					html.Button(id='submit-button-C20',
								n_clicks = 0,
								children = 'Submit',
								style = {'fontSize': 24, 'marginLeft': '30px','margin-left': '75px'}

					)

				], style={'display': 'None'}),
				# dcc.Markdown(''' --- '''),

				# Total Return Charts section
				# html.H1('Total Return Charts'),
					
				dcc.Markdown(''' --- '''), 
				
				html.Div([dcc.Graph(id='my_graph-C20',
							figure={'layout':go.Layout(title='daily Cancellations will be shown here', 
                               
                                         )}
				), ],style={'width': '85%', 'display': 'inline-block', 'padding': '0 20','margin-left': '220px','margin-top':'-30px'}),

    
               html.Div([dcc.RadioItems(
            id='my_radio-C20',
    options=[
            {'label': 'None', 'value': 'S'},
        {'label': 'Airlines/Carrier', 'value': 'CARRIER'},
        {'label': 'Weather', 'value': 'WEATHER'},
        {'label': 'National Air System', 'value': 'NAS'},
        {'label': 'Security', 'value': 'SECURITY'},
   
    ],
    value='S'
)],style={'display':'block','margin-top':'-380px','margin-left':'40px'}),
					
],style={'margin-top':'250px'}),
])
@app.callback(
                Output('my_ticker_symbol-C22', 'options'),
                [Input('my_ticker_symbol-C23', 'value')]
)
def set_origin_options(selected_airline):
                new_data = datadashdash[datadashdash['UNIQUE_CARRIER'] == selected_airline]
                return [{'label' : i, 'value' : i} for i in new_data['ORIGIN'].unique()]

#This callback is to return selected airline and origin airport for available options for destination airport
@app.callback(
                Output('my_ticker_symbol-C21', 'options'),
                [Input('my_ticker_symbol-C23', 'value'),
                Input('my_ticker_symbol-C22', 'value')]
)
def set_destination_options(selected_airline, origin_airport):
                new_data = datadashdash[datadashdash['UNIQUE_CARRIER'] == selected_airline]
                new_data = new_data[new_data['ORIGIN'] == origin_airport]
                return [{'label' : i, 'value' : i} for i in new_data['DEST'].unique()]
#This callback is to return selected airline, origin airport and destination airport for available options for flight number
@app.callback(
                dash.dependencies.Output('my_ticker_symbol-C20', 'options'),
                [dash.dependencies.Input('my_ticker_symbol-C23', 'value'),
                dash.dependencies.Input('my_ticker_symbol-C22', 'value'),
                dash.dependencies.Input('my_ticker_symbol-C21', 'value')]
)
def set_flightno_options(selected_airline, origin_airport, destination_airport):
                new_data = datadashdash[datadashdash['UNIQUE_CARRIER'] == selected_airline]
                new_data = new_data[new_data['ORIGIN'] == origin_airport]
                new_data = new_data[new_data['DEST'] == destination_airport]
                new_data = new_data[new_data['YEAR']==2018]
                print(new_data['FL_NUM'].unique())
                return [{'label' : i, 'value' : i} for i in new_data['FL_NUM'].unique()]

@app.callback(Output('my_graph-C20', 'figure'),
				[Input('submit-button-C20', 'n_clicks'),
                 Input('my_ticker_symbol-C23', 'value'),
                 Input('my_ticker_symbol-C22', 'value'),
                 Input('my_ticker_symbol-C21', 'value'),
				Input('my_ticker_symbol-C20', 'value'),  
				Input('my_date_picker-C20', 'start_date'),
                Input('my_date_picker-C20', 'end_date'),
                Input('my_radio-C20', 'value'),])
def update_graph(n_clicks,airline,origin,dest,stock_ticker,startdate,enddate,radio):
    
    print("looooooooooooooooooooooooool")
    print(type(startdate))
    startdate = startdate[:10]
    enddate = enddate[:10]
    print(startdate)
    #datadash['FLL_DATE']=datadash['FLL_DATE'].apply(lambda x: x.date())
    start_date = datetime.datetime.strptime(startdate,'%Y-%m-%d')
    end_date = datetime.datetime.strptime(enddate,'%Y-%m-%d')
    filtered_df = datadash1[datadash1.FLL_DATE.between(
        datetime.datetime.strftime(start_date, "%Y-%m-%d"),
        datetime.datetime.strftime(end_date, "%Y-%m-%d")
    )]
    crs_filtered_df = crstestdata[crstestdata.FLL_DATE.between(
        datetime.datetime.strftime(start_date, "%Y-%m-%d"),
        datetime.datetime.strftime(end_date, "%Y-%m-%d")
    )]
    
    str1 = 'UNIQUE_CARRIER_'+airline
    str2 = 'ORIGIN_'+origin
    str3 = 'DEST_'+dest
    filtered_df1 = filtered_df[filtered_df[str1]==1]
    filtered_df1 = filtered_df1[filtered_df1[str2]==1]
    filtered_df1 = filtered_df1[filtered_df1[str3]==1]
    crs_filtered_df_1 = crs_filtered_df[crs_filtered_df['UNIQUE_CARRIER']==airline]
    crs_filtered_df_1 = crs_filtered_df_1[crs_filtered_df_1['ORIGIN']==origin]
    crs_filtered_df_1 = crs_filtered_df_1[crs_filtered_df_1['DEST']==dest]
    
    
    print(type(stock_ticker))
    for i in stock_ticker:
        print(type(i))
    traces = []
    for i in stock_ticker:
        df = filtered_df1[filtered_df1['FL_NUM']==i]
        crs_df = crs_filtered_df_1[crs_filtered_df_1['FL_NUM']==i]
        print("done1")
        datadash3 = df[['DAY_OF_WEEK', 'FL_NUM', 'Hour', 'DISTANCE', 'traffic', 'O_SurfaceTemperatureFahrenheit', 'O_CloudCoveragePercent',
                        'O_WindSpeedMph', 'O_PrecipitationPreviousHourInches', 'O_SnowfallInches', 'D_SurfaceTemperatureFahrenheit', 
                        'D_CloudCoveragePercent', 'D_WindSpeedMph', 'D_PrecipitationPreviousHourInches', 'D_SnowfallInches', 'DATE_ONLY', 
                        'MONTH_ONLY', 'YEAR', 'UNIQUE_CARRIER_9E', 'UNIQUE_CARRIER_AA', 'UNIQUE_CARRIER_AS', 'UNIQUE_CARRIER_B6', 
                        'UNIQUE_CARRIER_DL', 'UNIQUE_CARRIER_EV', 'UNIQUE_CARRIER_F9', 'UNIQUE_CARRIER_MQ', 'UNIQUE_CARRIER_NK',
                        'UNIQUE_CARRIER_OH', 'UNIQUE_CARRIER_OO', 'UNIQUE_CARRIER_UA', 'UNIQUE_CARRIER_VX', 'UNIQUE_CARRIER_WN', 
                        'UNIQUE_CARRIER_YV', 'UNIQUE_CARRIER_YX', 'ORIGIN_BOS', 'ORIGIN_DEN', 'ORIGIN_DFW', 'ORIGIN_EWR', 'ORIGIN_IAH', 
                        'ORIGIN_JFK', 'ORIGIN_LGA', 'ORIGIN_ORD', 'ORIGIN_PHL', 'ORIGIN_SFO', 'DEST_BOS', 'DEST_DEN', 'DEST_DFW', 'DEST_EWR', 
                        'DEST_IAH','DEST_JFK', 'DEST_LGA', 'DEST_ORD', 'DEST_PHL', 'DEST_SFO','FLL_DATE']]
        
        datadash2 = df[['DAY_OF_WEEK', 'FL_NUM', 'Hour', 'DISTANCE', 'traffic', 'O_SurfaceTemperatureFahrenheit', 'O_CloudCoveragePercent',
                        'O_WindSpeedMph', 'O_PrecipitationPreviousHourInches', 'O_SnowfallInches', 'D_SurfaceTemperatureFahrenheit', 
                        'D_CloudCoveragePercent', 'D_WindSpeedMph', 'D_PrecipitationPreviousHourInches', 'D_SnowfallInches', 'DATE_ONLY', 
                        'MONTH_ONLY', 'YEAR', 'UNIQUE_CARRIER_9E', 'UNIQUE_CARRIER_AA', 'UNIQUE_CARRIER_AS', 'UNIQUE_CARRIER_B6', 
                        'UNIQUE_CARRIER_DL', 'UNIQUE_CARRIER_EV', 'UNIQUE_CARRIER_F9', 'UNIQUE_CARRIER_MQ', 'UNIQUE_CARRIER_NK',
                        'UNIQUE_CARRIER_OH', 'UNIQUE_CARRIER_OO', 'UNIQUE_CARRIER_UA', 'UNIQUE_CARRIER_VX', 'UNIQUE_CARRIER_WN', 
                        'UNIQUE_CARRIER_YV', 'UNIQUE_CARRIER_YX', 'ORIGIN_BOS', 'ORIGIN_DEN', 'ORIGIN_DFW', 'ORIGIN_EWR', 'ORIGIN_IAH', 
                        'ORIGIN_JFK', 'ORIGIN_LGA', 'ORIGIN_ORD', 'ORIGIN_PHL', 'ORIGIN_SFO', 'DEST_BOS', 'DEST_DEN', 'DEST_DFW', 'DEST_EWR', 
                        'DEST_IAH','DEST_JFK', 'DEST_LGA', 'DEST_ORD', 'DEST_PHL', 'DEST_SFO']]
        
        print("done3")
        print(list(datadash2))
        print(datadash2.shape[0])
        L = clf_rf.predict_proba(datadash2)
        L1 = clf_rf_class.predict_proba(datadash2)
        hoverData = []
        
        print("done4")
        df = pd.DataFrame(L)
        print(list(df))
        
        L = round((df[1]*100),2).tolist()
        L1 = pd.DataFrame(L1)
        l0 = round((L1[0]),2)
        l1 = round((L1[1]),2)
        l2 = round((L1[2]),2)
        l3  = round((L1[3]),2)
        print("done5")
        hoverList =[]
        if(radio=='S'):
            for j in range(len(L)):
                stra = "Cancellation Percentage :"+str(round(L[j],2))
                hoverList.append(stra)
        elif(radio=='CARRIER'):
            for j in range(len(L)):
                k = l0[j]*L[j]
                L[j] = L[j]-k
                stra = "Cancellation Percentage :"+str(round(L[j],2))+"<br>"+" Percentage decrement of "+str(k)
                hoverList.append(stra)
        elif(radio=='WEATHER'):
            for j in range(len(L)):
                k = l1[j]*L[j]
                L[j] = L[j]-k
                stra = "Cancellation Percentage :"+str(round(L[j],2))+"<br>"+" Percentage decrement of "+str(k)
                hoverList.append(stra)
        elif(radio=='NAS'):
            for j in range(len(L)):
                k = l2[j]*L[j]
                L[j] = L[j]-k
                stra = "Cancellation Percentage :"+str(round(L[j],2))+"<br>"+" Percentage decrement of "+str(k)
                hoverList.append(stra)
        elif(radio=='SECURITY'):
            for j in range(len(L)):
                k = l3[j]*L[j]
                L[j] = L[j]-k
                stra = "Cancellation Percentage :"+str(round(L[j],2))+"<br>"+" Percentage decrement of "+str(k)
                hoverList.append(stra)
        
        depHour = datadash3['FLL_DATE'].tolist()
        depD = L

        lists = sorted(zip( depHour,depD,hoverList))
        new_x, new_y, hoverData = list(zip(*lists))
        traces.append({'x':new_x, 'y':new_y, 'name': i,'text' : hoverData,'hoverinfo' : 'text'})
        print("done6")
    fig = {
		'data': traces,
		'layout': {'title':stock_ticker,
             'xaxis' : {'title':'Date'},
                'yaxis' : {'title':'Cancellation Probability(in %)'},}
	}
    return fig



if __name__ == '__main__':
    app.run_server()