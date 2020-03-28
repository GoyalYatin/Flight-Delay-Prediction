# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import datetime
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime as dt
import pickle

#initialize dash
app = dash.Dash()

#import the flights and weather merged data for the 3 years
#we have to train on 2016-17 while test on 2018
traindata = pd.read_csv('weathertrafficandflights-2016.csv')
testdata = pd.read_csv('weathertrafficandflights-2017.csv')
testdata1 = pd.read_csv('weathertrafficandflights-2018.csv')

#append a column indicating year in all 3 of dataframes
traindata['YEAR']=2016
testdata['YEAR']=2017
testdata1['YEAR']=2018

#set departure time of 2016 as 0 because we have to use that column for 2018 testdata 
traindata['CRS_DEP_TIME'] = 0
#take only those columns which are useful for predictions
#sometimes unwanted columns are added while merging/separating the data
#to counter that we should mention columns we are going to take
traindata = traindata[['DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','FL_NUM','ORIGIN','DEST','Hour',
'CANCELLED','CANCELLATION_CODE','DISTANCE','traffic','O_SurfaceTemperatureFahrenheit',
'O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent',
'D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches','YEAR','CRS_DEP_TIME']]

#set departure time of 2017 as 0 because we have to use that column for 2018 testdata 
testdata['CRS_DEP_TIME'] = 0
#take only those columns which are useful for predictions
#sometimes unwanted columns are added while merging/separating the data
#to counter that we should mention columns we are going to take
testdata= testdata[['DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','FL_NUM','ORIGIN','DEST','Hour',
'CANCELLED','CANCELLATION_CODE','DISTANCE','traffic','O_SurfaceTemperatureFahrenheit',
'O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent',
'D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches','YEAR','CRS_DEP_TIME']]

#take only those columns which are useful for predictions
#sometimes unwanted columns are added while merging/separating the data
#to counter that we should mention columns we are going to take
#Notice that crs_dep_time has values here that will be required in visualization
testdata1= testdata1[['DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','FL_NUM','ORIGIN','DEST','Hour',
'CANCELLED','CANCELLATION_CODE','DISTANCE','traffic','O_SurfaceTemperatureFahrenheit',
'O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent',
'D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches','YEAR','CRS_DEP_TIME']]

#aggregate all the data
aggr_dataset = [traindata,testdata,testdata1]
data = pd.concat(aggr_dataset)

#shuffle data
#It's been observed that accuracy increases somewhat by shuffling the data
data = data.sample(frac=1).reset_index(drop=True)

#We converted the date to int format while merging
#Now is the time to convert it back to datetime format
#That we will extract date and month from that column
#date and month can also be useful features for predictions
data['FLL_DATE'] = data['FL_DATE'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d")) 
data['DATE_ONLY']=data['FLL_DATE'].apply(lambda x: x.strftime('%d'))
data['MONTH_ONLY']=data['FLL_DATE'].apply(lambda x: x.strftime('%m'))

#copy the data to datadash so that we can use it further
#!!!Always use copy function to  copy the data
#!!!Simply using datadash=data can be harmful, as if you change something in datadash it will do same changes in data 
datadash = data.copy()

#!take only those column which are needed to proceed further
data = data[['DAY_OF_WEEK',
           	'UNIQUE_CARRIER','CANCELLED',	'FL_NUM',	'ORIGIN',	'DEST',	
           'Hour',	'DISTANCE',	
           'traffic',	'O_SurfaceTemperatureFahrenheit',	'O_CloudCoveragePercent',	
           'O_WindSpeedMph',	'O_PrecipitationPreviousHourInches',	'O_SnowfallInches',	
           'D_SurfaceTemperatureFahrenheit',	'D_CloudCoveragePercent','D_WindSpeedMph',	
           'D_PrecipitationPreviousHourInches',	'D_SnowfallInches','DATE_ONLY','MONTH_ONLY','YEAR','CRS_DEP_TIME']]


#!!!this is the prediction phase area
#!!!you should write predictions code here
#for better results use the following procedure
#1.Use one hot encoding on unique_carrier, origin and destination
#2.apply SMOTE/ADASYN for balncing the data
#3.scale the data using standard scalar
#4.remember to separate target values after applying smote
#5.train your model

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
#declare a file name in which you want to save your model
filename = 'cancellation_model.sav'
#below is the code for saving the file
#!!!replace clf_rf with your model name
#pickle.dump(clf_rf, open(filename, 'wb'))

#load the model from the file
#remember the model and code should be in same folder
clf_rf = pickle.load(open(filename, 'rb'))


#the below code is for the prediction of delay reason class
#again copy data from datadash
datadash_class = datadash.copy()

#we need only cancelled data for prediction of cancellation reason
datadash_class = datadash_class[datadash_class['CANCELLED']==1]

#take only that data that will be required for reason prediction
datadash_class = datadash_class[['DAY_OF_WEEK',
           	'UNIQUE_CARRIER','CANCELLATION_CODE',	'FL_NUM',	'ORIGIN',	'DEST',	
           'Hour',	'DISTANCE',	
           'traffic',	'O_SurfaceTemperatureFahrenheit',	'O_CloudCoveragePercent',	
           'O_WindSpeedMph',	'O_PrecipitationPreviousHourInches',	'O_SnowfallInches',	
           'D_SurfaceTemperatureFahrenheit',	'D_CloudCoveragePercent','D_WindSpeedMph',	
           'D_PrecipitationPreviousHourInches',	'D_SnowfallInches','DATE_ONLY','MONTH_ONLY','YEAR']]

#!!!this is the reason prediction phase area
#!!!you should write predictions code here
#for better results use the following procedure
#1.Use one hot encoding on unique_carrier, origin and destination
#2.apply SMOTE/ADASYN for balncing the data
#3.scale the data using standard scalar
#4.remember to separate target values after applying smote
#5.train your model

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
#declare a file name in which you want to save your model
filename = 'cancellation_reason_model.sav'

#below is the code for saving the file
#!!!replace clf_rf_class with your model name
#pickle.dump(clf_rf_class, open(filename, 'wb'))

#load the model from the file
#remember the model and code should be in same folder
clf_rf_class = pickle.load(open(filename, 'rb'))



#below is app layout for our app
#it is designed in python with HTML on top
#make sure that every id you give to each element should not be repeated.
#first of all we have to make the data suitable for app/ mould the data in the form you want to display with
datadash1 = datadash.drop(columns=['CANCELLED','CRS_DEP_TIME'])
datadash1 = datadash1[datadash1['YEAR']==2018]
datadashdash = datadash1.copy()


#make a list of unique airlines in data. It will update the dropdown menu later.
#the below code is for making the data suitable for visualization purpose
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

#below is app layout for our app
#it is designed in python with HTML on top
#make sure that every id you give to each element should not be repeated.
app.layout = html.Div([
        
				html.Div([html.H3('Cancellation Dashboard', style={'fontSize': 52,'margin-left': '210px'}),

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '150%'}),
				dcc.Markdown(''' --- '''),
                html.Div([html.H3('Enter a carrier code:', style={'margin-left': '149px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-03',
						  options=[{'label': i, 'value': i} for i in available_indicator_airline],
                          value='',
                          
						  style={'fontSize': 15, 'width': 300,'margin-left': '75px'},
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-left': '220px'}),
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
                
				html.Div([html.H3('Enter a flight number:', style={'margin-left': '149px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-00',
						   # value = ['SPY'], 
						  multi = True,
						  style={'fontSize': 15, 'width': 300,'margin-left': '75px'},
                          
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
                
				html.Div([html.H3('Enter start / end date:'),
					dcc.DatePickerRange(id='my_date_picker-00',
										min_date_allowed = dt(2018,1,1),
										max_date_allowed = dt(2018,12,12),
										start_date = dt(2018, 1,2),
										end_date = dt(2018, 4,28),
                                         display_format='MMM Do, YY',
                                         
					)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '28%','margin-left': '370px'}), 
                    
				html.Div([
					html.Button(id='submit-button-00',
								n_clicks = 0,
								children = 'Submit',
								style = {'fontSize': 24, 'marginLeft': '30px','margin-left': '75px'}

					)

				], style={'display': 'None'}),
				dcc.Markdown(''' --- '''), 
				
				html.Div([dcc.Graph(id='my_graph-00',
							figure={'layout':go.Layout(title='daily Cancellations will be shown here', 
                               
                                         )}
				), ],style={'width': '85%', 'display': 'inline-block', 'padding': '0 20','margin-left': '220px'}),

    
               
					

])
#Now we are starting app callbacks
# each callback calls a function
#the function name should be unique within the app.
#the called function takes the input in same order as written in callback
#below callback returns the origin options for selected airline
@app.callback(
                Output('my_ticker_symbol-02', 'options'),
                [Input('my_ticker_symbol-03', 'value')]
)
def set_origin_options(selected_airline):
                new_data = datadashdash[datadashdash['UNIQUE_CARRIER'] == selected_airline]
                return [{'label' : i, 'value' : i} for i in new_data['ORIGIN'].unique()]

#This callback is to return options for destination airport for selected airline and origin airport 
@app.callback(
                Output('my_ticker_symbol-01', 'options'),
                [Input('my_ticker_symbol-03', 'value'),
                Input('my_ticker_symbol-02', 'value')]
)
def set_destination_options(selected_airline, origin_airport):
                new_data = datadashdash[datadashdash['UNIQUE_CARRIER'] == selected_airline]
                new_data = new_data[new_data['ORIGIN'] == origin_airport]
                return [{'label' : i, 'value' : i} for i in new_data['DEST'].unique()]
#This callback is to return options for flight number for selected airline, origin and destination airport 
@app.callback(
                dash.dependencies.Output('my_ticker_symbol-00', 'options'),
                [dash.dependencies.Input('my_ticker_symbol-03', 'value'),
                dash.dependencies.Input('my_ticker_symbol-02', 'value'),
                dash.dependencies.Input('my_ticker_symbol-01', 'value')]
)
def set_flightno_options(selected_airline, origin_airport, destination_airport):
                new_data = datadashdash[datadashdash['UNIQUE_CARRIER'] == selected_airline]
                new_data = new_data[new_data['ORIGIN'] == origin_airport]
                new_data = new_data[new_data['DEST'] == destination_airport]
                new_data = new_data[new_data['YEAR']==2018]
                print(new_data['FL_NUM'].unique())
                return [{'label' : i, 'value' : i} for i in new_data['FL_NUM'].unique()]
#This callback updates the graph on selected data
#for updating the graph we always need two lists i.e. one for each x and y axis
#The graph takes every selected entity as input
@app.callback(Output('my_graph-00', 'figure'),
				[Input('submit-button-00', 'n_clicks'),
                 Input('my_ticker_symbol-03', 'value'),
                 Input('my_ticker_symbol-02', 'value'),
                 Input('my_ticker_symbol-01', 'value'),
				Input('my_ticker_symbol-00', 'value'),  
				Input('my_date_picker-00', 'start_date'),
                Input('my_date_picker-00', 'end_date')])
def update_graph(n_clicks,airline,origin,dest,stock_ticker,startdate,enddate):
    
    #The date picker sends the date time format in Y/M/D H:M:S format
    #But we need only date so we take first 10 characters of string
    startdate = startdate[:10]
    enddate = enddate[:10]
    #We convert the string to date time format
    start_date = datetime.datetime.strptime(startdate,'%Y-%m-%d')
    end_date = datetime.datetime.strptime(enddate,'%Y-%m-%d')
    #We select only that dataframe that falls between the selected date range 
    #crs_filtered_df is separatly made for diplaying crs departure time
    filtered_df = datadash1[datadash1.FLL_DATE.between(
        datetime.datetime.strftime(start_date, "%Y-%m-%d"),
        datetime.datetime.strftime(end_date, "%Y-%m-%d")
    )]
    crs_filtered_df = crstestdata[crstestdata.FLL_DATE.between(
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
    crs_filtered_df_1 = crs_filtered_df[crs_filtered_df['UNIQUE_CARRIER']==airline]
    crs_filtered_df_1 = crs_filtered_df_1[crs_filtered_df_1['ORIGIN']==origin]
    crs_filtered_df_1 = crs_filtered_df_1[crs_filtered_df_1['DEST']==dest]
    #This code helps us to keep the graph updating
    #For each flight number selected this code finds the probability of flight to be cancelled from test data using the loaded model 
    #it provides y list as that cancellation probability and  x list as date of that probability
    traces = []
    for i in stock_ticker:
        df = filtered_df1[filtered_df1['FL_NUM']==i]
        crs_df = crs_filtered_df_1[crs_filtered_df_1['FL_NUM']==i]
        #below dataframes will be use for prediction and for finding dates
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
        
        #probabbility prediction of testdata 
        L = clf_rf.predict_proba(datadash2)
        #probability prediction of class data
        L1 = clf_rf_class.predict_proba(datadash2)
        #hover data is separate list for additional information for business use and further insights
        hoverData = []
        
        #converting ndarray to dataframe/list for futher processing
        df = pd.DataFrame(L)
        depD = df[1].tolist()
        L1 = pd.DataFrame(L1)
        #these are the indexes in which cancellation probability is greater than 0.5, means this flight is going to be cancelled
        #these indexes will be helpfull in adding arrows to show the date and dep time of flight going to be cancelled
        indexes = [m for m in range(len(depD)) if depD[m] > 0.5]
        
        #separating list for probability of each reason
        l0 = round((L1[0]*100),2)
        l1 = round((L1[1]*100),2)
        l2 = round((L1[2]*100),2)
        l3  = round((L1[3]*100),2)
         
        #the code below appends the required information to hoverlist
        for j in range(len(depD)):
            if(depD[j]>=0.5):
                stra = "Cancellation probability: "+str(depD[j])+"<br>"+"Airline/Carrier Cancellation Probability: "+str(l0[j])+"%"+"<br>"+"Weather Cancellation Probability: "+str(l1[j])+"%"+"<br>"+"NAS Cancellation Probability: "+str(l2[j])+"%"+"<br>"+"Security Cancellation Probability: "+str(l3[j])+"%"
                hoverData.append(stra)
            else:
                stra = "Cancellation probability: "+str(depD[j])
                hoverData.append(stra)
                
        #these will form the X axis of our graph
        depHour = datadash3['FLL_DATE'].tolist()
        
        #below is the code for adding arrows with time information to the graph whose cancellation probability is greater than 0.5
        #we are using annotations for this purpose
        crs_Y = [depD[x] for x in indexes]
        crs_X = [depHour[x] for x in indexes]
        
        text = crs_df['CRS_DEP_TIME'].tolist()
        crs_text = [text[x] for x in indexes]
        annotations = []
        for o in range(len(crs_X)):
            my_dict = dict(x=crs_X[o],
            y=crs_Y[o],
            xref='x',
            yref='y',
            text = str(crs_text[o]),
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8)
            annotations.append(my_dict)
        lists = sorted(zip( depHour,depD,hoverData))
        new_x, new_y, hoverList = list(zip(*lists))
        traces.append({'x':new_x, 'y':new_y, 'name': i,'text' : hoverList,'hoverinfo' : 'text'})
        print("done6")
    fig = {
		'data': traces,
		'layout': {'title':stock_ticker,'annotations':annotations}
	}
    return fig



if __name__ == '__main__':
    app.run_server()