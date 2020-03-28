# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:55:21 2019

@author: bsoni
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import datetime
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime as dt
app = dash.Dash()

#Import data
data = pd.read_csv('flightsandweatherdata.csv')
#Drop unnamed columns
data.drop(['Unnamed: 0'], axis = 1, inplace = True)
#We converted FL_DATE to ease out the merging
#Now we convert it back to datetime format
#We extract the month %B converts 01->JAN
#We than convert this columns to string
#Than we extract the date from the data
data['FL_DATE'] = data['FL_DATE'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d")) 
data['FL_month'] = data['FL_DATE'].apply(lambda x: x.strftime('%B'))
data['FL_month'] = data['FL_month'].apply(lambda x: str(x))
data['FLY_date'] = data['FL_DATE'].apply(lambda x: x.date())

#Drop duplicate rows
data = data.drop_duplicates()

#Make a list of columns of database
cols = list(data)

#make a list of unique airlines in data. It will update the dropdown menu later.
available_indicator_airline = data['UNIQUE_CARRIER'].unique()  

#below is app layout for our app
#it is designed in python with HTML on top
#make sure that every id you give to each element should not be repeated.

app.layout = html.Div([
				html.H1('Cancellation Dashboard'),
				dcc.Markdown(''' --- '''),
                html.Div([html.H3('Enter a carrier code:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-3',
						  options=[{'label': i, 'value': i} for i in available_indicator_airline],
						  value='',
						  style={'fontSize': 15, 'width': 150},
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
                html.Div([html.H3('Enter origin city:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-2',
						   style={'fontSize': 15, 'width': 150},
                           
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
                html.Div([html.H3('Enter destination city:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-1',
						  style={'fontSize': 15, 'width': 150},
                          
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
                
				html.Div([html.H3('Enter a flight number:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol',
						   # value = ['SPY'], 
						  multi = True,
						  style={'fontSize': 15, 'width': 300},
                          
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
                
				html.Div([html.H3('Enter start / end date:'),
					dcc.DatePickerRange(id='my_date_picker',
										min_date_allowed = dt(2016,1,1),
										max_date_allowed = dt(2016,12,28),
										start_date = dt(2016, 1,2),
										end_date = dt(2016, 12,25),
                                         display_format='MMM Do, YY'
					)

				], style={'display':'inline-block'}), 
                    
				html.Div([
					html.Button(id='submit-button',
								n_clicks = 0,
								children = 'Submit',
								style = {'fontSize': 24, 'marginLeft': '30px'}

					)

				], style={'display': 'inline-block'}),
				
				dcc.Markdown(''' --- '''), 
				
				
    
                dcc.Graph(id='my_graph_1',
							figure={'data':[
								{'x':[1,2], 'y':[3,1]}

							], 'layout':go.Layout(title='Relative Stock Returns Comparison', 
                                                            yaxis = {'title':'Returns', 'tickformat':".2%"}
                                         )}
				),	
html.Div(id = 'textonly',children='Cancellation Distribution', style={
        'textAlign': 'center',
        
    }),
					

])
#Now we are starting app callbacks
# each callback calls a function
#the function name should be unique within the app.
#the called function takes the input in same order as written in callback
#below callback returns the origin options for selected airline
@app.callback(
                Output('my_ticker_symbol-2', 'options'),
                [Input('my_ticker_symbol-3', 'value')]
)
def set_origin_options(selected_airline):
                new_data = data[data['UNIQUE_CARRIER'] == selected_airline]
                return [{'label' : i, 'value' : i} for i in new_data['ORIGIN'].unique()]

#This callback is to return selected airline and origin airport for available options for destination airport
@app.callback(
                Output('my_ticker_symbol-1', 'options'),
                [Input('my_ticker_symbol-3', 'value'),
                Input('my_ticker_symbol-2', 'value')]
)
def set_destination_options(selected_airline, origin_airport):
                new_data = data[data['UNIQUE_CARRIER'] == selected_airline]
                new_data = new_data[new_data['ORIGIN'] == origin_airport]
                return [{'label' : i, 'value' : i} for i in new_data['DEST'].unique()]
#This callback is to return selected airline, origin airport and destination airport for available options for flight number
@app.callback(
                dash.dependencies.Output('my_ticker_symbol', 'options'),
                [dash.dependencies.Input('my_ticker_symbol-3', 'value'),
                dash.dependencies.Input('my_ticker_symbol-2', 'value'),
                dash.dependencies.Input('my_ticker_symbol-1', 'value')]
)
def set_flightno_options(selected_airline, origin_airport, destination_airport):
                new_data = data[data['UNIQUE_CARRIER'] == selected_airline]
                new_data = new_data[new_data['ORIGIN'] == origin_airport]
                new_data = new_data[new_data['DEST'] == destination_airport]
                return [{'label' : i, 'value' : i} for i in new_data['FL_NUM'].unique()]
#This callback updates the graph on selected data
#for updating the graph we always need two lists i.e. one for each x and y axis
@app.callback(Output('my_graph_1', 'figure'),
				[Input('submit-button', 'n_clicks'),
                 Input('my_ticker_symbol-3', 'value'),
                 Input('my_ticker_symbol-2', 'value'),
                 Input('my_ticker_symbol-1', 'value'),
				Input('my_ticker_symbol', 'value'),  
				Input('my_date_picker', 'start_date'),
                Input('my_date_picker', 'end_date')])
def update_graph_1(n_clicks,airline,origin,dest,stock_ticker,startdate,enddate):
    
    #The date picker sends the date time format in Y/M/D H:M:S format
    #But we need only date so we take first 10 characters of string
    startdate = startdate[:10]
    enddate = enddate[:10]
    
    #We convert the string tp date time format
    start_date = datetime.datetime.strptime(startdate,'%Y-%m-%d')
    end_date = datetime.datetime.strptime(enddate,'%Y-%m-%d')
    
    #We select only that dataframe that falls between the selected date range 
    filtered_df = data[data.FL_DATE.between(
        datetime.datetime.strftime(start_date, "%Y-%m-%d"),
        datetime.datetime.strftime(end_date, "%Y-%m-%d")
    )]
    
    #Now we scrutiny the data according to selected variables
    filtered_df1 = filtered_df[filtered_df['UNIQUE_CARRIER']==airline]
    filtered_df1 = filtered_df1[filtered_df1['ORIGIN']==origin]
    filtered_df1 = filtered_df1[filtered_df1['DEST']==dest]
    
    traces = []
    for i in stock_ticker:
        #For each airline now we carry out the desired operations to get x and y lists.
        cCount = []
        df = filtered_df1[filtered_df1['FL_NUM']==i]
        GGHH = df.CANCELLATION_CODE.unique()
        codess= []
        jjjj = []
        for o in GGHH:
            dffff = df[df['CANCELLATION_CODE']==o]
            KKKKK = dffff.shape[0]
            codess.append(KKKKK)
            jjjj.append(i)
        #Here we need this code only to keep graph working
        #We select the months and see how much flights are cancelled in each month in scrutinized dataframe
        months = df.FL_month.unique()
        for j in months:
            df1 = df[df['FL_month']==j]
            cCount.append(df1[df1.CANCELLED == 1].shape[0])
        
        #This code below enables us to see multiple selection output
        traces.append({'x':months, 'y':cCount, 'name': i})
        
    fig1 = {
		'data': traces,
		'layout': {'title':stock_ticker}
	}
    return fig1

@app.callback(Output('textonly', 'children'),
				[Input('submit-button', 'n_clicks'),
                 Input('my_ticker_symbol-3', 'value'),
                 Input('my_ticker_symbol-2', 'value'),
                 Input('my_ticker_symbol-1', 'value'),
				Input('my_ticker_symbol', 'value'),  
				Input('my_date_picker', 'start_date'),
                Input('my_date_picker', 'end_date')])
def update_carrier_1(n_clicks,airline,origin,dest,stock_ticker,startdate,enddate):
    
    startdate = startdate[:10]
    enddate = enddate[:10]
 
    start_date = datetime.datetime.strptime(startdate,'%Y-%m-%d')
    end_date = datetime.datetime.strptime(enddate,'%Y-%m-%d')
    filtered_df = data[data.FL_DATE.between(
        datetime.datetime.strftime(start_date, "%Y-%m-%d"),
        datetime.datetime.strftime(end_date, "%Y-%m-%d")
    )]
    filtered_df1 = filtered_df[filtered_df['UNIQUE_CARRIER']==airline]
    filtered_df1 = filtered_df1[filtered_df1['ORIGIN']==origin]
    filtered_df1 = filtered_df1[filtered_df1['DEST']==dest]
    
    traces = []
    #Here we need this code to keep div text working
    #It calculates all the cancellation reasons count from the selected data
    #Than it appends that string to the respective flight number
    #and than return a division consisting of a header with text as flight number->cancellation distrbution    
    for i in stock_ticker:
        traces1 = []
        df = filtered_df1[filtered_df1['FL_NUM']==i]
        GGHH = df.CANCELLATION_CODE.unique()
        GGHH = [x for x in GGHH if str(x) != 'nan']
        codess= []
        jjjj = []
        stri = ' '
        shape = df[df['CANCELLED']==1].shape[0]
        if(shape!=0):
            for k in GGHH:
                dffff = df[df['CANCELLATION_CODE']==k]
                KKKKK = ((dffff.shape[0])/shape)*100
                KKKKK = round(KKKKK,2)
                codess.append(KKKKK)
                jjjj.append(k)
        
        months = df.FL_month.unique()
        count=0
        for j in GGHH:
            stri=stri+str(jjjj[count]+":"+str(codess[count]))+"%"
            count=count+1
        traces.append("Flight Number-"+str(i)+" "+" Cancellation Division"+stri)
        
        for i in traces:
            traces1.append(html.Div(html.H1(i)))    
    return html.Div(traces1)


if __name__ == '__main__':
    app.run_server()