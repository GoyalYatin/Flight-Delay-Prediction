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
#Initialize dash
app = dash.Dash()

#Import data with traffic
data = pd.read_csv('2017flightwithtraffic.csv')            

#make a list of unique airlines in data. It will update the dropdown menu later.
available_indicator_airline = data['OP_UNIQUE_CARRIER'].unique()  

#below is app layout for our app
#it is designed in python with HTML on top
#make sure that every id you give to each element should not be repeated.
app.layout = html.Div([
				html.H1('Traffic-Cancellation Dashboard'),
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
						  multi = True,
						  style={'fontSize': 15, 'width': 300},
                          
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
                
				
                    
				html.Div([
					html.Button(id='submit-button',
								n_clicks = 0,
								children = 'Submit',
								style = {'fontSize': 24, 'marginLeft': '30px'}

					)

				], style={'display': 'inline-block'}),

				dcc.Markdown(''' --- '''), 
				
				dcc.Graph(id='my_graph',
							
				),
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
                new_data = data[data['OP_UNIQUE_CARRIER'] == selected_airline]
                return [{'label' : i, 'value' : i} for i in new_data['ORIGIN'].unique()]

#This callback is to return available options for destination for selected origin and airline
@app.callback(
                Output('my_ticker_symbol-1', 'options'),
                [Input('my_ticker_symbol-3', 'value'),
                Input('my_ticker_symbol-2', 'value')]
)
def set_destination_options(selected_airline, origin_airport):
                new_data = data[data['OP_UNIQUE_CARRIER'] == selected_airline]
                new_data = new_data[new_data['ORIGIN'] == origin_airport]
                return [{'label' : i, 'value' : i} for i in new_data['DEST'].unique()]
#This callback is to return available options for flight number for selected origin,dest and airline
@app.callback(
                dash.dependencies.Output('my_ticker_symbol', 'options'),
                [dash.dependencies.Input('my_ticker_symbol-3', 'value'),
                dash.dependencies.Input('my_ticker_symbol-2', 'value'),
                dash.dependencies.Input('my_ticker_symbol-1', 'value')]
)
def set_flightno_options(selected_airline, origin_airport, destination_airport):
                new_data = data[data['OP_UNIQUE_CARRIER'] == selected_airline]
                new_data = new_data[new_data['ORIGIN'] == origin_airport]
                new_data = new_data[new_data['DEST'] == destination_airport]
                return [{'label' : i, 'value' : i} for i in new_data['OP_CARRIER_FL_NUM'].unique()]
            
#This callback updates the graph on selected data
#for updating the graph we always need two lists i.e. one for each x and y axis            
#The graph takes each selected entity as input
@app.callback(Output('my_graph', 'figure'),
				[Input('submit-button', 'n_clicks'),
                 Input('my_ticker_symbol-3', 'value'),
                 Input('my_ticker_symbol-2', 'value'),
                 Input('my_ticker_symbol-1', 'value'),
				Input('my_ticker_symbol', 'value')])
def update_graph(n_clicks,airline,origin,dest,stock_ticker):
    #Now we scrutiny the data according to selected variables
    filtered_df = data
    filtered_df1 = filtered_df[filtered_df['OP_UNIQUE_CARRIER']==airline]
    filtered_df1 = filtered_df1[filtered_df1['ORIGIN']==origin]
    filtered_df1 = filtered_df1[filtered_df1['DEST']==dest]
    #This code helps us to keep the graph updating
    #For each flight number selected this code finds the total cancellations of flights in an iteration over a traffic
    #than against each traffic it calculates total number of cancellations
    #it provides number of cancellations on y axis while traffic on x axis 
    traces = []
    for i in stock_ticker:
        df = filtered_df1[filtered_df1['OP_CARRIER_FL_NUM']==i]
        KK = df.traffic.unique()
        depC = []
        deptraffic = []
        for j in KK:
            DFF = df[df['traffic']==j]
            DD = DFF[DFF['CANCELLED']==1]
            depC.append(DD.shape[0])
            KM = DFF.traffic.unique()
            KN =KM[0]
            deptraffic.append(KN)
        lists = sorted(zip( deptraffic,depC))
        new_x, new_y = list(zip(*lists))
        traces.append({'x':new_x, 'y':new_y, 'name': i})
    
    fig = {
            
		'data': traces,
		'layout': {'title':stock_ticker,
             'xaxis' : {'title':'Traffic'},
                'yaxis' : {'title':'Total Cancellations'},}
	}
    return fig



if __name__ == '__main__':
    app.run_server()