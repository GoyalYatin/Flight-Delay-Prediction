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
from app import app
from copy import deepcopy


data = pd.read_csv('Data/2017flightwithtraffic.csv')            
dataca = data[data['CANCELLED']==1]
available_indicator_airline = data['OP_UNIQUE_CARRIER'].unique()  


layout = html.Div([
        html.Div( 
                 [html.H3([html.Span('EDA-CANCELLATION DASHBOARD',className="main-heading-primary-in"),
                           html.Span('CANCELLATIONS VS TRAFFIC DASHBOARD',className="main-heading-secondary")]
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
                ],className='hello',href='/EDAapps/edacancellationindex')
            ],style={'float':'top','margin-left':'95%','height':'0px','width':'0px','background-color':'#ffffff'}) ,
    
				html.Div([
                html.Div([html.H3('Carrier code:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-ce53',
						  options=[{'label': i, 'value': i} for i in available_indicator_airline],
						  # value = ['SPY'], 
                          value='',
                          
						  style={'fontSize': 15, 'width': 150},
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-left':'120px','margin-top':'-30px'}),
                html.Div([html.H3('Origin Airport:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-ce52',
						   style={'fontSize': 15, 'width': 150},
                           
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-top':'-30px'}),
                html.Div([html.H3('Destination Airport:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-ce51',
						  style={'fontSize': 15, 'width': 150},
                          
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-top':'-30px'}),
                
				html.Div([html.H3('Flight number:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-ce5',
						   # value = ['SPY'], 
						  multi = True,
						  style={'fontSize': 15, 'width': 300},
                          
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-left':'120px','margin-top':'-10px'}),
                
				
                    
				html.Div([
					html.Button(id='submit-button-ce5',
								n_clicks = 0,
								children = 'Submit',
                                
								style = {'display':'none','fontSize': 24, 'marginLeft': '30px'}

					)

				], style={'display': 'none'}),
				# dcc.Markdown(''' --- '''),

				# Total Return Charts section
				# html.H1('Total Return Charts'),
					
				dcc.Markdown(''' --- '''), 
				
                html.Div([
				dcc.Graph(id='my_graph-ce5',
							
				),],style={'width': '85%', 'display': 'inline-block', 'padding': '0 20','margin-left': '220px','margin-top': '-35px'}),
    
],style={'margin-top':'250px'}),
               
					

])
@app.callback(
                Output('my_ticker_symbol-ce52', 'options'),
                [Input('my_ticker_symbol-ce53', 'value')]
)
def set_origin_options_ce5(selected_airline):
                new_data = data[data['OP_UNIQUE_CARRIER'] == selected_airline]
                return [{'label' : i, 'value' : i} for i in new_data['ORIGIN'].unique()]

#This callback is to return selected airline and origin airport for available options for destination airport
@app.callback(
                Output('my_ticker_symbol-ce51', 'options'),
                [Input('my_ticker_symbol-ce53', 'value'),
                Input('my_ticker_symbol-ce52', 'value')]
)
def set_destination_options_ce5(selected_airline, origin_airport):
                new_data = data[data['OP_UNIQUE_CARRIER'] == selected_airline]
                new_data = new_data[new_data['ORIGIN'] == origin_airport]
                return [{'label' : i, 'value' : i} for i in new_data['DEST'].unique()]
#This callback is to return selected airline, origin airport and destination airport for available options for flight number
@app.callback(
                dash.dependencies.Output('my_ticker_symbol-ce5', 'options'),
                [dash.dependencies.Input('my_ticker_symbol-ce53', 'value'),
                dash.dependencies.Input('my_ticker_symbol-ce52', 'value'),
                dash.dependencies.Input('my_ticker_symbol-ce51', 'value')]
)
def set_flightno_options_ce5(selected_airline, origin_airport, destination_airport):
                new_data = data[data['OP_UNIQUE_CARRIER'] == selected_airline]
                new_data = new_data[new_data['ORIGIN'] == origin_airport]
                new_data = new_data[new_data['DEST'] == destination_airport]
                return [{'label' : i, 'value' : i} for i in new_data['OP_CARRIER_FL_NUM'].unique()]

@app.callback(Output('my_graph-ce5', 'figure'),
				[Input('submit-button-ce5', 'n_clicks'),
                 Input('my_ticker_symbol-ce53', 'value'),
                 Input('my_ticker_symbol-ce52', 'value'),
                 Input('my_ticker_symbol-ce51', 'value'),
				Input('my_ticker_symbol-ce5', 'value')])
def update_graph_ce5(n_clicks,airline,origin,dest,stock_ticker):
    
    print("looooooooooooooooooooooooool")
    
    filtered_df = data
    filtered_df1 = filtered_df[filtered_df['OP_UNIQUE_CARRIER']==airline]
    filtered_df1 = filtered_df1[filtered_df1['ORIGIN']==origin]
    filtered_df1 = filtered_df1[filtered_df1['DEST']==dest]
    print(stock_ticker)
    print(type(stock_ticker))
    for i in stock_ticker:
        print(type(i))
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
        print(depC)
        print(deptraffic)
        zz = deptraffic.index(8)
        del depC[zz]
        del deptraffic[zz]
        depC = depC[:-1]
        deptraffic = deptraffic[:-1]
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