# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 21:15:31 2019

@author: bsoni
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import plotly.figure_factory as ff

#uncomment the below lines after adding the index and app page
#the code will not run without adding an index and app page
#to make the code working make a sub-directory names apps and put all the apps in that directory
#and in the original directory put two python codes app and index.
#refer to readme for more details

#from apps import app1,app2,app3,app4,app5,app6
#from app import app

#read the data fromcsv
data = pd.read_csv("us-bts-parsed-data.csv")

#create of list of unique carriers and delay from dataframe
#It will help us in future
available_indicator_airline = data['carrier'].unique()  
delay = [15,45,60]

#below is app layout for our app
#it is designed in python with HTML on top
#make sure that every id you give to each element should not be repeated.
layout = html.Div([
    html.Div([
            html.Div([
    
    html.Div([dcc.Link('OTP dashboard: by airport', href='/page-1')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
    html.Br(),
    html.Div([dcc.Link('OTP Dashboard: by time of the day', href='/page-2')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
    html.Br(),
    html.Div([dcc.Link('Taxi Out dashboard: Mean taxi-out time by airport', href='/page-3')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
    html.Br(),
    html.Div([dcc.Link('Taxi Out dashboard: Taxi out time distributions', href='/page-4')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
    html.Br(),
    html.Div([dcc.Link('Taxi Out dashboard: By time of day', href='/page-5')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
    html.Br(),
    html.Div([dcc.Link('Delay reason dashboard', href='/page-6')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
    html.Br(),
    html.Div([dcc.Link('Block time dashboard', href='/apps/app7')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
  html.Br(),
 html.Div([dcc.Link('Delay trends', href='/apps/app8')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
html.Br(),
 html.Div([dcc.Link('Cancellation trends', href='/apps/app9')],style={'padding': '6px 8px 6px 16px',
  'text-decoration': 'none',
  'font-size': '12px',
  'color': '#818181',
  'display':'block',}),
],
style={

  'height': '100%',   'width': '200px', 
  'position': 'fixed',
  'z-index': '1',   'top': '0',
  'left': '0',
  'background-color': '#111',
  'overflow-x': 'hidden',
  'padding-top': '20px',
}),


        html.Div([html.P('Select a carrier :'),dcc.Dropdown(
                    id='airline',
                    options=[{'label': i, 'value': i} for i in available_indicator_airline],
                    value=''
            ),
            
        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),


    html.Br(),
    html.Br(),


        html.Div([html.P('Select Origin state :'),dcc.Dropdown(
                id='origin_airport',
            ),

        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),
        
             html.Br(),
    html.Br(),




         html.Div([html.P('Select Destination state :'),dcc.Dropdown(
                id='destination_airport',
                
            ),

        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),
            
            html.Br(),
    html.Br(),
       
         html.Div([html.P('Select Flight number :'),dcc.Dropdown(
                id='flight_no',
                
            ),

        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),

            
             html.Br(),
    html.Br(), 
    html.Br(),
    html.Br(),
         html.Div([dcc.Graph(
              id='block_time_graph',     
        )
                    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 50','margin-left': '220px'}),



   


    ]),

])

#Now we are starting app callbacks
# each callback calls a function
#the function name should be unique within the app.
#the called function takes the input in same order as written in callback
#below callback return options for origin airport for selected airline
@app.callback(
                dash.dependencies.Output('origin_airport', 'options'),
                [dash.dependencies.Input('airline', 'value')]
)
def set_origin_options(selected_airline):
                new_data = data[data['carrier'] == selected_airline]
                return [{'label' : i, 'value' : i} for i in new_data['origin'].unique()]

#This callback is to return selected airline and origin airport for available options for destination airport
@app.callback(
                dash.dependencies.Output('destination_airport', 'options'),
                [dash.dependencies.Input('airline', 'value'),
                dash.dependencies.Input('origin_airport', 'value')]
)
def set_destination_options(selected_airline, origin_airport):
                new_data = data[data['carrier'] == selected_airline]
                new_data = new_data[new_data['origin'] == origin_airport]
                return [{'label' : i, 'value' : i} for i in new_data['dest'].unique()]

#This callback is to return selected airline, origin airport and destination airport for available options for flight number
@app.callback(
                dash.dependencies.Output('flight_no', 'options'),
                [dash.dependencies.Input('airline', 'value'),
                dash.dependencies.Input('origin_airport', 'value'),
                dash.dependencies.Input('destination_airport', 'value')]
)
def set_flightno_options(selected_airline, origin_airport, destination_airport):
                new_data = data[data['carrier'] == selected_airline]
                new_data = new_data[new_data['origin'] == origin_airport]
                new_data = new_data[new_data['dest'] == destination_airport]
                return [{'label' : i, 'value' : i} for i in new_data['flight_number'].unique()]

#This callback is to return graph figure (a probability distribution output of the given flight number's block time) based on the selected flight number 
@app.callback(
                dash.dependencies.Output('block_time_graph', 'figure'),
                [dash.dependencies.Input('origin_airport', 'value'),
                dash.dependencies.Input('destination_airport', 'value'),
                dash.dependencies.Input('flight_no', 'value')]  
)
def update_graph(orgairport,destairport,flightno):
                #We scrutinize the data according to selected values
                dxc = data[data['origin']==orgairport]
                dxc = dxc[dxc['dest']==destairport]
                filtered_data = dxc[dxc['flight_number'] == flightno]
                #we drop the na values so that they won'bother us in futuret
                filtered_data.dropna(axis=0, subset=['actual_block_time'], inplace=True)
                
                
                hist_data = [filtered_data['actual_block_time']]
                constantData = filtered_data['scheduled_block_time'].values.tolist()
                hist_data = [filtered_data['actual_block_time']]
                group_labels = ['Percentage Distribution curve']
                fig = ff.create_distplot(hist_data, group_labels, show_hist=False)
                fig['layout'].update(title='Block Time Plot')
                fig['layout'].update(xaxis={'title':'Time(in minutes)'})
                fig.layout.update(annotations=[
                                     dict(
                                          x=constantData[0],
                                          y=0,
                                          xref='x',
                                          yref='y',
                                          text='Scheduled',
                                          font=dict(color='#F00'),
                                          showarrow=True,
                                          arrowhead=7,
                                          ax=0,
                                          ay=-175,
                                          arrowcolor='#F00'
                                          )])
                return fig




if __name__ == '__main__':
    app.run_server(debug=True)