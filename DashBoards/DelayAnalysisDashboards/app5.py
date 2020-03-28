# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:04:52 2019

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
df = pd.read_csv("us-bts-parsed-data.csv")

#for making delay classes intialize columns as 0
df['15_flag']=0
#the below code updates this column to 1 if dep delay is greater than 15
df['15_flag'] = np.where(df.dep_delay >15, df['15_flag'],1)
#for making delay classes intialize columns as 0
df['45_flag']=0
#the below code updates this column to 1 if dep delay is greater than 15
df['45_flag'] = np.where(df.dep_delay >45, df['45_flag'],1)
#for making delay classes intialize columns as 0
df['60_flag']=0
#the below code updates this column to 1 if dep delay is greater than 15
df['60_flag'] = np.where(df.dep_delay >60, df['60_flag'],1)

#drop unnamed columns
df.drop(['Unnamed: 0'], axis=1,inplace =True)

#make a list of crs depature time
#It may help us in future
T = df.scheduled_dep_hour.unique()
T.sort()

#create of list of unique carriers,origin  and delay from dataframe
#It will help us in future
flights = df.carrier.unique()
delay = [15,45,60]
flight_origin = df.origin.unique()
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
                    id='xaxis-column-4',
                    options=[{'label': i, 'value': i} for i in flights],
                    value='WN'
            ),
            
        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),



             html.Br(),
    html.Br(),




         html.Div([html.P('Select Origin state :'),dcc.Dropdown(
                id='zaxis-column-4',
                options=[{'label': i, 'value': i} for i in flight_origin],
                value='ATL'
            ),

        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),
       



        html.Div([dcc.Graph(
              id='heatmap-indicator-scatter-4',     
        )
                    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20','margin-left': '220px'}),

    ]),

])


#this call back updates heatmap for hour vs day mean taxi out time
@app.callback(
    dash.dependencies.Output('heatmap-indicator-scatter-4', 'figure'),
    [dash.dependencies.Input('xaxis-column-4', 'value'),
    dash.dependencies.Input('zaxis-column-4', 'value')])
    
def update_graph_1(xaxis_column, zaxis_column):
    #first we scrutinize our data
    dff = df[df['carrier'] == xaxis_column]
    dfff = dff[dff['origin'] == zaxis_column]    
    
    #create a liost for unique day and hour values
    L = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    T = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    B = []
    
    #the code below calculates the mean of taxi out time for the entities selected
    #it than appends the value to a list & than this list will be feeded to heatmap data
    for i in L:
            dftemp = dfff[dfff['day_of_week']==i]
            for j in T:
                dftemp1 = dftemp[dftemp['scheduled_dep_hour']==j]
                B.append(dftemp1['taxi_out'].mean())
    
    #it converts 1d data to 2d matrix for proper visulization  of day vs hour heatmap
    if(len(B)==168):        
        B = np.reshape(B,(7,24))
    return {'data': [
                go.Heatmap(z=B,
                   x=T,
                   y=L)
            ],
            'layout': {
                'title': 'HourWise mean departure delay',
                'xaxis' : {'title':'Hour of day'},

            }}
    
if __name__ == '__main__':
    app.run_server(debug=True)