# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:50:58 2019

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
                    id='xaxis-column-3',
                    options=[{'label': i, 'value': i} for i in flights],
                    value='WN'
            ),
            
        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),


    html.Br(),
    html.Br(),


        html.Div([html.P('Select Origin state :'),dcc.Dropdown(
                id='yaxis-column-3',
                options=[{'label': i, 'value': i} for i in flight_origin],
                value='ATL'
            ),

        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),
        
             html.Br(),
             html.Br(),




         html.Div([html.P('Select Destination state :'),dcc.Dropdown(
                id='zaxis-column-3',
                options=[{'label': i, 'value': i} for i in flight_origin],
                value='RIC'
            ),

        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),
       


                    
         html.Div([dcc.Graph(
              id='pd-indicator-scatter',     
        )
                    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20','margin-left': '220px'}),



        html.Div([dcc.Graph(
              id='pd-indicator-scatter-1',     
        )
                    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20','margin-left': '220px'}),




        
        html.Div([dcc.Graph(
              id='pd-indicator-scatter-2',     
        )
                    ], style={'width': '49%', 'display': 'none', 'padding': '0 20','margin-left': '220px'}),


    ]),

])
#Now we are starting app callbacks
# each callback calls a function
#the function name should be unique within the app.
#the called function takes the input in same order as written in callback
#below callback returns origin options for selected airline
@app.callback(
    dash.dependencies.Output('yaxis-column-3', 'options'),
    [dash.dependencies.Input('xaxis-column-3', 'value')])
def set_dep_options_1(xaxis_column):
    mydata = df
    mydata1 = mydata[mydata['carrier']==xaxis_column]
    K = mydata1.origin.unique()
    return [{'label': i, 'value': i} for i in K]

#below callback returns dest options for selected airline and origin
@app.callback(
    dash.dependencies.Output('zaxis-column-3', 'options'),
    [dash.dependencies.Input('xaxis-column-3', 'value'),
     dash.dependencies.Input('yaxis-column-3', 'value')])
def set_dep_options(xaxis_column,yaxis_column):
    mydata = df
    mydata1 = mydata[mydata['carrier']==xaxis_column]
    mydata1 = mydata1[mydata1['origin']==yaxis_column]
    K = mydata1.dest.unique()
    return [{'label': i, 'value': i} for i in K]

#below callback returns graphs by taking all the selected option as input
@app.callback(
    dash.dependencies.Output('pd-indicator-scatter', 'figure'),
    [dash.dependencies.Input('xaxis-column-3', 'value'),
    dash.dependencies.Input('yaxis-column-3', 'value'),
    dash.dependencies.Input('zaxis-column-3', 'value')])

def update_graph(xaxis_column, yaxis_column, zaxis_column):
    #first we scrutinuize the data
    dff = df[df['carrier'] == xaxis_column]
    dfff = dff[dff['origin'] == yaxis_column]    
    
    #now we take taxi-out time into account
    dfToList = list(dfff['taxi_out'])
    dfToList = list(dfToList)
    
    #the taxoi out time will be provided as hitogram data
    hist_data = [dfToList]
    group_labels = ['Percentage Distribution Curve']
    #creating percentage distribution curve
    if (len(hist_data)!= 0):    
        myfig2 = ff.create_distplot(hist_data, group_labels, show_hist= False )
        myfig2['layout'].update(title='Taxi out time of selected carrier at departure airport')
        myfig2['layout'].update(xaxis={'title':'Time(in minutes)'})
        return myfig2
    
#below callback returns graph for taxi in time distribution at arrival airport by taking all the selected option as input
@app.callback(
    dash.dependencies.Output('pd-indicator-scatter-1', 'figure'),
    [dash.dependencies.Input('xaxis-column-3', 'value'),
    dash.dependencies.Input('yaxis-column-3', 'value'),
    dash.dependencies.Input('zaxis-column-3', 'value')])

def update_graph(xaxis_column, yaxis_column, zaxis_column):
    #first we scrutinuize the data
    dff = df[df['carrier'] == xaxis_column]
    dfff = dff[dff['dest'] == zaxis_column] 
    
    #now we take taxi-in time into account
    dfToList = list(dfff['taxi_in'])
    dfToList = list(dfToList)
    
    #the taxi in time will be provided as hitogram data
    hist_data = [dfToList]
    group_labels = ['Percentage Distribution Curve']
    #creating percentage distribution curve
    if (len(hist_data)!= 0):    
        myfig1 = ff.create_distplot(hist_data, group_labels, show_hist= False )
        myfig1['layout'].update(title='Taxi in time of selected carrier at arrival airport')
        myfig1['layout'].update(xaxis={'title':'Time(in minutes)'})

        return myfig1        

if __name__ == '__main__':
    app.run_server(debug=True)