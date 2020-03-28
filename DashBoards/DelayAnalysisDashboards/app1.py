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
h = len(df[df['15_flag'] == 1])

#drop unnamed columns
df.drop(['Unnamed: 0'], axis=1,inplace =True)

#make a list of crs depature time
#It may help us in future
T = df.scheduled_dep_hour.unique()
T.sort()

#create of list of unique carriers and delay from dataframe
#It will help us in future
flights = df.carrier.unique()
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

        html.Div([ html.P('Select a carrier :'),
                  dcc.Dropdown(
                    id='xaxis-column-11',
                    options=[{'label': i, 'value': i} for i in flights],
                    value='WN'
            ),
            
        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),

            html.Br(),
            html.Br(),
            html.Br(),



        html.Div([html.P('Select delay type :'),
                  dcc.Dropdown(
                id='yaxis-column-11',
                options=[{'label': i, 'value': i} for i in delay],
                value='15'
            ),

        ],
        style={'width': '48%', 'margin-left': '220px', 'display': 'inline-block'}),
        
         


        
        html.Div([dcc.Graph(
               id='crossfilter-indicator-scatter-11',
               hoverData={'points': [{'customdata': 'Japan'}]}
                            
               )], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20','margin-left': '220px'}),
                    
       


        html.Div([dcc.Slider(
            id='my-slider',
            min=1,
            max=50,
            step=1,
            marks={
                    1: '1',
                    10: '10',
                    20: '20',
                    30: '30',
                    40: '40',
                    50: '50'
                },
            value=10,
            )],
             style={'margin-left': '220px','margin-top': '20','display': 'inline-block','width':'75%'}),
        
    
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        dcc.Link('Go back to home', href='/')
        
    ]),

])
#Now we are starting app callbacks
# each callback calls a function
#the function name should be unique within the app.
#the called function takes the input in same order as written in callback
#below callback returns the GRAPH options for selected VALUES 
@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter-11', 'figure'),
    [dash.dependencies.Input('xaxis-column-11', 'value'),
    dash.dependencies.Input('yaxis-column-11', 'value'),
    dash.dependencies.Input('my-slider', 'value')])

def update_graph(xaxis_column, yaxis_column, my_slider):
    #Now we scrutiny the data according to selected variables
    dff = df[df['carrier'] == xaxis_column]    
    #updates the values according to slider
    L = dff.freq.unique()
    L = L[:my_slider]
    #scrutinizing the data again
    dff = dff[dff['freq'].isin(L)]
    
    #unique list of origin from scrutinized list
    M = dff.origin.unique()
    N = []
    
    #This code helps us to keep the graph updating
    #it takes delay type into account and check the delays occuring for each case
    #It than outputs the percentage for each case
    if(yaxis_column==15):
        for i in M:
            dftemp = dff[dff['origin'] == i]
            mylen = len(dftemp[dftemp['15_flag'] == 1])
            N.append((mylen/dftemp.shape[0])*100)
            print(N)
    elif(yaxis_column==45):
        for i in M:
            dftemp = dff[dff['origin'] == i]
            mylen = len(dftemp[dftemp['45_flag'] == 1])
            N.append((mylen/dftemp.shape[0])*100)
            print(N)
    elif(yaxis_column==60):
        for i in M:
            dftemp = dff[dff['origin'] == i]
            mylen = len(dftemp[dftemp['60_flag'] == 1])
            N.append((mylen/dftemp.shape[0])*100)
            print(N)
                   
    return {'data': [
                {'x': M, 'y': N, 'type': 'bar', 'name': 'SF'},
            ],
            'layout': {
                'title': 'On-Time performance of flights : Airport Wise',
                'xaxis' : {'title':'Airports'},
                'yaxis' : {'title':'Per\'age of On-Time flights'}
            }}
    



if __name__ == '__main__':
    app.run_server(debug=True)