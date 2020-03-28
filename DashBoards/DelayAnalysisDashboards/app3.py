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


#drop unnamed columns
df.drop(['Unnamed: 0'], axis=1,inplace =True)
#create of list of unique carriers and delay type from dataframe
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

        html.Div([html.P('Select a carrier :'),dcc.Dropdown(
                    id='xaxis-column-2',
                    options=[{'label': i, 'value': i} for i in flights],
                    value='WN'
            ),
            
        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),



        
        html.Div([dcc.Graph(
               id='crossfilter-indicator-scatter-2',
               hoverData={'points': [{'customdata': 'Japan'}]}
                            
               )], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20','margin-left': '220px'}),
    
    html.Div([dcc.Slider(
            id='my-slider-2',
            min=1,
            max=25,
            step=1,
            marks={
                    1: '1',
                    5: '5',
                    10: '10',
                    15: '15',
                    20: '20',
                    25: '25'
                },
            value=10,
            )],
             style={'margin-left': '220px','margin-top': '20','display': 'inline-block','width':'75%'},

        ),

                     
        
    ]),

])

#Now we are starting app callbacks
# each callback calls a function
#the function name should be unique within the app.
#the called function takes the input in same order as written in callback
#below callback returns the GRAPH options for selected VALUES 
@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter-2', 'figure'),
    [dash.dependencies.Input('xaxis-column-2', 'value'),
    dash.dependencies.Input('my-slider-2', 'value')])

def update_graph(xaxis_column, my_slider):
    #copy the data to datadash so that we can use it further
    #!Always use copy function to  copy the data
    #!Simply using datadash=data can be harmful, as if you change something in datadash it will do same changes in data 
    dff = df.copy()
    #updates the values according to slider
    L = dff.freq.unique()
    #Now we scrutiny the data according to selected variables
    dfftemp  = dff[dff['carrier']==xaxis_column]
    #continue with updating the values according to slider
    lngth = len(dfftemp.origin.unique())
    my_slider1 = lngth
    L = L[:my_slider1]
    #scrutinizing the data again
    dff = dff[dff['freq'].isin(L)]
    #unique list of origin from scrutinized list
    M = dff.origin.unique()
    #Since we are comparing two graphs here, we need two values of y axis
    #sm will show the overall mean taxi-out while sm_f1 will show the mean taxi out for selected airline
    Sm = []
    Sm_fl = []
    
    #below code calculates the mean taxi out at all airports and at selected airports respectively
    for i in M:
        df1 = dff[dff['origin']==i]
        Sm.append(df1['taxi_out'].mean())
        
    dfff = dff[dff['carrier'] == xaxis_column]
    for i in M:
        dfff1 = dfff[dfff['origin']==i]
        Sm_fl.append(dfff1['taxi_out'].mean())    
    
    #removing the nan values from the data
    myL = np.argwhere(np.isnan(Sm_fl))
    myL = myL.ravel()
    
    Sm_fl = [i for j, i in enumerate(Sm_fl) if j not in myL]
    Sm = [i for j, i in enumerate(Sm) if j not in myL]
    M = [i for j, i in enumerate(M) if j not in myL]
    
    return {'data': [
                {'x': M[:my_slider], 'y': Sm[:my_slider], 'type': 'bar', 'name': 'Selected Flight'},
                {'x': M[:my_slider], 'y': Sm_fl[:my_slider], 'type': 'bar', 'name': 'Overall Airport'},
            ],
            'layout': {
                'title': 'Mean Taxi Out Time comparison of Selected flight with cumulative mean on selected airport',
                'xaxis' : {'title':'Airports'},
                'yaxis' : {'title':'Mean Taxi-out time(flight vs Overall)'}
            }}


if __name__ == '__main__':
    app.run_server(debug=True)