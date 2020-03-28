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
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import cmocean
import matplotlib.pyplot as plt
from app import app
df = pd.read_csv("Data/us-bts-parsed-data.csv")
df['15_flag']=0
df['15_flag'] = np.where(df.dep_delay >15, df['15_flag'],1)
df['45_flag']=0
df['45_flag'] = np.where(df.dep_delay >45, df['45_flag'],1)
df['60_flag']=0
df['60_flag'] = np.where(df.dep_delay >60, df['60_flag'],1)
h = len(df[df['15_flag'] == 1])
delay = [15,45,60]
df.drop(['Unnamed: 0'], axis=1,inplace =True)
df['dep_delay'] = df['dep_delay'].clip_lower(0)
T = df.scheduled_dep_hour.unique()
T.sort()
flights = df.carrier.unique()
flight_origin = df.origin.unique()
layout = html.Div([
        html.Div( 
                 [html.H3([html.Span('EDA-DELAY DASHBOARD',className="main-heading-primary-in"),
                           html.Span('DAY-TIME WISE ON-TIME PERFORMANCE',className="main-heading-secondary")]
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
                ],className='hello',href='/EDAapps/edadelayindex')
            ],style={'float':'top','margin-left':'95%','height':'0px','width':'0px','background-color':'#ffffff'}) ,

        html.Div([
    html.Div([
                 

        html.Div([html.P('Carrier code:'),dcc.Dropdown(
                    id='xaxis-column-1',
                    options=[{'label': i, 'value': i} for i in flights],
                    value='WN'
            ),
            
        ],
       style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),


        html.Br(),
        html.Br(),

        html.Div([html.P('Delay type:'),dcc.Dropdown(
                id='yaxis-column-1',
                options=[{'label': i, 'value': i} for i in delay],
                value='15'
            ),

        ],
       style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),
 
         html.Br(),
         html.Br(),



         html.Div([html.P('Origin Airport:'),dcc.Dropdown(
                id='zaxis-column-1',
                options=[{'label': i, 'value': i} for i in flight_origin],
                value='ATL'
            ),

        ],
       style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),



        
         html.Div([dcc.Graph(
              id='heatmap-indicator-scatter',     
        )
                    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20','margin-left': '220px'}),

        


        html.Div([dcc.Graph(
              id='heatmap-indicator-scatter-1',     
        )
                    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20','margin-left': '220px'}),

    ]),
],style={'margin-top':'250px'}),
])


@app.callback(
    dash.dependencies.Output('heatmap-indicator-scatter', 'figure'),
    [dash.dependencies.Input('xaxis-column-1', 'value'),
    dash.dependencies.Input('yaxis-column-1', 'value'),
    dash.dependencies.Input('zaxis-column-1', 'value')])

def update_graph_1(xaxis_column, yaxis_column, zaxis_column):
    dff = df[df['carrier'] == xaxis_column]
    dfff = dff[dff['origin'] == zaxis_column]    
    L = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    T = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    B = []
    print(dfff.shape)
    if(yaxis_column==15):
        for i in L:
            dftemp = dfff[dfff['day_of_week']==i]
            for j in T:
                dftemp1 = dftemp[dftemp['scheduled_dep_hour']==j]
                mylen = len(dftemp1[dftemp1['15_flag'] == 1])
                if(dftemp1.shape[0]!=0):
                    B.append((mylen/dftemp1.shape[0]))
                else:
                    B.append(0)
    elif(yaxis_column==45):
        for i in L:
            dftemp = dfff[dfff['day_of_week']==i]
            for j in T:
                dftemp1 = dftemp[dftemp['scheduled_dep_hour']==j]
                mylen = len(dftemp1[dftemp1['45_flag'] == 1])
                if(dftemp1.shape[0]!=0):
                    B.append((mylen/dftemp1.shape[0]))
                else:
                    B.append(0)
    elif(yaxis_column==60):
        for i in L:
            dftemp = dfff[dfff['day_of_week']==i]
            for j in T:
                dftemp1 = dftemp[dftemp['scheduled_dep_hour']==j]
                mylen = len(dftemp1[dftemp1['60_flag'] == 1])
                if(dftemp1.shape[0]!=0):
                    B.append((mylen/dftemp1.shape[0]))
                else:
                    B.append(0)
    print("length of b is",len(B))
    print(B)   
    if(len(B)==168):        
        B = np.reshape(B,(7,24))
    return {'data': [
                go.Heatmap(z=B,
                   x=T,
                   y=L)
            ],
            'layout': {
                'title': 'Hourwise On-Time Performance values',
                'xaxis' : {'title':'Hour of day'},
                
            }}

@app.callback(
    dash.dependencies.Output('heatmap-indicator-scatter-1', 'figure'),
    [dash.dependencies.Input('xaxis-column-1', 'value'),
    dash.dependencies.Input('yaxis-column-1', 'value'),
    dash.dependencies.Input('zaxis-column-1', 'value')])
    
def update_graph_2(xaxis_column, yaxis_column, zaxis_column):
    dff = df[df['carrier'] == xaxis_column]
    dfff = dff[dff['origin'] == zaxis_column]    
    L = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    T = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    B = []
    print(dfff.shape)
    if(yaxis_column==15):
        for i in L:
            dftemp = dfff[dfff['day_of_week']==i]
            for j in T:
                dftemp1 = dftemp[dftemp['scheduled_dep_hour']==j]
                B.append(dftemp1['dep_delay'].mean())
    elif(yaxis_column==45):
        for i in L:
            dftemp = dfff[dfff['day_of_week']==i]
            for j in T:
                dftemp1 = dftemp[dftemp['scheduled_dep_hour']==j]
                mylen = len(dftemp1[dftemp1['45_flag'] == 1])
                B.append(dftemp1['dep_delay'].mean())
    elif(yaxis_column==60):
        for i in L:
            dftemp = dfff[dfff['day_of_week']==i]
            for j in T:
                dftemp1 = dftemp[dftemp['scheduled_dep_hour']==j]
                mylen = len(dftemp1[dftemp1['60_flag'] == 1])
                B.append(dftemp1['dep_delay'].mean())
    print("length of b is",len(B))
    print(B)   
    if(len(B)==168):        
        B = np.reshape(B,(7,24))
    return {'data': [
                go.Heatmap(z=B,
                   x=T,
                   y=L)
            ],
            'layout': {
                'title': 'Hourwise mean departure delay',
                'xaxis' : {'title':'Hour of day'},
                

            }}

    
if __name__ == '__main__':
    app.run_server(debug=True)