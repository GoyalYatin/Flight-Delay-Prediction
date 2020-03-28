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
print("running app 3")
flights = df.carrier.unique()
flight_origin = df.origin.unique()
layout =  html.Div([
        html.Div( 
                 [html.H3([html.Span('EDA-DELAY DASHBOARD',className="main-heading-primary-in"),
                           html.Span('MEAN TAXI-OUT TIME BY AIRPORT',className="main-heading-secondary")]
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
],style={'margin-top':'250px'}),
])

@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter-2', 'figure'),
    [dash.dependencies.Input('xaxis-column-2', 'value'),
    dash.dependencies.Input('my-slider-2', 'value')])

def update_graph(xaxis_column, my_slider):
    dff = df
    L = dff.freq.unique()
    dfftemp  = dff[dff['carrier']==xaxis_column]
    lngth = len(dfftemp.origin.unique())
    my_slider1 = lngth
    L = L[:my_slider1]
    dff = dff[dff['freq'].isin(L)]
    M = dff.origin.unique()
    Sm = []
    Sm_fl = []
    for i in M:
        df1 = dff[dff['origin']==i]
        Sm.append(df1['taxi_out'].mean())
        
    dfff = dff[dff['carrier'] == xaxis_column]
    for i in M:
        dfff1 = dfff[dfff['origin']==i]
        Sm_fl.append(dfff1['taxi_out'].mean())    
    
    myL = np.argwhere(np.isnan(Sm_fl))
    myL = myL.ravel()
    print(len(Sm_fl))
    print(Sm_fl)
    #print(myL)
    Sm_fl = [i for j, i in enumerate(Sm_fl) if j not in myL]
    print(Sm_fl)
    print(len(Sm_fl))               
    Sm = [i for j, i in enumerate(Sm) if j not in myL]
    #print(Sm_fl)
    M = [i for j, i in enumerate(M) if j not in myL]
    print(M)
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