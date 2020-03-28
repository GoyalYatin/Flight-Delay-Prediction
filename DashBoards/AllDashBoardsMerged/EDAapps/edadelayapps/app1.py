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
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from app import app
#from apps import app1,app2,app3,app4,app5,app6
#from app import app
#import dash
df = pd.read_csv("Data/us-bts-parsed-data.csv")
df['15_flag']=0
df['15_flag'] = np.where(df.dep_delay >15, df['15_flag'],1)
df['45_flag']=0
df['45_flag'] = np.where(df.dep_delay >45, df['45_flag'],1)
df['60_flag']=0
df['60_flag'] = np.where(df.dep_delay >60, df['60_flag'],1)
h = len(df[df['15_flag'] == 1])
#delay = [15,45,60]
df.drop(['Unnamed: 0'], axis=1,inplace =True)
T = df.scheduled_dep_hour.unique()
T.sort()
flights = df.carrier.unique()
flight_origin = df.origin.unique()
j=[1]
dfdelay = pd.read_csv("Data/us-bts_parsed_data_delay_reason.csv")
flightsdelay = df.carrier.unique()
#data = pd.read_csv("flights.csv")
#available_indicator_airline = data['AIRLINE'].unique()  
delay = [15,45,60]

print("this print every time")
flight_origin_delay = df.origin.unique()
print(dcc.__version__) # 0.6.0 or above is requir

#==============================================page 1==============================================================================
layout = html.Div([
        html.Div( 
                 [html.H3([html.Span('EDA-DELAY DASHBOARD',className="main-heading-primary-in"),
                           html.Span('AIRPORT WISE ON-TIME PERFORMANCE DASHBOARD',className="main-heading-secondary")]
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
                 
            
        html.Div([ html.P('Carrier code:'),
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



        html.Div([html.P('Delay type :'),
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
        
    ]),
],style={'margin-top':'250px'}),

])
@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter-11', 'figure'),
    [dash.dependencies.Input('xaxis-column-11', 'value'),
    dash.dependencies.Input('yaxis-column-11', 'value'),
    dash.dependencies.Input('my-slider', 'value')])

def update_graph(xaxis_column, yaxis_column, my_slider):
    print(df.head(3))
    dff = df[df['carrier'] == xaxis_column]    
    print('im here')
    L = dff.freq.unique()
    L = L[:my_slider]
    dff = dff[dff['freq'].isin(L)]
    M = dff.origin.unique()
    N = []
    print("working till here")
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