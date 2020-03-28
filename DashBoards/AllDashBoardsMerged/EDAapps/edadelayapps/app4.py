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
import plotly.figure_factory as ff
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
df = df.dropna()
T = df.scheduled_dep_hour.unique()
T.sort()
flights = df.carrier.unique()
flight_origin = df.origin.unique()
layout = html.Div([
        html.Div( 
                 [html.H3([html.Span('EDA-DELAY DASHBOARD',className="main-heading-primary-in"),
                           html.Span('AIRLINE ROUTE WISE TAXI-OUT TIME DISTRIBUTIONS',className="main-heading-secondary")]
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
                    id='xaxis-column-3',
                    options=[{'label': i, 'value': i} for i in flights],
                    value='WN'
            ),
            
        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),


    html.Br(),
    html.Br(),


        html.Div([html.P('Origin Airport:'),dcc.Dropdown(
                id='yaxis-column-3',
                options=[{'label': i, 'value': i} for i in flight_origin],
                value='ATL'
            ),

        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),
        
             html.Br(),
             html.Br(),




         html.Div([html.P('Destination Airport:'),dcc.Dropdown(
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
],style={'margin-top':'250px'}),
])
@app.callback(
    dash.dependencies.Output('yaxis-column-3', 'options'),
    [dash.dependencies.Input('xaxis-column-3', 'value')])
def set_dep_options_1(xaxis_column):
    mydata = df
    mydata1 = mydata[mydata['carrier']==xaxis_column]
    K = mydata1.origin.unique()
    return [{'label': i, 'value': i} for i in K]


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


@app.callback(
    dash.dependencies.Output('pd-indicator-scatter', 'figure'),
    [dash.dependencies.Input('xaxis-column-3', 'value'),
    dash.dependencies.Input('yaxis-column-3', 'value'),
    dash.dependencies.Input('zaxis-column-3', 'value')])

def update_graph(xaxis_column, yaxis_column, zaxis_column):
    dff = df[df['carrier'] == xaxis_column]
    dfff = dff[dff['origin'] == yaxis_column]    
    dfToList = list(dfff['taxi_out'])
    dfToList = list(dfToList)
    hist_data = [dfToList]
    group_labels = ['Percentage Distribution Curve']
    if (len(hist_data)!= 0):    
        myfig2 = ff.create_distplot(hist_data, group_labels, show_hist= False )
        myfig2['layout'].update(title='Taxi out time of selected carrier at departure airport')
        myfig2['layout'].update(xaxis={'title':'Time(in minutes)'})
        return myfig2
    
    
@app.callback(
    dash.dependencies.Output('pd-indicator-scatter-1', 'figure'),
    [dash.dependencies.Input('xaxis-column-3', 'value'),
    dash.dependencies.Input('yaxis-column-3', 'value'),
    dash.dependencies.Input('zaxis-column-3', 'value')])

def update_graph(xaxis_column, yaxis_column, zaxis_column):
    dff = df[df['carrier'] == xaxis_column]
    dfff = dff[dff['dest'] == zaxis_column]    
    dfToList = list(dfff['taxi_in'])
    dfToList = list(dfToList)
    hist_data = [dfToList]
    group_labels = ['Percentage Distribution Curve']
    if (len(hist_data)!= 0):    
        myfig1 = ff.create_distplot(hist_data, group_labels, show_hist= False )
        myfig1['layout'].update(title='Taxi in time of selected carrier at arrival airport')
        myfig1['layout'].update(xaxis={'title':'Time(in minutes)'})

        return myfig1        
    
@app.callback(
    dash.dependencies.Output('pd-indicator-scatter-2', 'figure'),
    [dash.dependencies.Input('xaxis-column-3', 'value'),
    dash.dependencies.Input('yaxis-column-3', 'value'),
    dash.dependencies.Input('zaxis-column-3', 'value')])

def update_graph(xaxis_column, yaxis_column, zaxis_column):
    dff = df[df['carrier'] == xaxis_column]
    dfff = dff[dff['origin'] == yaxis_column]  
    dfff = dfff[dfff['dest'] == zaxis_column]
    
    dfToList = list(dfff['actual_block_time'])
    dfToList = list(dfToList)
    jj = dfff['scheduled_block_time'].mean()
    dfToList2 = [jj]
    hist_data = [dfToList]
    group_labels = ['Percentage Distribution Curve']
    if (len(hist_data)!= 0):    
        myfig = ff.create_distplot(hist_data, group_labels, show_hist= False )
        myfig['layout'].update(title='Block time of selected carrier')
        myfig['layout'].update(xaxis={'title':'Time(in minutes)'})
        myfig['layout'].update(display='inline-block')
        return myfig

if __name__ == '__main__':
    app.run_server(debug=True)