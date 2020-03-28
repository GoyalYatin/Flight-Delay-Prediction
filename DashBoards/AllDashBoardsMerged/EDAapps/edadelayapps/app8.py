# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:17:50 2019

@author: bsoni
"""

import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import plotly.io as pio
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from datetime import datetime as dt
from dash.dependencies import Input, Output
from app import app

delaytraindata1 = pd.read_csv('Data/DelayData2016.csv')
delaytraindata2 = pd.read_csv('Data/DelayData2017.csv')
aggr_dataset = [delaytraindata1 , delaytraindata2]
delaydata = pd.concat(aggr_dataset)
available_indicator_airline = delaydata.UNIQUE_CARRIER.unique()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

layout = html.Div([
                html.Div( 
                 [html.H3([html.Span('EDA-DELAY DASHBOARD',className="main-heading-primary-in"),
                           html.Span('DESTINATION WISE ARRIVAL DELAYS DISTRIBUTION MAP',className="main-heading-secondary")]
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
                
				 html.Div([html.H3('Carrier code:', style={'margin-left': '149px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-a803',
						  options=[{'label': i, 'value': i} for i in available_indicator_airline],
						  # value = ['SPY'], 
                          value='',
                          
						  style={'fontSize': 15, 'width': 300,'margin-left': '75px',},
				)], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-left': '220px'}),
    
    
                    
				html.Div([
					html.Button(id='submit-button-a800',
								n_clicks = 0,
								children = 'Submit',
								style = {'fontSize': 24, 'marginLeft': '30px','margin-left': '75px'}

					)

				], style={'display': 'None'}),
				# dcc.Markdown(''' --- '''),

				# Total Return Charts section
				# html.H1('Total Return Charts'),
					
				dcc.Markdown(''' --- '''), 
				
				html.Div([dcc.Graph(id='my_graph-a800',
							figure={'layout':go.Layout(title='daily Cancellations will be shown here', 
                               
                                         )}
				), ],id='graph',style={'width': '85%', 'display': 'inline-block', 'padding': '0 20','margin-left': '340px'}),

    
               
			],style={'margin-top':'250px'}),		

], id='particles-js')
    

@app.callback(Output('my_graph-a800', 'figure'),
				[Input('submit-button-a800', 'n_clicks'),
                 Input('my_ticker_symbol-a803', 'value'),])
def update_graph(n_clicks,airline):
    
    delaydata1 = delaydata[delaydata['UNIQUE_CARRIER']==airline]
    L = ['BOS','EWR','JFK','LGA','ORD','DEN','DFW','IAH','PHL','SFO']
    DelayCount = []
    TotalFlights = []
    for i in range(len(L)):
        dfTotal = delaydata1[delaydata1['DEST']==L[i]]
        TotalFlights.append(dfTotal.shape[0])
    for i in range(len(L)):
        df = delaydata1[delaydata1['DEST']==L[i]]
        df = df[df['ARR_DELAY']>10]
        DelayCount.append(df.shape[0])
    
    Lat = [42.3656,40.6895,40.6413,40.7769,41.9742,39.8561,32.8998,29.9902,39.8744,37.6213]
    Long = [-71.0096,-74.1745,-73.7781,-73.8740,-87.9073,104.6737,-97.0403,-95.3368,-75.2424,-122.3790]
    
    GeoData = pd.DataFrame(
        {'AIRPORT': L,
         'LAT': Lat,
         'LONG': Long,
         'DELAY_COUNT':DelayCount,
         'TOTAL_FLIGHTS':TotalFlights,
        })
    color = ['rgb(234, 129, 129)','rgb(234, 169, 129)','rgb(234, 206, 129)','rgb(158, 234, 129)','rgb(129, 234, 165)',
             'rgb(129, 234, 230)','rgb(129, 174, 234)','rgb(129, 134, 234)','rgb(204, 129, 234)','rgb(234, 129, 186)']
    print("====================",max(DelayCount))
    if(max(DelayCount)<=30):
        scale = 0.009
    elif((max(DelayCount)>30) & (max(DelayCount)<100)):
        scale = 0.01
    elif((max(DelayCount)>=100) & (max(DelayCount)<1000)):
        scale = 0.05
    elif((max(DelayCount)>=1000)&(max(DelayCount)<5000)):
        scale = 0.2
    elif((max(DelayCount)>=5000)&(max(DelayCount)<6000)):
        scale = 0.6
    elif((max(DelayCount)>=6000)):
        scale = 1.2
    print("++++++++++++++++",scale)
    cities = []
    count=0 
    for i in L:
        df1 = GeoData[GeoData['AIRPORT']==i]
        K = df1['DELAY_COUNT'].tolist()
        M = df1['TOTAL_FLIGHTS'].tolist()
        K = K[0]
        M = M[0]
        city = go.Scattergeo(
            locationmode = 'USA-states',
            lon = list(df1['LONG']),
            lat = df1['LAT'],
            text = "Total Delays: "+str(K)+" Total Flights: "+str(M),
            hoverinfo = 'text',
            marker = go.scattergeo.Marker(
                size = df1['DELAY_COUNT']/scale,
                color = color[count],
                
                line = go.scattergeo.marker.Line(
                    width=0.5, color='rgb(40,40,40)'
                ),
                sizemode = 'area'
            ),name= i)
        cities.append(city)
        count=count+1
    layout = go.Layout(
            title = go.layout.Title(
                text = 'Airport Wise Delays'
            ),
            width=1400,
            height=800,
            showlegend = True,
            geo = go.layout.Geo(
                scope = 'usa',
                projection = go.layout.geo.Projection(
                    type='albers usa'
                ),
                showland = True,
                landcolor = 'rgb(217, 217, 217)',
                subunitwidth=1,
                countrywidth=1,
                subunitcolor="rgb(255, 255, 255)",
                countrycolor="rgb(255, 255, 255)"
            )
        )
    fig = go.Figure(data=cities, layout=layout)
    return fig


    
if __name__ == '__main__':
    app.run_server()