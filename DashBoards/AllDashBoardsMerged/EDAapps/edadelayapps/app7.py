# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 21:15:31 2019

@author: bsoni
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.figure_factory as ff
from app import app
data = pd.read_csv("Data/us-bts-parsed-data.csv")
available_indicator_airline = data['carrier'].unique()  
delay = [15,45,60]
layout = html.Div([
        html.Div( 
                 [html.H3([html.Span('EDA-DELAY DASHBOARD',className="main-heading-primary-in"),
                           html.Span('BLOCK TIME DASHBOARD',className="main-heading-secondary")]
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
                    id='airline',
                    options=[{'label': i, 'value': i} for i in available_indicator_airline],
                    value=''
            ),
            
        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),


    html.Br(),
    html.Br(),


        html.Div([html.P('Origin Airport:'),dcc.Dropdown(
                id='origin_airport',
            ),

        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),
        
             html.Br(),
    html.Br(),




         html.Div([html.P('Destination Airport:'),dcc.Dropdown(
                id='destination_airport',
                
            ),

        ],
        style={'width': '48%', 'display': 'inline-block','margin-left': '220px'}),
            
            html.Br(),
    html.Br(),
       
         html.Div([html.P('Flight number:'),dcc.Dropdown(
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
],style={'margin-top':'250px'}),
])
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
                
                dxc = data[data['origin']==orgairport]
                dxc = dxc[dxc['dest']==destairport]
                filtered_data = dxc[dxc['flight_number'] == flightno]
                filtered_data.dropna(axis=0, subset=['actual_block_time'], inplace=True)
                print(filtered_data.shape)

                hist_data = [filtered_data['actual_block_time']]
                constantData = filtered_data['scheduled_block_time'].values.tolist()
                print(type(constantData))
                #zzz = constantData[0]
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
                print(fig)
                #fig['layout'].update(shapes = {'type':'line', 'x0':0, 'y0':constantData[0] , 'x1':0 , 'y1':constantData[0]})
                #trace2 = go.Scatter( x = [0,0], y = [zz,zz], mode = 'lines',name = 'lines')
                #dataz = [fig,trace2]
                return fig




if __name__ == '__main__':
    app.run_server(debug=True)