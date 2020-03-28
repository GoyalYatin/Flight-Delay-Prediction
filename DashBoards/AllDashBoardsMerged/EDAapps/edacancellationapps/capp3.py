# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:18:54 2019

@author: bsoni
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:55:21 2019

@author: bsoni
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import datetime
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime as dt
from app import app

data = pd.read_csv('Data/flightsandweatherdata.csv')
data.ORIGIN.unique()

data['FL_DATE'] = data['FL_DATE'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d")) 

data['FL_month'] = data['FL_DATE'].apply(lambda x: x.strftime('%B'))
data['FL_month'] = data['FL_month'].apply(lambda x: str(x))
data['FLY_date'] = data['FL_DATE'].apply(lambda x: x.date())
cols = list(data)
available_indicator_airline = data['ORIGIN'].unique()  


layout = html.Div([
        html.Div( 
                 [html.H3([html.Span('EDA-CANCELLATION DASHBOARD',className="main-heading-primary-in"),
                           html.Span('ROUTE-WISE CANCELLATIONS',className="main-heading-secondary")]
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
                ],className='hello',href='/EDAapps/edacancellationindex')
            ],style={'float':'top','margin-left':'95%','height':'0px','width':'0px','background-color':'#ffffff'}) ,
				

html.Div([
                html.Div([html.H3('Origin Airport:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
                        options=[{'label': i, 'value': i} for i in available_indicator_airline],
						  id='my_ticker_symbol-ce32',
						   style={'fontSize': 15, 'width': 200},
                           
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-top':'-30px','margin-left':'70px'}),
                html.Div([html.H3('Destination Airport:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-ce31',
						  style={'fontSize': 15, 'width': 200},
                          
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-top':'-30px'}),
                
				html.Div([html.H3('Flight number:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-ce3',
						   # value = ['SPY'], 
						  multi = True,
						  style={'fontSize': 15, 'width': 200},
                          
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-top':'-30px'}),
                
				html.Div([html.H3('Date Range'),
					dcc.DatePickerRange(id='my_date_picker-ce3',
										min_date_allowed = dt(2016,1,1),
										max_date_allowed = dt(2016,12,28),
										start_date = dt(2016, 1,2),
										end_date = dt(2016, 12,25),
                                         display_format='MMM Do, YY'
					)

				], style={'display':'inline-block','margin-top':'-10px','margin-left':'70px'}), 
                    
				html.Div([
					html.Button(id='submit-button-ce3',
								n_clicks = 0,
								children = 'Submit',
								style = {'fontSize': 24, 'marginLeft': '30px'}

					)

				], style={'display': 'none'}),
				# dcc.Markdown(''' --- '''),

				# Total Return Charts section
				# html.H1('Total Return Charts'),
					
				dcc.Markdown(''' --- '''), 
				
				html.Div([dcc.Graph(id='my_graph-ce3',
							
				),],style={'width': '85%', 'display': 'inline-block', 'padding': '0 20','margin-left': '220px','margin-top': '-35px'}),	

html.Div(id = 'textonly-ce3',children='Cancellation Distribution', style={
        'textAlign': 'center',
        
    }),

],style={'margin-top':'250px'}),					

])

#This callback is to return selected airline and origin airport for available options for destination airport
@app.callback(
                Output('my_ticker_symbol-ce31', 'options'),
                [
                Input('my_ticker_symbol-ce32', 'value')]
)
def set_destination_options_ce3( origin_airport):
                new_data = data[data['ORIGIN'] == origin_airport]
                return [{'label' : i, 'value' : i} for i in new_data['DEST'].unique()]
#This callback is to return selected airline, origin airport and destination airport for available options for flight number
@app.callback(
                dash.dependencies.Output('my_ticker_symbol-ce3', 'options'),
                [dash.dependencies.Input('my_ticker_symbol-ce32', 'value'),
                dash.dependencies.Input('my_ticker_symbol-ce31', 'value')]
)
def set_flightno_options_ce3(origin_airport, destination_airport):
                
                new_data = data[data['ORIGIN'] == origin_airport]
                new_data = new_data[new_data['DEST'] == destination_airport]
                return [{'label' : i, 'value' : i} for i in new_data['FL_NUM'].unique()]

@app.callback(Output('my_graph-ce3', 'figure'),
				[Input('submit-button-ce3', 'n_clicks'),
                 Input('my_ticker_symbol-ce32', 'value'),
                 Input('my_ticker_symbol-ce31', 'value'),
				Input('my_ticker_symbol-ce3', 'value'),  
				Input('my_date_picker-ce3', 'start_date'),
                Input('my_date_picker-ce3', 'end_date')])
def update_graph_ce3(n_clicks,origin,dest,stock_ticker,startdate,enddate):
    
    print("looooooooooooooooooooooooool")
    print(type(startdate))
    startdate = startdate[:10]
    enddate = enddate[:10]
    print(startdate)
    start_date = datetime.datetime.strptime(startdate,'%Y-%m-%d')
    end_date = datetime.datetime.strptime(enddate,'%Y-%m-%d')
    filtered_df = data[data.FL_DATE.between(
        datetime.datetime.strftime(start_date, "%Y-%m-%d"),
        datetime.datetime.strftime(end_date, "%Y-%m-%d")
    )]
    filtered_df1 = filtered_df[filtered_df['ORIGIN']==origin]
    filtered_df1 = filtered_df1[filtered_df1['DEST']==dest]
    print(stock_ticker)
    print(type(stock_ticker))
    for i in stock_ticker:
        print(type(i))
    traces = []
    for i in stock_ticker:
        cCount = []
        df = filtered_df1[filtered_df1['FL_NUM']==i]
        GGHH = df.CANCELLATION_CODE.unique()
        codess= []
        jjjj = []
        for o in GGHH:
            dffff = df[df['CANCELLATION_CODE']==o]
            KKKKK = dffff.shape[0]
            codess.append(KKKKK)
            jjjj.append(i)
        print("================",codess)
        months = df.FL_month.unique()
        for j in months:
            df1 = df[df['FL_month']==j]
            cCount.append(df1[df1.CANCELLED == 1].shape[0])
        #depD = df['CANCELLED'].tolist()
        #depHour = df['FLY_date'].tolist()
        traces.append({'x':months, 'y':cCount, 'name': i})
        
    fig1 = {
		'data': traces,
		'layout': {'title':stock_ticker,
             'xaxis' : {'title':'Months'},
                'yaxis' : {'title':'Total Cancellations'},
                          }
	}
    return fig1


@app.callback(Output('textonly-ce3', 'children'),
				[Input('submit-button-ce3', 'n_clicks'),
                 Input('my_ticker_symbol-ce32', 'value'),
                 Input('my_ticker_symbol-ce31', 'value'),
				Input('my_ticker_symbol-ce3', 'value'),  
				Input('my_date_picker-ce3', 'start_date'),
                Input('my_date_picker-ce3', 'end_date')])
def update_text1_ce3(n_clicks,origin,dest,stock_ticker,startdate,enddate):
    
    print("looooooooooooooooooooooooool")
    print(type(startdate))
    startdate = startdate[:10]
    enddate = enddate[:10]
    print(startdate)
    start_date = datetime.datetime.strptime(startdate,'%Y-%m-%d')
    end_date = datetime.datetime.strptime(enddate,'%Y-%m-%d')
    filtered_df = data[data.FL_DATE.between(
        datetime.datetime.strftime(start_date, "%Y-%m-%d"),
        datetime.datetime.strftime(end_date, "%Y-%m-%d")
    )]
    filtered_df1 = filtered_df[filtered_df['ORIGIN']==origin]
    filtered_df1 = filtered_df1[filtered_df1['DEST']==dest]
    print(stock_ticker)
    print(type(stock_ticker))
    for i in stock_ticker:
        print(type(i))
    traces = []
    for i in stock_ticker:
        traces1 = []
        df = filtered_df1[filtered_df1['FL_NUM']==i]
        GGHH = df.CANCELLATION_CODE.unique()
        GGHH = [x for x in GGHH if str(x) != 'nan']
        codess= []
        jjjj = []
        stri = ' '
        shape = df[df['CANCELLED']==1].shape[0]
        if(shape!=0):
            for k in GGHH:
                dffff = df[df['CANCELLATION_CODE']==k]
                KKKKK = ((dffff.shape[0])/shape)*100
                KKKKK = round(KKKKK,2)
                codess.append(KKKKK)
                jjjj.append(k)
        print("================",codess)
        months = df.FL_month.unique()
        count=0
        for j in GGHH:
            stri=stri+str(jjjj[count]+":"+str(codess[count]))+"%"
            count=count+1
        traces.append("Flight Number-"+str(i)+" "+" Cancellation Division"+stri)
        #traces1 = html.Div(html.P1(traces))
        for i in traces:
            traces1.append(html.Div(html.H1(i)))    
    return html.Div(traces1)



if __name__ == '__main__':
    app.run_server()