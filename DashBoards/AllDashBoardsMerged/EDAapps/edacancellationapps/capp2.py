# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:05:07 2019

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
available_indicator_airline = data['UNIQUE_CARRIER'].unique()  


layout =html.Div([
        html.Div( 
                 [html.H3([html.Span('EDA-CANCELLATION DASHBOARD',className="main-heading-primary-in"),
                           html.Span('AIRLINE WISE CANCELLATIONS',className="main-heading-secondary")]
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
                html.Div([html.H3('Carrier code:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-ce23',
						  options=[{'label': i, 'value': i} for i in available_indicator_airline],
						  # value = ['SPY'], 
                          value='',
                          
						  style={'fontSize': 15, 'width': 150},
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-top':'-30px','margin-left':'40px'}),
                
				html.Div([html.H3('Flight number:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-ce2',
						   # value = ['SPY'], 
						  multi = True,
						  style={'fontSize': 15, 'width': 300},
                          
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%','margin-top':'-30px'}),
                
				html.Div([html.H3('Date Range:'),
					dcc.DatePickerRange(id='my_date_picker-ce2',
										min_date_allowed = dt(2016,1,1),
										max_date_allowed = dt(2016,12,28),
										start_date = dt(2016, 1,2),
										end_date = dt(2016, 12,25),
                                         display_format='MMM Do, YY'
					)

				], style={'display':'inline-block','margin-top':'-30px'}), 
                    
				html.Div([
					html.Button(id='submit-button-ce2',
								n_clicks = 0,
								children = 'Submit',
								style = {'fontSize': 24, 'marginLeft': '30px'}

					)

				], style={'display': 'None'}),
				# dcc.Markdown(''' --- '''),

				# Total Return Charts section
				# html.H1('Total Return Charts'),
					
				dcc.Markdown(''' --- '''), 
				
				html.Div([dcc.Graph(id='my_graph-ce2',
							
				),],style={'width': '85%', 'display': 'inline-block', 'padding': '0 20','margin-left': '220px','margin-top': '-35px'}),
					
html.Div(id = 'textonly-ce2',children='Cancellation Distribution', style={
        'textAlign': 'center',
        
    }),

],style={'margin-top':'250px'}),
])
@app.callback(
                Output('my_ticker_symbol-ce2', 'options'),
                [Input('my_ticker_symbol-ce23', 'value')]
)
def set_flightNumber_options_ce2(selected_airline):
                new_data = data[data['UNIQUE_CARRIER'] == selected_airline]
                return [{'label' : i, 'value' : i} for i in new_data['FL_NUM'].unique()]


@app.callback(Output('my_graph-ce2', 'figure'),
				[Input('submit-button-ce2', 'n_clicks'),
                 Input('my_ticker_symbol-ce23', 'value'),
				Input('my_ticker_symbol-ce2', 'value'),  
				Input('my_date_picker-ce2', 'start_date'),
                Input('my_date_picker-ce2', 'end_date')])
def update_graph_ce2(n_clicks,airline,stock_ticker,startdate,enddate):
    
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
    filtered_df1 = filtered_df[filtered_df['UNIQUE_CARRIER']==airline]
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
                'yaxis' : {'title':'Total Cancellations'},}
	}
    return fig1

@app.callback(Output('textonly-ce2', 'children'),
				[Input('submit-button-ce2', 'n_clicks'),
                 Input('my_ticker_symbol-ce23', 'value'),
				Input('my_ticker_symbol-ce2', 'value'),  
				Input('my_date_picker-ce2', 'start_date'),
                Input('my_date_picker-ce2', 'end_date')])
def update_text_ce2(n_clicks,airline,stock_ticker,startdate,enddate):
    
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
    filtered_df1 = filtered_df[filtered_df['UNIQUE_CARRIER']==airline]
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