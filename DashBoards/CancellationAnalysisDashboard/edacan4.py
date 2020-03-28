# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:07:07 2019

@author: bsoni
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import datetime
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime as dt
#Initialize dash
app = dash.Dash()

#Import data
data = pd.read_csv('flightsandweatherdata.csv')

#We converted FL_DATE to ease out the merging
#Now we convert it back to datetime format
#We extract the month %B converts 01->JAN
#We than convert this columns to string
#Than we extract the date from the data
data['FL_DATE'] = data['FL_DATE'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d")) 
data['FL_month'] = data['FL_DATE'].apply(lambda x: x.strftime('%B'))
data['FL_month'] = data['FL_month'].apply(lambda x: str(x))
data['FLY_date'] = data['FL_DATE'].apply(lambda x: x.date())

#Make a list of columns of database
cols = list(data)

#make a list of unique airlines in data. It will update the dropdown menu later.
available_indicator_airline = data['ORIGIN'].unique()  

#below is app layout for our app
#it is designed in python with HTML on top
#make sure that every id you give to each element should not be repeated.
app.layout = html.Div([
				html.H1('Cancellation Dashboard'),
				dcc.Markdown(''' --- '''),
                html.Div([html.H3('Enter Origin city:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-3',
						  options=[{'label': i, 'value': i} for i in available_indicator_airline],
                          value='',
                          
						  style={'fontSize': 15, 'width': 150},
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
                
				html.Div([html.H3('Enter Destination City:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol-2',
                           style={'fontSize': 15, 'width': 300},
                          
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),

                html.Div([html.H3('Enter a Carrier:', style={'paddingRight': '30px'}),
				dcc.Dropdown(
						  id='my_ticker_symbol',
						  multi = True,
						  style={'fontSize': 15, 'width': 300},
                          
				)

				], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),

                
				html.Div([html.H3('Enter start / end date:'),
					dcc.DatePickerRange(id='my_date_picker',
										min_date_allowed = dt(2016,1,1),
										max_date_allowed = dt(2016,12,28),
										start_date = dt(2016, 1,2),
										end_date = dt(2016, 12,25),
                                         display_format='MMM Do, YY'
					)

				], style={'display':'inline-block'}), 
                    
				html.Div([
					html.Button(id='submit-button',
								n_clicks = 0,
								children = 'Submit',
								style = {'fontSize': 24, 'marginLeft': '30px'}

					)

				], style={'display': 'inline-block'}),
				dcc.Markdown(''' --- '''), 
				
                 dcc.Graph(id='my_graph_1',
							
				),	

html.Div(id = 'textonly',children='Cancellation Distribution', style={
        'textAlign': 'center',
        
    }),

])
#Now we are starting app callbacks
# each callback calls a function
#the function name should be unique within the app.
#the called function takes the input in same order as written in callback
#below callback returns the dest options for selected origin
@app.callback(
                Output('my_ticker_symbol-2', 'options'),
                [Input('my_ticker_symbol-3', 'value')]
)
def set_flightNumber_options(origin):
                new_data = data[data['ORIGIN'] == origin]
                return [{'label' : i, 'value' : i} for i in new_data['DEST'].unique()]
#below callback returns the carrier/airline options for selected origin-dest pair
@app.callback(
                Output('my_ticker_symbol', 'options'),
                [Input('my_ticker_symbol-3', 'value'),
                Input('my_ticker_symbol-2', 'value')]
)
def set_destination_options(origin, dest):
                new_data = data[data['ORIGIN'] == origin]
                new_data = new_data[new_data['DEST'] == dest]
                return [{'label' : i, 'value' : i} for i in new_data['UNIQUE_CARRIER'].unique()]
#This callback updates the graph on selected data
#for updating the graph we always need two lists i.e. one for each x and y axis
                #It takes all the selected variables as input
@app.callback(Output('my_graph_1', 'figure'),
				[Input('submit-button', 'n_clicks'),
                 Input('my_ticker_symbol-3', 'value'),
                 Input('my_ticker_symbol-2', 'value'),
				Input('my_ticker_symbol', 'value'),  
				Input('my_date_picker', 'start_date'),
                Input('my_date_picker', 'end_date')])
def update_graph_1(n_clicks,origin,dest,stock_ticker,startdate,enddate):
    #The date picker sends the date time format in Y/M/D H:M:S format
    #But we need only date so we take first 10 characters of string
    startdate = startdate[:10]
    enddate = enddate[:10]
    #We convert the string to date time format
    start_date = datetime.datetime.strptime(startdate,'%Y-%m-%d')
    end_date = datetime.datetime.strptime(enddate,'%Y-%m-%d')
    #We select only that dataframe that falls between the selected date range 
    filtered_df = data[data.FL_DATE.between(
        datetime.datetime.strftime(start_date, "%Y-%m-%d"),
        datetime.datetime.strftime(end_date, "%Y-%m-%d")
    )]
    #Now we scrutiny the data according to selected variables
    filtered_df1 = filtered_df[filtered_df['ORIGIN']==origin]
    filtered_df1 = filtered_df1[filtered_df1['DEST']==dest]
    #This code helps us to keep the graph updating
    #For each flight number selected this code finds the total cancelled flights in  an iteration over a month
    #it provides y list as that cancellation counts and  x list as months
    traces = []
    for i in stock_ticker:
        cCount = []
        df = filtered_df1[filtered_df1['UNIQUE_CARRIER']==i]
        GGHH = df.CANCELLATION_CODE.unique()
        codess= []
        jjjj = []
        for o in GGHH:
            dffff = df[df['CANCELLATION_CODE']==o]
            KKKKK = dffff.shape[0]
            codess.append(KKKKK)
            jjjj.append(i)
        
        months = df.FL_month.unique()
        for j in months:
            df1 = df[df['FL_month']==j]
            cCount.append(df1[df1.CANCELLED == 1].shape[0])
        #This code below enables us to see multiple selection output
        traces.append({'x':months, 'y':cCount, 'name': i})
        
    fig1 = {
		'data': traces,
		'layout': {'title':stock_ticker}
	}
    return fig1

#This callback allows us to see the cancellation division of the flight selected
#It takes same inputs as graph but return a division with a heading
@app.callback(Output('textonly','children'),
				[Input('submit-button', 'n_clicks'),
                 Input('my_ticker_symbol-3', 'value'),
                 Input('my_ticker_symbol-2', 'value'),
				Input('my_ticker_symbol', 'value'),  
				Input('my_date_picker', 'start_date'),
                Input('my_date_picker', 'end_date')])
def update_text_1(n_clicks,origin,dest,stock_ticker,startdate,enddate):
    #Initial operations are same as the graph function
    startdate = startdate[:10]
    enddate = enddate[:10]
    start_date = datetime.datetime.strptime(startdate,'%Y-%m-%d')
    end_date = datetime.datetime.strptime(enddate,'%Y-%m-%d')
    filtered_df = data[data.FL_DATE.between(
        datetime.datetime.strftime(start_date, "%Y-%m-%d"),
        datetime.datetime.strftime(end_date, "%Y-%m-%d")
    )]
    filtered_df1 = filtered_df[filtered_df['ORIGIN']==origin]
    filtered_df1 = filtered_df1[filtered_df1['DEST']==dest]
    #Here we need this code to keep div text working
    #It calculates all the cancellation reasons count from the selected data
    #Than it appends that string to the respective flight number
    #and than return a division consisting of a header with text as flight number->cancellation distrbution 
    traces = []
    for i in stock_ticker:
        traces1 = []
        df = filtered_df1[filtered_df1['UNIQUE_CARRIER']==i]
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
        months = df.FL_month.unique()
        count=0
        for j in GGHH:
            stri=stri+str(jjjj[count]+":"+str(codess[count]))+"%"
            count=count+1
        traces.append("Flight Number-"+str(i)+" "+" Cancellation Division"+stri)
        for i in traces:
            traces1.append(html.Div(html.H1(i)))    
    return html.Div(traces1)


if __name__ == '__main__':
    app.run_server()